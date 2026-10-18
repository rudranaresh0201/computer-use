"""
The eyes: our fine-tuned grounder (Qwen2-VL-2B + the LoRA adapter from the paper's runs) finds a
described element in the pixels -- the channel for things the accessibility tree does not name.

    session.locate("the Brush tool")  ->  (412, 88) in the screenshot the model just saw

It answers in [0,1000) normalized coordinates (training/chat_format.py's convention), so the
screenshot can be shrunk before it goes in -- fewer visual tokens, faster on a CPU -- and the
answer still maps back onto the full screenshot. scripts/bench_grounder_cpu.py measures what
that shrinking costs in accuracy, per long edge.

The question goes through vlm_grounder.predict, the same path the eval scored, so the live tool
asks exactly the way the model was trained to be asked.
"""

from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ADAPTER_REPO = "rudra201/computer-use-grounder"
ADAPTER_PATH = "adapters/step-01179"  # the checkpoint the eval chose on dev (eval/selection.json)
LONG_EDGE = int(os.environ.get("CU_EYES_EDGE", "1024"))  # chosen from the CPU benchmark
INT8 = os.environ.get("CU_EYES_INT8", "1") == "1"


@dataclass
class Located:
    point: Optional[tuple[int, int]]  # in the caller's image pixels; None = no readable answer
    raw: str  # what the model wrote, for the trace
    seconds: float


def fit(size: tuple[int, int], long_edge: int) -> tuple[int, int]:
    """The size to shrink an image to so its long edge is at most `long_edge` (never enlarged)."""
    w, h = size
    scale = min(1.0, long_edge / max(w, h))
    return max(28, round(w * scale)), max(28, round(h * scale))


def to_pixels(point_norm: tuple[int, int], size: tuple[int, int]) -> tuple[int, int]:
    """[0,1000) normalized -> pixels of an image of `size`, clamped inside it."""
    from .training.prepare_dataset import COORD_NORM_RANGE

    w, h = size
    x = round(point_norm[0] / COORD_NORM_RANGE * w)
    y = round(point_norm[1] / COORD_NORM_RANGE * h)
    return max(0, min(x, w - 1)), max(0, min(y, h - 1))


def adapter_dir() -> Path:
    from huggingface_hub import snapshot_download

    root = snapshot_download(ADAPTER_REPO, allow_patterns=[f"{ADAPTER_PATH}/*"])
    return Path(root) / ADAPTER_PATH


def quantize_int8(model):
    """Every Linear layer to 8-bit weights (PyTorch's built-in dynamic quantization), one layer at a
    time so memory never holds two full copies; the rest goes to float32, which those layers expect.
    Why: in bfloat16 the model holds ~5.4 GB while it runs, more than this 7.5 GB laptop has free
    beside Chrome and an editor (measured 2026-10-17: it paged out and stalled). And a CPU without
    native bfloat16 math (this Core Ultra 5) runs 8-bit integer matmuls far better."""
    from torch import nn
    from torch.ao.nn.quantized.dynamic import Linear as QLinear
    from torch.ao.quantization import default_dynamic_qconfig

    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.Linear:
                child.float()
                child.qconfig = default_dynamic_qconfig
                setattr(parent, name, QLinear.from_float(child))
    return model.float()


class Grounder:
    """Loads once, then answers locate() calls from one thread."""

    def __init__(self, long_edge: int = LONG_EDGE, threads: Optional[int] = None, int8: bool = INT8,
                 device: str = "cpu") -> None:
        # on a GPU: the eval's own setup (training/train_lora's precision, no quantization), so the
        # live eyes are the model that scored 46.5 -- the CPU int8 path measured 1/12 (G11)
        self.long_edge, self.threads, self.device = long_edge, threads, device
        self.int8 = int8 and device == "cpu"
        self.model = self.processor = self.tokenizer = None
        self.load_seconds = 0.0
        self._lock = threading.Lock()

    def load(self) -> "Grounder":
        with self._lock:  # a warm-up thread and the first locate() may both get here
            if self.model is None:
                self._load()
        return self

    def _load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from .eval.vlm_grounder import MODEL_ID

        start = time.perf_counter()
        if self.threads:
            torch.set_num_threads(self.threads)
        dtype = torch.bfloat16
        if self.device != "cpu":
            from .training.train_lora import resolve_precision, torch_dtype_for

            dtype = torch_dtype_for(resolve_precision())
        base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="sdpa")
        if self.device != "cpu":
            base = base.to(self.device)
        # merged: one set of weights, no adapter detour on every layer at inference
        self.model = PeftModel.from_pretrained(base, str(adapter_dir())).merge_and_unload().eval()
        if self.int8:
            self.model = quantize_int8(self.model)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.tokenizer = self.processor.tokenizer
        self.load_seconds = time.perf_counter() - start

    def locate(self, image, instruction: str) -> Located:
        """`image`: a PIL image or JPEG/PNG bytes. Returns the point in THAT image's pixels."""
        import torch
        from PIL import Image

        from .eval.vlm_grounder import parse_point, predict
        from .training.prepare_dataset import TrainingExample

        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image))
        image = image.convert("RGB")
        small = image.resize(fit(image.size, self.long_edge), Image.LANCZOS)
        self.load()
        ask = TrainingExample(id="live", image_path="", prompt=instruction, target="", app="", split="live")
        start = time.perf_counter()
        with torch.inference_mode():
            raw = predict(self.model, self.processor, self.tokenizer, small, ask)
        point = parse_point(raw)
        return Located(to_pixels(point, image.size) if point else None, raw, time.perf_counter() - start)
