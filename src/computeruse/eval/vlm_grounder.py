"""
The zero-shot VLM and fine-tuned-grounder ablation arms
(docs/research/gui-grounding-hypothesis.md). Both arms share this module --
the only difference between them is whether a LoRA adapter is loaded onto
the base model before generating, which is a one-line difference at the
call site, not two separate implementations that could silently drift.

Coordinate convention, verified against the literature before writing any
of this (2026-07-19): Qwen2-VL (the exact model this project uses, NOT
Qwen2.5-VL) natively predicts bounding-box coordinates normalized to
[0,1000) during its own pretraining -- see arxiv 2409.12191. That already
matches training/prepare_dataset.py's COORD_NORM_RANGE=1000 convention, so
the same system prompt (training/chat_format.py's SYSTEM_PROMPT) is valid
for both the zero-shot and fine-tuned arms; there is no prompt-format
mismatch to correct for. This matters because Qwen2.5-VL switched to
*absolute* pixel coordinates -- a lot of generic "how to prompt Qwen-VL for
grounding" tutorials online are written against 2.5-VL, and blindly
following one of those here would have silently taught the zero-shot arm
the wrong convention and made it look worse than it actually is, for a
reason having nothing to do with fine-tuning.

Functions here split into two groups, same as train_lora.py's own split:
- Pure logic (parse_point, normalize_bbox, is_hit_normalized): no GPU, no
  model, tested directly.
- Model-running (predict, evaluate_arm): needs the real ~4GB model and a
  GPU, only exercised from the Kaggle notebook, not by the test suite --
  same reasoning as train_lora.build_trainer.
"""

from __future__ import annotations

import copy
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from computeruse.eval.report import EvalRecord
from computeruse.training.chat_format import to_conversation
from computeruse.training.dataset import resolve_path
from computeruse.training.prepare_dataset import COORD_NORM_RANGE, TrainingExample

# The base model both VLM arms score. Duplicated from train_lora.MODEL_ID
# rather than imported, deliberately: train_lora pulls in peft and torch,
# and this module is otherwise dependency-light (the caller supplies an
# already-loaded model). test_model_id_matches_the_trained_base_model
# guards the two from drifting -- if they ever disagree, the zero-shot arm
# would be scoring a different base model than the fine-tuned arm was
# built on, which silently invalidates the H1 comparison.
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Matches "(423,551)", "(423, 551)", or the same embedded in surrounding
# text ("The point is (423,551).") -- generated text isn't guaranteed to be
# clean, especially zero-shot, so this deliberately doesn't anchor to the
# whole string.
_POINT_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
# A second pair directly after the first: "(x1,y1),(x2,y2)" is a box.
_BOX_TAIL_RE = re.compile(r"\s*,?\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
# Four numbers in one bracket, "(x1, y1, x2, y2)" or "[x1, y1, x2, y2]": also a
# box. The Kaggle smoke run (2026-10-12, 4 dev examples, no test example
# scored) showed the zero-shot model answering this way 3 times out of 4.
_BOX4_RE = re.compile(
    r"[\(\[]\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]"
)

# Room for Qwen2-VL's native grounding answer, which can name the object
# before the box ("<|object_ref_start|>Save<|object_ref_end|><|box_start|>
# (x1,y1),(x2,y2)<|box_end|>"). 16 tokens could cut that off mid-box. A
# fine-tuned answer ends at ~10 tokens with EOS, so this costs it nothing.
MAX_NEW_TOKENS = 48


def _parse(text: str) -> tuple[Optional[tuple[int, int]], str]:
    match = _POINT_RE.search(text)
    box4 = _BOX4_RE.search(text)
    if box4 is not None and (match is None or box4.start() < match.start()):
        x1, y1, x2, y2 = (int(g) for g in box4.groups())
        return ((x1 + x2) // 2, (y1 + y2) // 2), "box"
    if match is None:
        return None, "none"
    x, y = int(match.group(1)), int(match.group(2))
    tail = _BOX_TAIL_RE.match(text, match.end())
    if tail is None:
        return (x, y), "point"
    x2, y2 = int(tail.group(1)), int(tail.group(2))
    return ((x + x2) // 2, (y + y2) // 2), "box"


def parse_point(text: str) -> Optional[tuple[int, int]]:
    """Extract the first "(x,y)" pair from generated text. Returns None if
    none is found -- a real, expected outcome for a zero-shot model that
    hasn't learned to reliably follow the requested output format, and
    must be scored as a miss, not an error.

    A box -- "(x1,y1),(x2,y2)", the format Qwen2-VL was pretrained to ground
    in, with its <|box_start|> tokens stripped by decoding, or the
    "(x1, y1, x2, y2)" the zero-shot model actually wrote on Kaggle -- is
    scored at its center, as SeeClick-style evals score box-predicting models. Reading
    its first pair instead would score the zero-shot arm by the box's
    top-left corner: a miss on most small targets, for a reason that has
    nothing to do with whether it found the element (decided 2026-10-11,
    before any zero-shot or fine-tuned number on v3 was seen)."""
    return _parse(text)[0]


def answer_format(text: str) -> str:
    """"point", "box", or "none" -- which shape the model answered in. A
    diagnostic for the zero-shot arm: it shows whether the base model
    followed the (x,y) instruction or fell back to its pretraining format."""
    return _parse(text)[1]


def normalize_bbox(
    bbox_real: tuple[int, int, int, int],
    real_size: tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    """Same real->scaled->[0,1000) transform as prepare_dataset.
    normalized_center, applied to both corners instead of just the center,
    so a predicted normalized point can be checked against a normalized
    box directly -- comparing in the same space the model was actually
    trained (or, zero-shot, prompted) to predict in."""
    left, top, right, bottom = bbox_real
    real_w, real_h = real_size
    scaled_w = real_w / scale_x
    scaled_h = real_h / scale_y

    norm_left = round((left / scale_x) / scaled_w * COORD_NORM_RANGE)
    norm_top = round((top / scale_y) / scaled_h * COORD_NORM_RANGE)
    norm_right = round((right / scale_x) / scaled_w * COORD_NORM_RANGE)
    norm_bottom = round((bottom / scale_y) / scaled_h * COORD_NORM_RANGE)
    return norm_left, norm_top, norm_right, norm_bottom


def is_hit_normalized(
    point: Optional[tuple[int, int]], bbox_norm: tuple[int, int, int, int]
) -> bool:
    """Click-accuracy check in normalized [0,1000) space -- same semantics
    as eval.uia_only.is_hit (point inside ground-truth box), just operating
    on the coordinate space the VLM arms actually predict in rather than
    real screen pixels. A None point (unparseable generation) is always a
    miss, never a free pass."""
    if point is None:
        return False
    left, top, right, bottom = bbox_norm
    x, y = point
    return left <= x <= right and top <= y <= bottom


def center_distance_normalized(
    point: Optional[tuple[int, int]], bbox_norm: tuple[int, int, int, int]
) -> Optional[float]:
    """Euclidean distance from the predicted point to the ground-truth box
    *center*, in the same [0,1000) space, or None if nothing was parseable.

    Why this exists alongside is_hit_normalized: the median target in this
    dataset is 37x44 units, i.e. 0.16% of screen area. Binary hit-rate over
    targets that small is a nearly-all-zeros signal early in training -- a
    model that has genuinely learned "the Close button lives in the top
    right" and one that points at the taskbar both score 0.0, so hit-rate
    alone cannot tell you whether a run improved. Distance is continuous and
    does distinguish them, which makes it the right thing to steer dev
    decisions by. It is a *diagnostic*, not the reported metric: H1-H3 are
    stated in terms of click accuracy (is_hit_normalized), and that stays
    the headline number in the ablation.
    """
    if point is None:
        return None
    left, top, right, bottom = bbox_norm
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    x, y = point
    return ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5


@dataclass
class ArmResult:
    """Per-example outcome, kept around (not just a running total) so the
    caller can slice accuracy by app / app_richness / split afterward, per
    the hypothesis doc's "report sliced three ways, not one aggregate
    number" requirement."""

    example_id: str
    app: str
    split: str
    predicted_text: str
    predicted_point: Optional[tuple[int, int]]
    ground_truth_bbox_norm: tuple[int, int, int, int]
    hit: bool
    # None when the generation had no parseable point at all -- kept
    # distinct from "far away", which is a different failure.
    center_distance: Optional[float] = None


def predict(model, processor, tokenizer, image, instruction_example: TrainingExample) -> str:
    """Run one example through the model and return the raw generated
    text (not yet parsed). Mirrors the dev spot-check cell already
    verified in the Kaggle notebook: build the conversation, drop the
    ground-truth assistant turn, add the generation prompt, generate, and
    decode only the newly-generated tokens (not prompt + completion)."""
    prompt_only = to_conversation(instruction_example)[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_only, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, generation_config=greedy_config(model))
    return tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def greedy_config(model):
    """Greedy decoding, stated rather than inherited. Qwen2-VL-Instruct's
    shipped generation_config samples (do_sample=True with top_k=1, which
    is greedy in effect); making it explicit means a re-scored example
    gives the same answer, which is what lets an interrupted eval resume
    instead of starting over (eval/resumable.py)."""
    config = copy.deepcopy(model.generation_config)
    config.do_sample = False
    config.temperature = config.top_p = config.top_k = None
    config.max_new_tokens = MAX_NEW_TOKENS
    if config.pad_token_id is None and config.eos_token_id is not None:
        # otherwise generate() logs "Setting pad_token_id..." on every call
        eos = config.eos_token_id
        config.pad_token_id = eos[0] if isinstance(eos, list) else eos
    return config


def score_example(
    model,
    processor,
    tokenizer,
    example: TrainingExample,
    label_row: dict[str, Any],
    dataset_root: Path,
) -> ArmResult:
    """One example: predict, parse, score. `label_row` is the original
    labels.jsonl row (needed for bbox_real/real_size/scale_x/scale_y, which
    TrainingExample itself doesn't carry -- see
    prepare_dataset.TrainingExample's field list)."""
    from PIL import Image

    image = Image.open(resolve_path(dataset_root, example.image_path)).convert("RGB")
    raw_text = predict(model, processor, tokenizer, image, example)
    point = parse_point(raw_text)
    bbox_norm = normalize_bbox(
        tuple(label_row["element"]["bbox_real"]),
        tuple(label_row["real_size"]),
        label_row["scale_x"],
        label_row["scale_y"],
    )
    return ArmResult(
        example_id=example.id,
        app=example.app,
        split=example.split,
        predicted_text=raw_text,
        predicted_point=point,
        ground_truth_bbox_norm=bbox_norm,
        hit=is_hit_normalized(point, bbox_norm),
        center_distance=center_distance_normalized(point, bbox_norm),
    )


def evaluate_arm(
    model,
    processor,
    tokenizer,
    examples: list[TrainingExample],
    labels_by_id: dict[str, dict[str, Any]],
    dataset_root: Path,
) -> list[ArmResult]:
    """Run every example through `score_example`. `labels_by_id` maps
    TrainingExample.id -> its labels.jsonl row. GPU-only, not covered by the
    local test suite for the same reason train_lora.build_trainer isn't."""
    return [
        score_example(model, processor, tokenizer, e, labels_by_id[e.id], dataset_root)
        for e in examples
    ]


ZERO_SHOT = "zero_shot"


def load_adapters(base, adapter_dirs: dict[str, Path]):
    """Every adapter onto one copy of the base model, each under its arm
    name ("step-00700", "final"), so switching arms costs no reload.

    PEFT injects the LoRA layers into `base` in place: after this call,
    `base(...)` runs the active adapter too. The untouched base model is
    only reachable as `arm_active(model, ZERO_SHOT)` -- checked on a tiny
    Qwen2-VL to give bit-identical logits to the base before loading."""
    from peft import PeftModel

    names = list(adapter_dirs)
    model = PeftModel.from_pretrained(base, str(adapter_dirs[names[0]]), adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(str(adapter_dirs[name]), adapter_name=name)
    model.eval()
    return model


@contextmanager
def arm_active(model, arm: str):
    """Run the block as one arm: "zero_shot" turns every adapter off, which
    is exactly the base model (H1's control); any other name runs that
    adapter alone."""
    if arm == ZERO_SHOT:
        with model.disable_adapter():
            yield
    else:
        model.set_adapter(arm)
        yield


def result_to_json(result: ArmResult) -> dict[str, Any]:
    return asdict(result)


def result_from_json(row: dict[str, Any]) -> ArmResult:
    """Inverse of result_to_json. JSON has no tuples, so the two
    coordinate fields come back as lists and are turned back here --
    otherwise a reloaded result would compare unequal to the original."""
    point = row["predicted_point"]
    return ArmResult(
        **{
            **row,
            "predicted_point": tuple(point) if point is not None else None,
            "ground_truth_bbox_norm": tuple(row["ground_truth_bbox_norm"]),
        }
    )


def to_eval_records(
    results: list[ArmResult], arm: str, richness_by_app: dict[str, str]
) -> list[EvalRecord]:
    """Join ArmResult against apps.yaml's richness (via
    dataset.registry.load_registry -- pass `{c.name: c.richness for c in
    load_registry(...)}`) to get report.build_report's shape. `arm` is
    "zero_shot" or "fine_tuned" -- the two arms this module scores; the
    caller decides which by whether a LoRA adapter was loaded before
    calling evaluate_arm, not anything in ArmResult itself."""
    return [
        EvalRecord(
            example_id=r.example_id,
            arm=arm,
            app=r.app,
            app_richness=richness_by_app[r.app],
            split=r.split,
            hit=r.hit,
        )
        for r in results
    ]


def summarize(results: list[ArmResult]) -> dict[str, float]:
    """Aggregate accuracy overall and by app -- the app-level slice is what
    the hypothesis doc's rich-tree/weak-tree comparison (H2) is computed
    from; this function doesn't itself know app_richness, that join happens
    at the report-writing step against apps.yaml."""
    by_app: dict[str, list[bool]] = {}
    for r in results:
        by_app.setdefault(r.app, []).append(r.hit)

    summary = {"overall": sum(r.hit for r in results) / len(results) if results else 0.0}
    for app, hits in by_app.items():
        summary[app] = sum(hits) / len(hits)
    return summary


def summarize_diagnostics(results: list[ArmResult]) -> dict[str, float]:
    """The steering numbers, kept in a separate function from `summarize`
    so the reported-accuracy contract (app -> hit rate) stays uncontaminated
    by diagnostics that are not part of H1-H3.

    - `parse_rate`: fraction of generations containing a readable (x,y) at
      all. If this is low, nothing downstream means anything yet -- the
      model has an output-format problem, not a grounding problem, and
      that distinction is invisible in hit-rate alone (both score 0).
    - `median_center_distance` / `mean_center_distance`: over parseable
      predictions only, in [0,1000) units. For scale: a prediction 500
      units away is half a screen off; the median ground-truth box is ~40
      units across, so anything under ~20 is essentially a hit.
    """
    if not results:
        return {"parse_rate": 0.0, "median_center_distance": 0.0, "mean_center_distance": 0.0}

    distances = sorted(r.center_distance for r in results if r.center_distance is not None)
    parse_rate = len(distances) / len(results)
    if not distances:
        return {"parse_rate": 0.0, "median_center_distance": 0.0, "mean_center_distance": 0.0}

    mid = len(distances) // 2
    median = (
        distances[mid] if len(distances) % 2 else (distances[mid - 1] + distances[mid]) / 2
    )
    return {
        "parse_rate": parse_rate,
        "median_center_distance": median,
        "mean_center_distance": sum(distances) / len(distances),
    }


def format_counts(results: list[ArmResult]) -> dict[str, int]:
    """How many answers came back as a point, a box, or nothing parseable."""
    counts = {"point": 0, "box": 0, "none": 0}
    for r in results:
        counts[answer_format(r.predicted_text)] += 1
    return counts
