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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from computeruse.training.chat_format import to_conversation
from computeruse.training.dataset import resolve_path
from computeruse.training.prepare_dataset import COORD_NORM_RANGE, TrainingExample

# Matches "(423,551)", "(423, 551)", or the same embedded in surrounding
# text ("The point is (423,551).") -- generated text isn't guaranteed to be
# clean, especially zero-shot, so this deliberately doesn't anchor to the
# whole string.
_POINT_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def parse_point(text: str) -> Optional[tuple[int, int]]:
    """Extract the first "(x,y)" pair from generated text. Returns None if
    none is found -- a real, expected outcome for a zero-shot model that
    hasn't learned to reliably follow the requested output format, and
    must be scored as a miss, not an error."""
    match = _POINT_RE.search(text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


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
    generated = model.generate(**inputs, max_new_tokens=16)
    return tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def evaluate_arm(
    model,
    processor,
    tokenizer,
    examples: list[TrainingExample],
    labels_by_id: dict[str, dict[str, Any]],
    dataset_root: Path,
) -> list[ArmResult]:
    """Run every example through `predict`, parse, and score. `labels_by_id`
    maps TrainingExample.id -> the original labels.jsonl row (needed for
    bbox_real/real_size/scale_x/scale_y, which TrainingExample itself
    doesn't carry -- see prepare_dataset.TrainingExample's field list).
    GPU-only, not covered by the local test suite for the same reason
    train_lora.build_trainer isn't."""
    from PIL import Image

    results: list[ArmResult] = []
    for example in examples:
        label_row = labels_by_id[example.id]
        image_path = resolve_path(dataset_root, example.image_path)
        image = Image.open(image_path).convert("RGB")

        raw_text = predict(model, processor, tokenizer, image, example)
        point = parse_point(raw_text)
        bbox_norm = normalize_bbox(
            tuple(label_row["element"]["bbox_real"]),
            tuple(label_row["real_size"]),
            label_row["scale_x"],
            label_row["scale_y"],
        )
        results.append(
            ArmResult(
                example_id=example.id,
                app=example.app,
                split=example.split,
                predicted_text=raw_text,
                predicted_point=point,
                ground_truth_bbox_norm=bbox_norm,
                hit=is_hit_normalized(point, bbox_norm),
            )
        )
    return results


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
