"""
Converts labels.jsonl (Phase 3's UIA-auto-labeled dataset) into the
(image, prompt, target) triples a LoRA-fine-tuned VLM trains on, and
partitions them into train/dev/test_same_app/test_held_out_app files using
the `split` field each sample already carries from collection time (see
docs/research/gui-grounding-dataset-design.md) -- this module never
re-derives splits, since doing so at this stage could silently contradict
the screenshot-level split discipline H3 (docs/research/gui-grounding-
hypothesis.md) depends on.

Deliberately pure-stdlib: no torch/transformers dependency. This step is
prompt/file formatting, not model code, so it runs and is tested on any
machine. The actual LoRA training run happens on free-tier cloud GPU
(Kaggle/Colab, per ADR-0003's $0-budget commitment); this module's output
(data/gui_grounding/splits/*.jsonl) is what gets uploaded there.

Chat-template wrapping (system prompt, special tokens, few-shot framing)
is deliberately NOT done here -- that's specific to whichever exact model
processor version the Kaggle notebook loads, and guessing at it without a
way to verify against the real model locally would be worse than leaving
it to the training script that actually has the model loaded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Qwen2-VL/SeeClick/UGround convention: represent a click point as text,
# normalized to a fixed 0-1000 integer range rather than raw pixels. This
# makes the target resolution-independent (a 4K and a 1080p screenshot of
# the same dialog get the same target) and lets a language-model-shaped
# output head "predict a click" by just generating a short token sequence.
COORD_NORM_RANGE = 1000

SPLITS = ("train", "dev", "test_same_app", "test_held_out_app")


@dataclass
class TrainingExample:
    """One (image, prompt, target) triple ready for SFT."""

    id: str
    image_path: str
    prompt: str
    target: str
    app: str
    split: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalized_center(
    bbox_real: list[int],
    real_size: tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> tuple[int, int]:
    """bbox_real is in real-screen pixel space, but the PNG on disk is the
    *scaled* image (perception/screenshot.py saves scaled_image, not the
    raw grab) -- so the target must go through the same real->scaled
    conversion Screenshot.to_real_coords inverts, before normalizing.
    Skipping this would train against coordinates that don't match the
    image the model is actually looking at."""
    left, top, right, bottom = bbox_real
    real_cx = (left + right) / 2
    real_cy = (top + bottom) / 2

    scaled_cx = real_cx / scale_x
    scaled_cy = real_cy / scale_y

    real_w, real_h = real_size
    scaled_w = real_w / scale_x
    scaled_h = real_h / scale_y

    norm_x = round((scaled_cx / scaled_w) * COORD_NORM_RANGE)
    norm_y = round((scaled_cy / scaled_h) * COORD_NORM_RANGE)
    return norm_x, norm_y


def to_training_example(sample: dict) -> TrainingExample:
    norm_x, norm_y = normalized_center(
        sample["element"]["bbox_real"],
        tuple(sample["real_size"]),
        sample["scale_x"],
        sample["scale_y"],
    )
    return TrainingExample(
        id=sample["id"],
        image_path=sample["screenshot_path"],
        prompt=sample["instruction"],
        target=f"({norm_x},{norm_y})",
        app=sample["app"],
        split=sample["split"],
    )


def load_labels(labels_path: Path) -> list[dict]:
    samples = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def prepare(dataset_root: Path) -> dict[str, list[TrainingExample]]:
    samples = load_labels(dataset_root / "labels.jsonl")
    by_split: dict[str, list[TrainingExample]] = {s: [] for s in SPLITS}
    for sample in samples:
        example = to_training_example(sample)
        by_split[example.split].append(example)
    return by_split


def write_splits(by_split: dict[str, list[TrainingExample]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, examples in by_split.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")


def main(dataset_root: Path = Path("data/gui_grounding")) -> None:
    by_split = prepare(dataset_root)
    write_splits(by_split, dataset_root / "splits")
    for split_name in SPLITS:
        print(f"{split_name}: {len(by_split[split_name])} examples")


if __name__ == "__main__":
    main()
