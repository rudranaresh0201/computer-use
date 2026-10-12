"""
ScreenSpot (Cheng et al., SeeClick, 2024): the public GUI-grounding
benchmark, scored with exactly the prediction, parsing and hit rule our own
ablation uses (vlm_grounder.predict / parse_point / is_hit_normalized).

Why it matters here: every number so far is on our own v3 splits. ScreenSpot
is other people's screenshots -- iOS/Android, macOS/Windows, four websites --
with instructions written by people ("close", "check the date and time"),
not generated from UIA names. It answers two questions our splits can't:
how the model compares on a benchmark others report, and whether training
on 10 Windows apps helps, hurts, or does nothing on GUIs it was never near.

Data: 1,272 examples in 3 parquet shards on the Hub (the repo moved from
rootsautomation/ScreenSpot to bevaya/ScreenSpot; the old name redirects).
bbox is [x1, y1, x2, y2] normalized to [0, 1] -- checked on the real files
2026-10-12, not taken from the dataset card.

Decided before any ScreenSpot number was seen (2026-10-12):
- Instructions are used verbatim as the prompt; our training prompts have
  the same shape ("Select Nine"), so there is no template to add.
- Images larger than the training pixel budget (1568x882) are scaled down
  to it, aspect ratio kept; smaller ones are left alone. Coordinates are
  normalized, so the answer does not depend on the scale. Both arms see
  identical pixels.
- The adapter is the one chosen on our dev split (eval/selection.json).
  Nothing about ScreenSpot feeds back into any choice.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from computeruse.eval.report import bootstrap_ci, paired_difference
from computeruse.eval.vlm_grounder import (
    ArmResult,
    center_distance_normalized,
    is_hit_normalized,
    parse_point,
    predict,
)
from computeruse.training.prepare_dataset import COORD_NORM_RANGE, TrainingExample

REPO_ID = "rootsautomation/ScreenSpot"
SHARDS = tuple(f"data/test-0000{i}-of-00003.parquet" for i in range(3))
EXPECTED_ROWS = 1272
TRAINING_PIXEL_BUDGET = 1568 * 882
PLATFORM_OF = {
    "ios": "mobile", "android": "mobile",
    "macos": "desktop", "windows": "desktop",
    "gitlab": "web", "shop": "web", "forum": "web", "tool": "web",
}
PLATFORMS = ("mobile", "desktop", "web")
TYPES = ("text", "icon")


@dataclass(frozen=True)
class Item:
    id: str
    instruction: str
    bbox_norm: tuple[int, int, int, int]  # [x1, y1, x2, y2] in our 0-1000 space
    data_type: str  # "text" | "icon"
    data_source: str  # "windows", "ios", "gitlab", ...
    image_bytes: bytes

    @property
    def platform(self) -> str:
        return PLATFORM_OF[self.data_source]

    @property
    def slice(self) -> str:
        return f"{self.platform}/{self.data_type}"


def items_from_rows(rows: list[dict[str, Any]], start_index: int = 0) -> list[Item]:
    """Rows as read from a shard -> Items. ids are the row's position across
    all shards plus its file name: stable across sessions (the shards are
    read in a fixed order), unique even though one screenshot carries
    several instructions."""
    items = []
    for offset, row in enumerate(rows):
        x1, y1, x2, y2 = (float(v) for v in row["bbox"])
        if not (0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0):
            raise ValueError(f"row {start_index + offset}: bbox {row['bbox']} is not normalized [x1, y1, x2, y2]")
        index = start_index + offset
        items.append(
            Item(
                id=f"{index:04d}_{Path(row['file_name']).stem}",
                instruction=row["instruction"],
                bbox_norm=tuple(round(v * COORD_NORM_RANGE) for v in (x1, y1, x2, y2)),
                data_type=row["data_type"],
                data_source=row["data_source"],
                image_bytes=row["image"]["bytes"],
            )
        )
    return items


def load(cache_dir: Path | None = None) -> list[Item]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    items: list[Item] = []
    for shard in SHARDS:
        path = hf_hub_download(REPO_ID, shard, repo_type="dataset", cache_dir=cache_dir)
        rows = pq.read_table(path).to_pylist()
        items += items_from_rows(rows, start_index=len(items))
    return items


def fit_to_budget(image, max_pixels: int = TRAINING_PIXEL_BUDGET):
    """Scale an image down to at most `max_pixels`, keeping its aspect
    ratio. Never scales up."""
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))))


def as_example(item: Item) -> TrainingExample:
    """The shape predict() takes: the image is passed separately, the prompt
    is the instruction as written."""
    return TrainingExample(
        id=item.id, image_path=f"{item.id}.png", prompt=item.instruction, target="",
        app=item.data_source, split=item.slice,
    )


def score_item(model, processor, item: Item) -> ArmResult:
    from PIL import Image

    image = fit_to_budget(Image.open(io.BytesIO(item.image_bytes)).convert("RGB"))
    raw = predict(model, processor, processor.tokenizer, image, as_example(item))
    point = parse_point(raw)
    return ArmResult(
        example_id=item.id,
        app=item.data_source,
        split=item.slice,
        predicted_text=raw,
        predicted_point=point,
        ground_truth_bbox_norm=item.bbox_norm,
        hit=is_hit_normalized(point, item.bbox_norm),
        center_distance=center_distance_normalized(point, item.bbox_norm),
    )


def _accuracy(results: list[ArmResult]) -> dict[str, Any]:
    hits = [r.hit for r in results]
    if not hits:
        return {"n": 0, "accuracy": None, "ci95": None}
    return {"n": len(hits), "accuracy": sum(hits) / len(hits), "ci95": list(bootstrap_ci(hits))}


def report(arms: dict[str, list[ArmResult]], baseline: str = "zero_shot") -> dict[str, Any]:
    """ScreenSpot's usual table -- platform x {text, icon}, plus each
    platform and the overall average -- for every arm, with 95% intervals,
    and every other arm's paired difference against the baseline."""
    slices = [f"{p}/{t}" for p in PLATFORMS for t in TYPES]
    out: dict[str, Any] = {"arms": {}, "vs_" + baseline: {}}
    for name, results in arms.items():
        table = {s: _accuracy([r for r in results if r.split == s]) for s in slices}
        for platform in PLATFORMS:
            table[platform] = _accuracy([r for r in results if r.split.startswith(platform + "/")])
        table["overall"] = _accuracy(results)
        out["arms"][name] = table
    base = {r.example_id: r for r in arms[baseline]}
    for name, results in arms.items():
        if name == baseline:
            continue
        diffs = {}
        for group in (*PLATFORMS, "overall"):
            chosen = [r for r in results if group == "overall" or r.split.startswith(group + "/")]
            diffs[group] = paired_difference(
                {r.example_id: r.hit for r in chosen}, {r.example_id: base[r.example_id].hit for r in chosen}
            )
        out["vs_" + baseline][name] = diffs
    return out
