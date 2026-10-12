"""
Recompute headline intervals with a clustered bootstrap (eval/report.py,
clustered_paired_difference): examples on one screenshot resampled together,
then -- the strictest view -- whole apps resampled together.

    python scripts/clustered_intervals.py test-b <eval/results/step-01179.jsonl>
    python scripts/clustered_intervals.py h1-h3 <eval/results/>     (step-01179.jsonl vs zero_shot.jsonl)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from computeruse.eval import real_tree
from computeruse.eval.report import clustered_paired_difference, paired_difference

LABELS = Path("data/gui_grounding/labels.jsonl")


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def three_ways(a: dict[str, bool], b: dict[str, bool], screenshot: dict[str, str], app: dict[str, str]) -> dict:
    row = paired_difference(a, b)
    by_shot = clustered_paired_difference(a, b, screenshot)
    by_app = clustered_paired_difference(a, b, app)
    return {"n": row["n"], "diff": round(row["diff"] * 100, 1),
            "item_ci": [round(x * 100, 1) for x in row["ci95"]],
            "screenshot_ci": [round(x * 100, 1) for x in by_shot["ci95"]], "screenshots": by_shot["clusters"],
            "app_ci": [round(x * 100, 1) for x in by_app["ci95"]], "apps": by_app["clusters"]}


def test_b(predictions_path: Path) -> dict:
    labels = _read(LABELS)
    shot_of = {row["id"]: row["screenshot_id"] for row in labels}
    scored = real_tree.run(labels, _read(predictions_path))
    out = {}
    for split in ("test_same_app", "test_held_out_app"):
        for condition in real_tree.CONDITIONS:
            rows = [s for s in scored if s["split"] == split and s["condition"] == condition]
            shot = {s["example_id"]: shot_of[s["example_id"]] for s in rows}
            app = {s["example_id"]: s["app"] for s in rows}
            for a in ("guarded", "snap", "fallback"):
                hits_a = {s["example_id"]: s["hits"][a] for s in rows}
                hits_v = {s["example_id"]: s["hits"]["vlm"] for s in rows}
                out[f"{split}|{condition}|{a}-vlm"] = three_ways(hits_a, hits_v, shot, app)
    return out


def h1_h3(results_dir: Path) -> dict:
    shot_of = {row["id"]: row["screenshot_id"] for row in _read(LABELS)}
    tuned = _read(results_dir / "step-01179.jsonl")
    zero = {r["example_id"]: r["hit"] for r in _read(results_dir / "zero_shot.jsonl")}
    out = {}
    for split, name in (("test_same_app", "H1"), ("test_held_out_app", "H3")):
        rows = [r for r in tuned if r["split"] == split]
        out[f"{name} {split}|fine_tuned-zero_shot"] = three_ways(
            {r["example_id"]: r["hit"] for r in rows}, {r["example_id"]: zero[r["example_id"]] for r in rows},
            {r["example_id"]: shot_of[r["example_id"]] for r in rows}, {r["example_id"]: r["app"] for r in rows})
    return out


if __name__ == "__main__":
    which, path = sys.argv[1], Path(sys.argv[2])
    result = {"test-b": test_b, "h1-h3": h1_h3}[which](path)
    for key, row in result.items():
        print(f"{key:48s} n={row['n']:4d} diff={row['diff']:+6.1f}  item{row['item_ci']}  "
              f"screenshot{row['screenshot_ci']} ({row['screenshots']})  app{row['app_ci']} ({row['apps']})")
