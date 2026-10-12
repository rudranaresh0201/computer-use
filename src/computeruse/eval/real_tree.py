"""
Test B: tree vs vision vs both on REAL Windows screens, with the UIA tree
damaged on purpose -- no GPU, no new predictions.

The grounder never reads the tree, so its answers on the v3 eval items
(eval/results/step-01179.jsonl from the Kaggle eval) are the same whatever
the tree contains. That lets us re-run the comparison on real pixels and
real UIA under conditions we impose:

    intact    the tree as UIA reported it (the instruction names the target,
              so name matching should be ~perfect -- the v3 baseline)
    nameless  the target stays in the tree, with its name blanked -- a real
              screen version of sandbox C2
    absent    the target is removed from the tree -- a real screen version of C4

Arms (eval/compare.py, same rules as the sandbox test): tree (name matching),
vlm (the dev-chosen adapter), fallback (tree, else vlm), snap (vlm point
moved into the tree element it lands in, or the nearest within 20/1000).

Fixed before running (2026-10-12):
- the tree of a screenshot is every element labeled on it in labels.jsonl
  (UIA's named elements; the raw trees, with unnamed and decorative
  elements, were not saved -- so this tree is cleaner than a real one);
- every element counts as clickable for snapping;
- the headline is the two test splits; dev is reported separately, since
  200 dev items helped choose the adapter.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from computeruse.eval import compare
from computeruse.eval.report import bootstrap_ci, paired_difference
from computeruse.eval.vlm_grounder import normalize_bbox
from computeruse.web.browser import WebElement

NORM = (1000, 1000)  # boxes are built in 0-1000 units, so the "viewport" is the unit square
CONDITIONS = ("intact", "nameless", "absent")
ARMS = ("tree", "vlm", "fallback", "snap", "guarded")
SPLITS = ("test_same_app", "test_held_out_app", "dev")


def element_of(row: dict[str, Any]) -> WebElement:
    x1, y1, x2, y2 = normalize_bbox(tuple(row["element"]["bbox_real"]), tuple(row["real_size"]), row["scale_x"], row["scale_y"])
    return WebElement(ref=row["id"], role="button", name=row["element"]["name"], box=(x1, y1, x2 - x1, y2 - y1))


def damage(elements: list[WebElement], target_id: str, condition: str) -> list[WebElement]:
    if condition == "intact":
        return elements
    if condition == "nameless":
        return [WebElement(e.ref, e.role, "", e.box) if e.ref == target_id else e for e in elements]
    if condition == "absent":
        return [e for e in elements if e.ref != target_id]
    raise ValueError(condition)


def run(labels: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["id"]: row for row in labels}
    screens: dict[str, list[WebElement]] = defaultdict(list)
    for row in labels:
        screens[row["screenshot_id"]].append(element_of(row))
    scored = []
    for prediction in predictions:
        row = by_id[prediction["example_id"]]
        point = tuple(prediction["predicted_point"]) if prediction["predicted_point"] else None
        for condition in CONDITIONS:
            item = {"id": f"{row['id']}|{condition}", "condition": condition, "instruction": row["instruction"],
                    "gt_box_norm": prediction["ground_truth_bbox_norm"]}
            result = compare.score_item(item, damage(screens[row["screenshot_id"]], row["id"], condition), point, None, NORM)
            result.update({"split": prediction["split"], "app": row["app"], "example_id": row["id"]})
            scored.append(result)
    return scored


def report(scored: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLITS:
        rows = [s for s in scored if s["split"] == split]
        table = {}
        for condition in CONDITIONS:
            chosen = [s for s in rows if s["condition"] == condition]
            table[condition] = {}
            for arm in ARMS:
                hits = [s["hits"][arm] for s in chosen]
                table[condition][arm] = {"n": len(hits), "accuracy": sum(hits) / len(hits) if hits else None,
                                         "ci95": list(bootstrap_ci(hits)) if hits else None}

            def diff(a, b):
                return paired_difference({s["example_id"]: s["hits"][a] for s in chosen},
                                         {s["example_id"]: s["hits"][b] for s in chosen})

            table[condition]["snap_minus_vlm"] = diff("snap", "vlm")
            table[condition]["fallback_minus_vlm"] = diff("fallback", "vlm")
            table[condition]["guarded_minus_vlm"] = diff("guarded", "vlm")
        out[split] = table
    return out
