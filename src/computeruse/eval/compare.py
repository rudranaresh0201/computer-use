"""
Tree vs vision vs both, on the sandbox test set (sandbox/build.py) --
hypotheses H4-H6 in docs/research/gui-grounding-hypothesis.md.

Every item is one screen, one instruction, one true box that came from
neither method. Each arm answers with a point in the 0-1000 space; a hit is
a point inside the true box -- the same rule as every other eval here.

    tree      match the instruction against element names in the
              accessibility tree; click the best match's center. No match
              above the threshold -> no answer (a miss). The classic way a
              tree is used; an LLM choosing from the element list is the
              stronger tree baseline and is a separate, later arm.
    vlm       the fine-tuned grounder, from pixels alone (and its zero-shot
              base, as a reference)
    fallback  tree's answer when it has one, else the vlm's point
    snap      the vlm's point, moved to the center of the interactive tree
              element it falls in (the smallest one), or the nearest one
              within SNAP_RADIUS; left alone if none -- vision decides
              where, the tree supplies an exact box

Fixed before any result (2026-10-12): the token rules, the 0.6 match
threshold, the snap radius, and the arms above.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Optional

from computeruse.eval.report import bootstrap_ci, paired_difference
from computeruse.eval.vlm_grounder import is_hit_normalized
from computeruse.web.browser import INTERACTIVE_ROLES, WebElement

MATCH_THRESHOLD = 0.6
SNAP_RADIUS = 20  # in 0-1000 units, ~30 px on the 1568-wide screen
CONDITIONS = ("C1", "C2", "C3", "C4", "C5")
ARMS = ("tree", "vlm_zero_shot", "vlm", "fallback", "snap")

# Words that say what to do, not which element: a name matcher ignores them.
_IGNORED = {
    "click", "press", "tap", "select", "choose", "hit", "open", "go", "show", "make", "get", "give", "put",
    "the", "a", "an", "on", "in", "of", "to", "for", "my", "this", "it", "these", "that", "please",
    "button", "icon", "panel", "settings", "section", "item", "image", "me", "i", "need",
}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _IGNORED}


def name_score(instruction: str, name: str) -> float:
    if not name:
        return 0.0
    asked, named = _tokens(instruction), _tokens(name)
    if named and named <= asked:
        return 1.0  # every word of the name is in the instruction
    union = asked | named
    jaccard = len(asked & named) / len(union) if union else 0.0
    ratio = SequenceMatcher(None, " ".join(sorted(asked)), " ".join(sorted(named))).ratio()
    return max(jaccard, ratio)


def interactive(elements: list[WebElement]) -> list[WebElement]:
    return [e for e in elements if e.role in INTERACTIVE_ROLES and e.box and e.box[2] > 0 and e.box[3] > 0]


def tree_answer(instruction: str, elements: list[WebElement]) -> Optional[WebElement]:
    """Best-named interactive element, first in tree order on a tie; None below the threshold."""
    best, best_score = None, 0.0
    for element in interactive(elements):
        score = name_score(instruction, element.name)
        if score > best_score:
            best, best_score = element, score
    return best if best_score >= MATCH_THRESHOLD else None


def norm_box(element: WebElement, viewport: tuple[int, int]) -> tuple[float, float, float, float]:
    x, y, w, h = element.box
    vw, vh = viewport
    return x / vw * 1000, y / vh * 1000, (x + w) / vw * 1000, (y + h) / vh * 1000


def center(element: WebElement, viewport: tuple[int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = norm_box(element, viewport)
    return round((x1 + x2) / 2), round((y1 + y2) / 2)


def snap(point: Optional[tuple[int, int]], elements: list[WebElement], viewport: tuple[int, int]) -> Optional[tuple[int, int]]:
    if point is None:
        return None
    px, py = point
    candidates = interactive(elements)
    containing = [e for e in candidates if _inside(point, norm_box(e, viewport))]
    if containing:
        smallest = min(containing, key=lambda e: e.box[2] * e.box[3])
        return center(smallest, viewport)
    nearest, nearest_d = None, float("inf")
    for e in candidates:
        x1, y1, x2, y2 = norm_box(e, viewport)
        dx, dy = max(x1 - px, 0, px - x2), max(y1 - py, 0, py - y2)
        d = (dx * dx + dy * dy) ** 0.5
        if d < nearest_d:
            nearest, nearest_d = e, d
    return center(nearest, viewport) if nearest is not None and nearest_d <= SNAP_RADIUS else point


def _inside(point: tuple[int, int], box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def score_item(
    item: dict[str, Any],
    elements: list[WebElement],
    vlm_point: Optional[tuple[int, int]],
    vlm_zero_shot_point: Optional[tuple[int, int]],
    viewport: tuple[int, int],
) -> dict[str, Any]:
    """Every arm's answer and hit for one item."""
    truth = tuple(item["gt_box_norm"])
    matched = tree_answer(item["instruction"], elements)
    tree_point = center(matched, viewport) if matched is not None else None
    points = {
        "tree": tree_point,
        "vlm_zero_shot": vlm_zero_shot_point,
        "vlm": vlm_point,
        "fallback": tree_point if tree_point is not None else vlm_point,
        "snap": snap(vlm_point, elements, viewport),
    }
    return {
        "id": item["id"], "condition": item["condition"],
        "points": {arm: list(p) if p else None for arm, p in points.items()},
        "hits": {arm: is_hit_normalized(p, truth) for arm, p in points.items()},
        "tree_matched": matched.name if matched is not None else None,
    }


def score_vlm(model, processor, item: dict[str, Any], root) -> "ArmResult":
    """One test item through the grounder: the screenshot, the instruction
    as written, the same predict/parse/hit rule as every eval here."""
    from pathlib import Path

    from PIL import Image

    from computeruse.eval.vlm_grounder import ArmResult, center_distance_normalized, parse_point, predict
    from computeruse.training.prepare_dataset import TrainingExample

    image = Image.open(Path(root) / item["image"]).convert("RGB")
    example = TrainingExample(item["id"], item["image"], item["instruction"], "", item["condition"], item["condition"])
    raw = predict(model, processor, processor.tokenizer, image, example)
    point = parse_point(raw)
    truth = tuple(item["gt_box_norm"])
    return ArmResult(item["id"], item["condition"], item["condition"], raw, point, truth,
                     is_hit_normalized(point, truth), center_distance_normalized(point, truth))


def _cell(hits: list[bool]) -> dict[str, Any]:
    if not hits:
        return {"n": 0, "accuracy": None, "ci95": None}
    return {"n": len(hits), "accuracy": sum(hits) / len(hits), "ci95": list(bootstrap_ci(hits))}


def report(scored: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        row = {c: _cell([s["hits"][arm] for s in scored if s["condition"] == c]) for c in CONDITIONS}
        accuracies = [row[c]["accuracy"] for c in CONDITIONS if row[c]["accuracy"] is not None]
        row["mean"] = {"accuracy": sum(accuracies) / len(accuracies) if accuracies else None}
        table[arm] = row

    def diff(a: str, b: str, conditions: tuple[str, ...]) -> dict[str, Any]:
        chosen = [s for s in scored if s["condition"] in conditions]
        return paired_difference({s["id"]: s["hits"][a] for s in chosen}, {s["id"]: s["hits"][b] for s in chosen})

    return {
        "table": table,
        "hypotheses": {
            "H4 C1: tree - vlm (>= 0 expected)": diff("tree", "vlm", ("C1",)),
            "H5a C2: vlm - tree (> 0 expected)": diff("vlm", "tree", ("C2",)),
            "H5b C4: vlm - tree (> 0 expected)": diff("vlm", "tree", ("C4",)),
            "H6a all: snap - tree": diff("snap", "tree", CONDITIONS),
            "H6b all: snap - vlm": diff("snap", "vlm", CONDITIONS),
            "H6c all: fallback - vlm": diff("fallback", "vlm", CONDITIONS),
        },
    }
