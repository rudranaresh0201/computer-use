"""
Who to trust, the tree or the grounder -- the candidate rules for a better
guard, fixed 2026-10-13 BEFORE the grounder's answers on the dev pages
(runs/guard_dev, seeds 100-159) exist.

The pre-registered llm_guarded (tree_llm.py + the grounder) scored 83.2 on the
test subset, but lost 6 points to vision alone on unnamed icons: on 15 of 50
the LLM picked SOME nameless button instead of saying NONE, overriding a
grounder that was right more often. Every candidate below keeps the LLM's NONE
as the hand-over to vision and changes only what happens when it picks.

    llm_guarded   the LLM's pick, else the grounder               (baseline, 1ad9836)
    llm_named     the pick only if it has a name, else the grounder
    llm_agree     the pick if it has a name OR the grounder's point lands inside it, else the grounder
    llm_snap      llm_named, but when vision answers, snap its point into the tree (compare.snap)

Selection rule (fixed now): highest mean over C1-C5 on the dev pages; within
1 point of the best, the simplest rule in the order listed. The chosen rule is
then scored ONCE on the 250-item test subset, and on Test C.
"""

from __future__ import annotations

from typing import Any

from computeruse.eval import compare
from computeruse.web.browser import WebElement

Point = tuple[int, int] | None
GUARDS = ("llm_guarded", "llm_named", "llm_agree", "llm_snap")


def picked(llm_row: dict[str, Any], elements: list[WebElement]) -> WebElement | None:
    """The element the LLM chose, looked up in the tree, or None for NONE."""
    if not llm_row.get("ref"):
        return None
    return next((e for e in elements if e.ref == llm_row["ref"]), None)


def decide(rule: str, llm_row: dict[str, Any], vlm_point: Point, elements: list[WebElement],
           viewport: tuple[int, int]) -> Point:
    pick = picked(llm_row, elements)
    pick_point = compare.center(pick, viewport) if pick is not None else None
    named = pick is not None and bool(pick.name.strip())
    if rule == "llm_guarded":
        return pick_point if pick is not None else vlm_point
    if rule == "llm_named":
        return pick_point if named else vlm_point
    if rule == "llm_agree":
        inside = (pick is not None and vlm_point is not None
                  and compare._inside(vlm_point, compare.norm_box(pick, viewport)))
        return pick_point if named or inside else vlm_point
    if rule == "llm_snap":
        return pick_point if named else compare.snap(vlm_point, elements, viewport)
    raise ValueError(rule)


def choose(dev_means: dict[str, float]) -> str:
    """The selection rule above: best dev mean; within 1 point of it, the earliest-listed rule."""
    best = max(dev_means.values())
    return next(rule for rule in GUARDS if rule in dev_means and dev_means[rule] >= best - 0.01)
