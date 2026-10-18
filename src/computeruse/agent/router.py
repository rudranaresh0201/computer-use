"""
The router: where on the screen an intent lands, and which channel put it there. The paper's
results, as code, so no prompt can skip them:

    arm "paper" (and "paper_blind", for brains that cannot see):
        a NAMED element the brain picked      -> tree      llm_named: the LLM's pick counts only if named
        a nameless element, or a description  -> eyes      ...else the grounder decides (85.2 vs 66.4, Test A)
        the eyes can't answer                 -> the brain's own x, y if it gave one, else ask again
        x, y for a click with no target       -> refused   say WHAT to click; where is the router's job

    the comparison arms (docs/where-we-are.md, the A/B/C/D comparison):
        "vision"  the brain's x, y only (screenshot-only computer use)
        "tree"    named elements, or a description with exactly one exact-name match (compare.tree_exact)
        "both"    named elements or the brain's own x, y: what the MCP tools allowed before this router

Nameless elements are never clicked through the tree: "tree first" with nameless or fuzzy
matches is the naive hybrid that lost 9.5-12 points to vision alone (Tests A, B).
Snapping the eyes' point into a tree box (Test B: +11.5 when the target is in the tree, -21.6
when it is not) is off unless asked for -- it was not the rule chosen on dev.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from .intent import POINTED, Intent

Box = tuple[int, int, int, int]  # left, top, right, bottom in screenshot pixels
Boxes = dict[str, tuple[str, str, Box]]  # ref -> (name, role, box)
Eyes = Callable[[str], tuple[Optional[tuple[int, int]], float]]  # description -> (point, seconds)

ARMS = ("paper", "paper_blind", "vision", "tree", "both")
# a description of empty space ("an empty spot in the middle of the board"): nothing for the grounder to find
EMPTY_PLACE = re.compile(r"\b(empty|blank|free|clear)\b[^.]*\b(spot|space|area|place|region|part|point|board|canvas)\b",
                         re.IGNORECASE)
SNAP_FRACTION = 0.02  # snap radius as a share of the screenshot width (compare.SNAP_RADIUS = 20/1000)


@dataclass
class Grounded:
    """What the Session should do: act on `element`, or at x, y; and how that was decided."""

    action: str
    channel: str  # api | tree | eyes | vision
    element: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    to_x: Optional[int] = None
    to_y: Optional[int] = None
    why: str = ""
    eyes_seconds: float = 0.0


class Unroutable(Exception):
    """The intent can't be placed under this arm's rules; the message is for the brain."""


def center(box: Box) -> tuple[int, int]:
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


_IGNORED = {"click", "press", "tap", "select", "choose", "the", "a", "an", "on", "in", "of", "to", "button", "icon"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _IGNORED}


def tree_exact(description: str, boxes: Boxes) -> Optional[str]:
    """The ref of the ONE element whose name says exactly what was asked (compare.tree_exact's rule,
    the guarded hybrid chosen on dev); None when there is no such element or more than one."""
    wanted = _tokens(description)
    if not wanted:
        return None
    same = [ref for ref, (name, _, _) in boxes.items() if name and _tokens(name) == wanted]
    return same[0] if len(same) == 1 else None


def snap(point: tuple[int, int], boxes: Boxes, width: int) -> tuple[int, int]:
    """compare.snap in pixels: into the smallest box holding the point, else the nearest box within
    the radius, else unchanged."""
    px, py = point
    holding = [b for _, _, b in boxes.values() if b[0] <= px <= b[2] and b[1] <= py <= b[3]]
    if holding:
        return center(min(holding, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
    best, best_d = None, float("inf")
    for _, _, b in boxes.values():
        dx, dy = max(b[0] - px, 0, px - b[2]), max(b[1] - py, 0, py - b[3])
        d = (dx * dx + dy * dy) ** 0.5
        if d < best_d:
            best, best_d = b, d
    return center(best) if best is not None and best_d <= SNAP_FRACTION * width else point


@dataclass
class Router:
    arm: str = "paper"
    eyes: Optional[Eyes] = None
    use_snap: bool = False

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm {self.arm!r} is not one of {ARMS}")

    def route(self, intent: Intent, boxes: Boxes, width: int) -> Grounded:
        a = intent.action
        if a in ("key", "wait"):
            return Grounded(a, "api")
        if a == "type" and not (intent.element or intent.target or intent.x is not None):
            return Grounded(a, "api", why="into the focused field")
        if a == "scroll" and not (intent.element or intent.target or intent.x is not None):
            return Grounded(a, "api")
        if a == "drag":
            return self._drag(intent, boxes, width)
        if a in POINTED or a in ("type", "scroll"):
            element, point, channel, why, secs = self._place(intent.element, intent.target, intent.x, intent.y,
                                                              boxes, width, free_xy=False)
            if element:
                return Grounded(a, channel, element=element, why=why)
            return Grounded(a, channel, x=point[0], y=point[1], why=why, eyes_seconds=secs)
        raise Unroutable(f"{a} is not an action the router places")

    # --- internals ------------------------------------------------------------------

    def _place(self, element, target, x, y, boxes: Boxes, width: int, free_xy: bool):
        """-> (element ref or None, point or None, channel, why, eyes seconds) for one place on screen.
        free_xy: the brain's own x, y are allowed here (drawing), whatever the arm."""
        arm = self.arm
        has_xy = x is not None and y is not None
        if arm == "vision":
            if not has_xy:
                raise Unroutable("this arm acts by x, y only: give x and y in screenshot pixels")
            return None, (x, y), "vision", "the brain's reading of the screenshot", 0.0

        named = element is not None and element in boxes and bool(boxes[element][0])
        if element is not None and element not in boxes:
            raise Unroutable(f"no element {element!r} in the latest observation; observe again")
        if named:
            return element, None, "tree", f'named element "{boxes[element][0]}"', 0.0

        if arm == "tree":
            ref = tree_exact(target or "", boxes) if target else None
            if ref:
                return ref, None, "tree", f'exactly one element named "{boxes[ref][0]}"', 0.0
            raise Unroutable("the tree has no single element named that; pick a named element from the list")

        if arm == "both":
            if element is not None:
                raise Unroutable(f"{element} has no name: the tree can't say which element it is; give x, y")
            if has_xy:
                return None, (x, y), "vision", "the brain's reading of the screenshot", 0.0
            raise Unroutable("give a named element or x, y")

        # the paper arms: a nameless pick or a description goes to our grounder
        description = target
        if description is None:
            if element is not None:
                raise Unroutable(f"{element} has no name: describe it in target (e.g. 'the save icon') "
                                 "so the grounder can find it")
            if has_xy and free_xy:
                return None, (x, y), "vision", "the brain's own x, y (drawing)", 0.0
            raise Unroutable("say WHAT to act on: a named element, or a target description; "
                             "the router decides where")
        if arm == "paper" and EMPTY_PLACE.search(description):
            # nothing there for the grounder to find: it learned things ("click Save"), not empty space, and on
            # Tinkercad it put every "empty spot on the board" on the toolbar. The seeing brain picks the place.
            if has_xy:
                return None, (x, y), "vision", "an empty place: the brain's own x, y", 0.0
            raise Unroutable("an empty place has nothing for the grounder to find: give x, y yourself "
                             "(to_x, to_y for where a drag ends), in screenshot pixels")
        point, seconds = self.eyes(description) if self.eyes else (None, 0.0)
        if point is not None:
            if self.use_snap:
                snapped = snap(point, boxes, width)
                if snapped != point:
                    return None, snapped, "eyes", f"grounder found {description!r}, snapped into a tree box", seconds
            return None, point, "eyes", f"grounder found {description!r}", seconds
        if has_xy and arm == "paper":
            return None, (x, y), "vision", f"grounder had no answer for {description!r}; the brain's x, y", seconds
        why = "the grounder is not running" if self.eyes is None else "the grounder found nothing readable"
        raise Unroutable(f"{why} for {description!r}; describe it another way or use a named element")

    def _drag(self, intent: Intent, boxes: Boxes, width: int) -> Grounded:
        start_el, start, c1, _, s1 = self._place(intent.element, intent.target, intent.x, intent.y, boxes, width,
                                                 free_xy=True)
        end_el, end, c2, _, s2 = self._place(None, intent.to_target, intent.to_x, intent.to_y, boxes, width,
                                             free_xy=True)
        if start_el:
            start = center(boxes[start_el][2])
        if end_el:
            end = center(boxes[end_el][2])
        channel = c1 if c1 == c2 else f"{c1}+{c2}"
        return Grounded("drag", channel, x=start[0], y=start[1], to_x=end[0], to_y=end[1],
                        why=f"from {c1} to {c2}", eyes_seconds=s1 + s2)
