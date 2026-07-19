"""
The UIA-only ablation arm (docs/research/gui-grounding-hypothesis.md).

Scope, read before trusting a number out of this module: this arm answers
"given that UIA's tree already contains the target element, can name-based
text matching alone (no vision) correctly resolve which element an
instruction refers to" -- it is NOT a measure of how often UIA can find any
conceivable click target in the wild. Our dataset only contains elements
the UIA tree could see and that survived collector.filter_elements, so the
correct element for any given instruction is always present in this arm's
candidate pool by construction. Weak-tree apps don't fail this arm because
matching is hard -- they fail to have full click-target coverage in the
first place (Paint's canvas, GIMP's panels have no UIA representation at
all), which is a coverage gap the dataset's app_richness field already
represents structurally, not something a per-example text matcher can
detect from data that's already been UIA-filtered.

What this arm DOES measure, honestly: whether duplicate/ambiguous element
names (the 2026-07-18 Windows Terminal tab collision, see docs/journal.md)
can be resolved by text alone. Real signal, narrower claim than "UIA vs.
vision in general" -- that broader comparison is what the app_richness
slice (rich-tree vs. weak-tree) in the full ablation report is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from computeruse.dataset.collector import (
    _FALLBACK_TEMPLATES,
    _INSTRUCTION_TEMPLATES,
    _disambiguated_names,
)
from computeruse.perception.uia import UIElement


@dataclass
class UiaOnlyPrediction:
    """Result of matching one instruction against a screenshot's candidate
    UIA elements. `center` is None when the arm couldn't confidently
    resolve a single element -- callers should treat that as a miss (0
    accuracy for that example), not an error."""

    center: Optional[tuple[int, int]]
    matched_name: Optional[str]
    ambiguous: bool


def _candidate_elements(rows: list[dict]) -> list[UIElement]:
    """Reconstruct lightweight UIElements from labels.jsonl rows sharing
    one screenshot_id -- the closest proxy we have to "the UIA tree for
    this screenshot" without a live window to re-query."""
    return [
        UIElement(
            name=row["element"]["name"],
            control_type=row["element"]["control_type"],
            rect=tuple(row["element"]["bbox_real"]),
            automation_id=row["element"].get("automation_id"),
        )
        for row in rows
    ]


def predict(instruction: str, screenshot_rows: list[dict]) -> UiaOnlyPrediction:
    """Resolve `instruction` to a click point using only text matching
    against the UIA elements captured for the same screenshot.

    Matching works by *exact reconstruction*: for each candidate, try every
    template registered for its control_type (collector._INSTRUCTION_
    TEMPLATES) formatted with its disambiguated display name, and check
    whether the result is byte-identical to `instruction`. This is
    deliberately not loose substring containment -- an earlier version of
    this matcher used "does the name appear anywhere in the instruction",
    which broke on real data: Paint has an actual toolbar element named
    "Select", and several templates' boilerplate literally starts with the
    word "Select" ("Select {name} from the list"), so the boilerplate text
    itself matched the unrelated "Select" element instead of the true
    target ("Gray", "Orange", ...). Exact reconstruction can't be fooled by
    this, since "Select Select"/"Click Select"/etc. (Select's own possible
    instructions) never equals "Select Gray from the list" verbatim.

    A genuine ambiguous case (two distinct candidates whose reconstructed
    instruction is identical) is reported with no prediction, since that's
    a real unresolvable collision, not a matcher bug.
    """
    elements = _candidate_elements(screenshot_rows)
    display_names = _disambiguated_names(elements)

    matches: list[tuple[str, UIElement]] = []
    for el in elements:
        name = display_names[id(el)]
        templates = _INSTRUCTION_TEMPLATES.get(el.control_type, _FALLBACK_TEMPLATES)
        if any(t.format(name=name) == instruction for t in templates):
            matches.append((name, el))

    if not matches:
        return UiaOnlyPrediction(center=None, matched_name=None, ambiguous=False)

    if len(matches) > 1:
        return UiaOnlyPrediction(center=None, matched_name=None, ambiguous=True)

    name, element = matches[0]
    return UiaOnlyPrediction(center=element.center, matched_name=name, ambiguous=False)


def is_hit(prediction: UiaOnlyPrediction, ground_truth_bbox: tuple[int, int, int, int]) -> bool:
    """Click-accuracy check per the hypothesis doc's primary metric: does
    the predicted point fall inside the ground-truth element's bbox. A
    miss (no prediction) is always a miss, never a free pass."""
    if prediction.center is None:
        return False
    left, top, right, bottom = ground_truth_bbox
    x, y = prediction.center
    return left <= x <= right and top <= y <= bottom
