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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from computeruse.dataset.collector import (
    _FALLBACK_TEMPLATES,
    _INSTRUCTION_TEMPLATES,
    _disambiguated_names,
)
from computeruse.eval.report import EvalRecord
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


@dataclass
class UiaArmResult:
    """Per-example outcome plus `available` -- whether UIA resolved a
    prediction at all (matches != 0, not ambiguous). The hybrid arm
    (eval/hybrid.py) needs this to decide whether to trust this result or
    fall back to the fine-tuned grounder; EvalRecord alone (hit/miss only)
    can't distinguish "UIA confidently pointed at the wrong element" from
    "UIA had nothing to say", and hybrid's policy only defers on the
    latter."""

    example_id: str
    app: str
    app_richness: str
    split: str
    available: bool
    hit: bool


def run_arm(
    dataset_root: Path,
    splits: tuple[str, ...] = ("dev", "test_same_app", "test_held_out_app"),
) -> list[UiaArmResult]:
    """Run the UIA-only arm over every labels.jsonl row in `splits`. CPU/
    text-only -- no model, no GPU, safe to run on this machine against the
    real dataset (unlike vlm_grounder.evaluate_arm).

    Candidates for each row are every element sharing its screenshot_id
    (labels.jsonl's closest proxy for "the UIA tree behind this
    screenshot" -- see _candidate_elements), which includes the row's own
    ground-truth element as well as its siblings, matching what predict()
    would see from a live UIA query.
    """
    rows = []
    with (dataset_root / "labels.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_screenshot: dict[str, list[dict]] = {}
    for row in rows:
        by_screenshot.setdefault(row["screenshot_id"], []).append(row)

    results: list[UiaArmResult] = []
    for row in rows:
        if row["split"] not in splits:
            continue
        screenshot_rows = by_screenshot[row["screenshot_id"]]
        prediction = predict(row["instruction"], screenshot_rows)
        results.append(
            UiaArmResult(
                example_id=row["id"],
                app=row["app"],
                app_richness=row["app_richness"],
                split=row["split"],
                available=prediction.center is not None,
                hit=is_hit(prediction, tuple(row["element"]["bbox_real"])),
            )
        )
    return results


def to_eval_records(results: list[UiaArmResult]) -> list[EvalRecord]:
    """Drop `available` (an arm-internal detail hybrid.combine needs, but
    report.build_report doesn't) to get the shape build_report expects."""
    return [
        EvalRecord(
            example_id=r.example_id,
            arm="uia_only",
            app=r.app,
            app_richness=r.app_richness,
            split=r.split,
            hit=r.hit,
        )
        for r in results
    ]
