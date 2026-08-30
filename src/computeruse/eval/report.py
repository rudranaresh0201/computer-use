"""
Shared result schema + the 3-way slicing the hypothesis doc requires
(docs/research/gui-grounding-hypothesis.md: "Accuracy must be reported
sliced three ways ... An aggregate-only number would average away exactly
the distinctions this hypothesis set is designed to detect").

Every ablation arm (uia_only, vlm_grounder's zero-shot/fine-tuned, hybrid)
converts its own richer per-example result type down to EvalRecord before
reaching this module -- this is the one shape the report doesn't need to
know which arm produced.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalRecord:
    """One example's outcome under one arm. `app_richness` and `split` are
    carried per-record (not joined later against apps.yaml) so this module
    stays a pure function of its input -- no filesystem access, easy to
    unit test with fixture data."""

    example_id: str
    arm: str
    app: str
    app_richness: str
    split: str
    hit: bool


def accuracy(records: list[EvalRecord]) -> float:
    """Empty input is 0.0, not a ZeroDivisionError -- a slice with no
    examples (e.g. a richness value that happens to have zero dev rows) is
    a real, reportable state, not a bug to crash on."""
    if not records:
        return 0.0
    return sum(r.hit for r in records) / len(records)


def _group_accuracy(records: list[EvalRecord], key) -> dict[str, float]:
    groups: dict[str, list[EvalRecord]] = {}
    for r in records:
        groups.setdefault(key(r), []).append(r)
    return {k: accuracy(v) for k, v in groups.items()}


def build_report(records: list[EvalRecord]) -> dict:
    """The three required slices, each computed within arm so a reader can
    see e.g. "hybrid on weak-tree apps" directly rather than reconstructing
    it from an aggregate. `counts` is included alongside `by_arm` so a
    reader can tell "0% accuracy" apart from "0 examples in this slice" --
    the latter looks identical to a real result otherwise.
    """
    arms = sorted({r.arm for r in records})
    by_arm: dict[str, float] = {}
    counts: dict[str, int] = {}
    by_arm_and_richness: dict[str, dict[str, float]] = {}
    by_arm_and_split: dict[str, dict[str, float]] = {}

    for arm in arms:
        arm_records = [r for r in records if r.arm == arm]
        by_arm[arm] = accuracy(arm_records)
        counts[arm] = len(arm_records)
        by_arm_and_richness[arm] = _group_accuracy(arm_records, lambda r: r.app_richness)
        by_arm_and_split[arm] = _group_accuracy(arm_records, lambda r: r.split)

    return {
        "by_arm": by_arm,
        "counts": counts,
        "by_arm_and_richness": by_arm_and_richness,
        "by_arm_and_split": by_arm_and_split,
    }
