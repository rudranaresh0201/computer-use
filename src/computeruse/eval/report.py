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

import random
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


def _percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))]


def bootstrap_ci(hits: list[bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile-bootstrap interval for an accuracy. On a few hundred
    examples per slice a point estimate alone overstates what is known --
    the hypothesis doc flags exactly this as a threat to validity -- so
    every reported accuracy carries one of these."""
    if not hits:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(hits)
    means = sorted(sum(rng.choices(hits, k=n)) / n for _ in range(n_boot))
    return _percentile(means, 0.025), _percentile(means, 0.975)


def paired_difference(
    a: dict[str, bool], b: dict[str, bool], n_boot: int = 2000, seed: int = 0
) -> dict:
    """Accuracy of `a` minus accuracy of `b` on the same examples, with a
    paired 95% bootstrap interval: each resample draws example ids and
    scores both arms on them, so the interval reflects the difference
    itself rather than two independent noisy accuracies. An interval that
    excludes 0 is the evidence H1 and H3 ask for.

    Raises ValueError if the two arms scored different examples -- a
    difference over mismatched sets would not be a paired comparison."""
    if a.keys() != b.keys():
        raise ValueError(
            f"arms scored different examples: {len(a.keys() - b.keys())} only in a, "
            f"{len(b.keys() - a.keys())} only in b"
        )
    ids = sorted(a)
    if not ids:
        return {"n": 0, "a": 0.0, "b": 0.0, "diff": 0.0, "ci95": [0.0, 0.0]}
    deltas = [int(a[i]) - int(b[i]) for i in ids]
    rng = random.Random(seed)
    n = len(ids)
    boots = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(n_boot))
    return {
        "n": n,
        "a": sum(a.values()) / n,
        "b": sum(b.values()) / n,
        "diff": sum(deltas) / n,
        "ci95": [_percentile(boots, 0.025), _percentile(boots, 0.975)],
    }


def clustered_paired_difference(
    a: dict[str, bool], b: dict[str, bool], cluster_of: dict[str, str], n_boot: int = 2000, seed: int = 0
) -> dict:
    """paired_difference, resampling whole clusters (screenshots, apps)
    instead of single examples. Ten targets on one screenshot share its
    theme, resolution and layout, so they are not ten independent draws;
    resampling them one by one makes the interval too narrow. Drawing
    clusters keeps each screenshot's examples together, and the interval
    widens by however much they really move together."""
    if a.keys() != b.keys():
        raise ValueError(
            f"arms scored different examples: {len(a.keys() - b.keys())} only in a, "
            f"{len(b.keys() - a.keys())} only in b"
        )
    groups: dict[str, list[int]] = {}
    for i in sorted(a):
        groups.setdefault(cluster_of[i], []).append(int(a[i]) - int(b[i]))
    if not groups:
        return {"n": 0, "clusters": 0, "a": 0.0, "b": 0.0, "diff": 0.0, "ci95": [0.0, 0.0]}
    sums = [(sum(d), len(d)) for d in groups.values()]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        drawn = rng.choices(sums, k=len(sums))
        boots.append(sum(s for s, _ in drawn) / sum(k for _, k in drawn))
    boots.sort()
    n = len(a)
    return {
        "n": n,
        "clusters": len(groups),
        "a": sum(a.values()) / n,
        "b": sum(b.values()) / n,
        "diff": sum(s for s, _ in sums) / n,
        "ci95": [_percentile(boots, 0.025), _percentile(boots, 0.975)],
    }


def ci_by_arm_and_split(records: list[EvalRecord], n_boot: int = 2000) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for arm in sorted({r.arm for r in records}):
        for split in sorted({r.split for r in records if r.arm == arm}):
            hits = [r.hit for r in records if r.arm == arm and r.split == split]
            out.setdefault(arm, {})[split] = list(bootstrap_ci(hits, n_boot=n_boot))
    return out
