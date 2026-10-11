"""Tests for the cross-arm result schema + 3-way slicing (eval/report.py)."""

import pytest

from computeruse.eval.report import EvalRecord, accuracy, build_report


def _rec(arm, app, richness, split, hit, example_id="x"):
    return EvalRecord(
        example_id=example_id, arm=arm, app=app, app_richness=richness, split=split, hit=hit
    )


def test_accuracy_empty_is_zero_not_a_crash():
    assert accuracy([]) == 0.0


def test_accuracy_is_hit_fraction():
    records = [_rec("a", "notepad", "rich", "dev", True), _rec("a", "notepad", "rich", "dev", False)]
    assert accuracy(records) == 0.5


def test_build_report_slices_by_arm_independently():
    records = [
        _rec("uia_only", "notepad", "rich", "dev", True),
        _rec("uia_only", "notepad", "rich", "dev", False),
        _rec("zero_shot", "notepad", "rich", "dev", True),
    ]
    report = build_report(records)
    assert report["by_arm"]["uia_only"] == 0.5
    assert report["by_arm"]["zero_shot"] == 1.0


def test_build_report_counts_distinguish_zero_accuracy_from_no_examples():
    records = [_rec("uia_only", "notepad", "rich", "dev", False)]
    report = build_report(records)
    assert report["by_arm"]["uia_only"] == 0.0
    assert report["counts"]["uia_only"] == 1
    # a richness value with zero rows for this arm just isn't a key
    assert "weak" not in report["by_arm_and_richness"]["uia_only"]


def test_build_report_slices_by_richness_within_arm():
    records = [
        _rec("uia_only", "notepad", "rich", "dev", True),
        _rec("uia_only", "paint", "weak", "dev", False),
    ]
    report = build_report(records)
    assert report["by_arm_and_richness"]["uia_only"]["rich"] == 1.0
    assert report["by_arm_and_richness"]["uia_only"]["weak"] == 0.0


def test_build_report_slices_by_split_within_arm():
    records = [
        _rec("uia_only", "notepad", "rich", "dev", True),
        _rec("uia_only", "notepad", "rich", "test_held_out_app", False),
    ]
    report = build_report(records)
    assert report["by_arm_and_split"]["uia_only"]["dev"] == 1.0
    assert report["by_arm_and_split"]["uia_only"]["test_held_out_app"] == 0.0


# -- uncertainty: bootstrap intervals and paired differences --------------


def test_bootstrap_ci_brackets_the_accuracy_and_narrows_with_more_data():
    from computeruse.eval.report import bootstrap_ci

    small = bootstrap_ci([True, False] * 20)
    large = bootstrap_ci([True, False] * 500)
    assert small[0] < 0.5 < small[1] and large[0] < 0.5 < large[1]
    assert (large[1] - large[0]) < (small[1] - small[0])
    assert bootstrap_ci([True] * 30) == (1.0, 1.0)
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_ci_is_reproducible():
    from computeruse.eval.report import bootstrap_ci

    hits = [i % 3 == 0 for i in range(200)]
    assert bootstrap_ci(hits) == bootstrap_ci(hits)


def test_paired_difference_detects_a_consistent_gain():
    from computeruse.eval.report import paired_difference

    ids = [f"e{i}" for i in range(300)]
    b = {i: n % 4 == 0 for n, i in enumerate(ids)}            # 25%
    a = {i: b[i] or n % 4 == 1 for n, i in enumerate(ids)}    # 50%, never worse on any example
    result = paired_difference(a, b)
    assert result["n"] == 300 and result["a"] == 0.5 and result["b"] == 0.25
    assert result["diff"] == 0.25
    assert 0 < result["ci95"][0] <= 0.25 <= result["ci95"][1]


def test_paired_difference_interval_includes_zero_when_arms_tie():
    from computeruse.eval.report import paired_difference

    ids = [f"e{i}" for i in range(200)]
    a = {i: n % 2 == 0 for n, i in enumerate(ids)}
    b = {i: n % 2 == 1 for n, i in enumerate(ids)}  # same accuracy, different examples
    lo, hi = paired_difference(a, b)["ci95"]
    assert lo < 0 < hi


def test_paired_difference_refuses_mismatched_examples():
    from computeruse.eval.report import paired_difference

    with pytest.raises(ValueError, match="different examples"):
        paired_difference({"a": True}, {"b": True})
