"""Tests for the cross-arm result schema + 3-way slicing (eval/report.py)."""

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
