"""
Tests for the UIA-only ablation arm (src/computeruse/eval/uia_only.py).

Fixtures build screenshot_rows the same shape collect_from_current_window
emits into labels.jsonl -- just the fields uia_only.py actually reads
(element name/control_type/bbox_real), not a full LabeledSample.
"""

import json

from computeruse.eval.uia_only import UiaOnlyPrediction, is_hit, predict, run_arm, to_eval_records


def _row(name: str, control_type: str, bbox: tuple[int, int, int, int]) -> dict:
    return {"element": {"name": name, "control_type": control_type, "bbox_real": list(bbox)}}


def test_predict_matches_the_only_candidate():
    rows = [_row("Save", "Button", (10, 10, 30, 30))]
    result = predict("Click Save", rows)
    assert result.center == (20, 20)
    assert result.matched_name == "Save"
    assert not result.ambiguous


def test_predict_picks_the_right_candidate_among_several():
    rows = [
        _row("Save", "Button", (10, 10, 30, 30)),
        _row("Cancel", "Button", (40, 10, 60, 30)),
        _row("Enabled", "CheckBox", (0, 50, 20, 70)),
    ]
    result = predict("Press the Cancel button", rows)
    assert result.center == (50, 20)
    assert result.matched_name == "Cancel"


def test_predict_resolves_duplicate_names_via_ordinal_disambiguation():
    # the real Windows Terminal case: two tabs, same name, different positions
    rows = [
        _row("Windows PowerShell", "TabItem", (100, 140, 220, 160)),
        _row("Windows PowerShell", "TabItem", (300, 140, 420, 160)),
    ]
    first = predict("Open the first Windows PowerShell tab", rows)
    second = predict("Open the second Windows PowerShell tab", rows)

    assert first.center == (160, 150)
    assert second.center == (360, 150)
    assert not first.ambiguous
    assert not second.ambiguous


def test_predict_prefers_longest_match_for_prefix_collisions():
    # "Save" is a literal substring of "Save As" -- the longer, more
    # specific name should win rather than the shorter partial match
    rows = [
        _row("Save", "Button", (10, 10, 30, 30)),
        _row("Save As", "Button", (40, 10, 90, 30)),
    ]
    result = predict("Click Save As", rows)
    assert result.matched_name == "Save As"
    assert result.center == (65, 20)


def test_predict_reports_ambiguous_when_two_control_types_share_a_template():
    # Button and MenuItem both include "Click {name}" as a template -- a
    # Button and a MenuItem with the same name produce byte-identical
    # instructions, a genuine unresolvable collision the matcher should
    # refuse to guess on rather than silently pick one.
    rows = [
        _row("Settings", "Button", (0, 0, 10, 10)),
        _row("Settings", "MenuItem", (20, 0, 30, 10)),
    ]
    result = predict("Click Settings", rows)
    assert result.center is None
    assert result.ambiguous


def test_predict_returns_no_match_when_nothing_fits():
    rows = [_row("Save", "Button", (10, 10, 30, 30))]
    result = predict("Click something else entirely", rows)
    assert result.center is None
    assert not result.ambiguous


def test_is_hit_true_when_point_inside_bbox():
    prediction = UiaOnlyPrediction(center=(20, 20), matched_name="Save", ambiguous=False)
    assert is_hit(prediction, ground_truth_bbox=(10, 10, 30, 30))


def test_is_hit_false_when_point_outside_bbox():
    prediction = UiaOnlyPrediction(center=(100, 100), matched_name="Save", ambiguous=False)
    assert not is_hit(prediction, ground_truth_bbox=(10, 10, 30, 30))


def test_is_hit_false_for_a_miss():
    prediction = UiaOnlyPrediction(center=None, matched_name=None, ambiguous=False)
    assert not is_hit(prediction, ground_truth_bbox=(10, 10, 30, 30))


def _label_row(id_, screenshot_id, app, richness, split, instruction, name, control_type, bbox):
    return {
        "id": id_,
        "screenshot_id": screenshot_id,
        "app": app,
        "app_richness": richness,
        "split": split,
        "instruction": instruction,
        "element": {"name": name, "control_type": control_type, "bbox_real": list(bbox)},
    }


def _write_labels(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_run_arm_scores_a_resolvable_example_as_a_hit(tmp_path):
    rows = [
        _label_row("n1", "shot1", "notepad", "rich", "dev", "Click Save", "Save", "Button", (10, 10, 30, 30)),
    ]
    _write_labels(tmp_path / "labels.jsonl", rows)

    results = run_arm(tmp_path)
    assert len(results) == 1
    assert results[0].available is True
    assert results[0].hit is True
    assert results[0].app == "notepad"
    assert results[0].app_richness == "rich"
    assert results[0].split == "dev"


def test_run_arm_excludes_train_split_by_default(tmp_path):
    rows = [
        _label_row("n1", "shot1", "notepad", "rich", "train", "Click Save", "Save", "Button", (10, 10, 30, 30)),
        _label_row("n2", "shot1", "notepad", "rich", "dev", "Click Save", "Save", "Button", (10, 10, 30, 30)),
    ]
    _write_labels(tmp_path / "labels.jsonl", rows)

    results = run_arm(tmp_path)
    assert [r.example_id for r in results] == ["n2"]


def test_run_arm_marks_ambiguous_examples_unavailable(tmp_path):
    # both elements share a template -> byte-identical instructions -> unresolvable
    rows = [
        _label_row("n1", "shot1", "notepad", "rich", "dev", "Click Settings", "Settings", "Button", (0, 0, 10, 10)),
        _label_row("n2", "shot1", "notepad", "rich", "dev", "Click Settings", "Settings", "MenuItem", (20, 0, 30, 10)),
    ]
    _write_labels(tmp_path / "labels.jsonl", rows)

    results = run_arm(tmp_path)
    assert all(r.available is False for r in results)
    assert all(r.hit is False for r in results)


def test_run_arm_candidates_are_scoped_to_the_same_screenshot(tmp_path):
    # a Cancel button on a *different* screenshot must not become a false
    # candidate for this example -- run_arm groups by screenshot_id, not
    # by app, before calling predict()
    rows = [
        _label_row("n1", "shot1", "notepad", "rich", "dev", "Click Save", "Save", "Button", (10, 10, 30, 30)),
        _label_row("n2", "shot2", "notepad", "rich", "dev", "Click Cancel", "Cancel", "Button", (40, 10, 60, 30)),
    ]
    _write_labels(tmp_path / "labels.jsonl", rows)

    results = run_arm(tmp_path)
    by_id = {r.example_id: r for r in results}
    assert by_id["n1"].hit is True
    assert by_id["n2"].hit is True


def test_to_eval_records_drops_available_and_tags_the_arm(tmp_path):
    rows = [
        _label_row("n1", "shot1", "notepad", "rich", "dev", "Click Save", "Save", "Button", (10, 10, 30, 30)),
    ]
    _write_labels(tmp_path / "labels.jsonl", rows)

    records = to_eval_records(run_arm(tmp_path))
    assert len(records) == 1
    assert records[0].arm == "uia_only"
    assert records[0].hit is True
    assert not hasattr(records[0], "available")
