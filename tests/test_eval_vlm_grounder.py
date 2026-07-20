"""
Tests for the pure-logic half of src/computeruse/eval/vlm_grounder.py
(parsing, normalization, the metric). The model-running half (predict,
evaluate_arm) needs a real ~4GB model and a GPU -- exercised from the
Kaggle notebook, not here, same split as train_lora.py's own test coverage.
"""

from computeruse.eval.vlm_grounder import (
    ArmResult,
    center_distance_normalized,
    is_hit_normalized,
    normalize_bbox,
    parse_point,
    summarize_diagnostics,
)
from computeruse.training.prepare_dataset import normalized_center


def test_parse_point_extracts_a_clean_pair():
    assert parse_point("(423,551)") == (423, 551)


def test_parse_point_handles_a_space_after_the_comma():
    assert parse_point("(423, 551)") == (423, 551)


def test_parse_point_extracts_from_surrounding_text():
    # zero-shot models aren't guaranteed to follow the "only the point"
    # instruction -- the parser should still find it embedded in prose
    assert parse_point("The target point is (423,551) on the image.") == (423, 551)


def test_parse_point_returns_none_when_nothing_matches():
    # a real, expected zero-shot failure mode: no parseable point at all
    assert parse_point("I cannot determine the exact coordinates.") is None


def test_parse_point_takes_the_first_match_when_several_present():
    assert parse_point("(1,2) and also (3,4)") == (1, 2)


def test_normalize_bbox_agrees_with_prepare_dataset_center_on_the_midpoint():
    # normalize_bbox and prepare_dataset.normalized_center independently
    # implement the same real->scaled->[0,1000) transform -- if they ever
    # drift apart, the VLM arms and the training targets would be scored
    # in subtly different coordinate spaces without either side erroring.
    bbox_real = (100, 200, 300, 400)
    real_size = (1920, 1080)
    scale_x, scale_y = 1.2244897959183674, 1.2244897959183674

    left, top, right, bottom = normalize_bbox(bbox_real, real_size, scale_x, scale_y)
    expected_cx, expected_cy = normalized_center(list(bbox_real), real_size, scale_x, scale_y)

    midpoint_x = round((left + right) / 2)
    midpoint_y = round((top + bottom) / 2)
    assert abs(midpoint_x - expected_cx) <= 1
    assert abs(midpoint_y - expected_cy) <= 1


def test_normalize_bbox_preserves_corner_order():
    left, top, right, bottom = normalize_bbox((100, 200, 300, 400), (1920, 1080), 1.0, 1.0)
    assert left < right
    assert top < bottom


def test_is_hit_normalized_true_inside_box():
    assert is_hit_normalized((150, 250), (100, 200, 300, 400))


def test_is_hit_normalized_false_outside_box():
    assert not is_hit_normalized((500, 500), (100, 200, 300, 400))


def test_is_hit_normalized_false_on_the_boundary_is_actually_true():
    # inclusive bounds, matching eval.uia_only.is_hit's semantics
    assert is_hit_normalized((100, 200), (100, 200, 300, 400))


def test_is_hit_normalized_false_for_an_unparseable_prediction():
    assert not is_hit_normalized(None, (100, 200, 300, 400))


def test_center_distance_is_zero_at_the_box_center():
    assert center_distance_normalized((200, 300), (100, 200, 300, 400)) == 0.0


def test_center_distance_measures_from_the_center_not_the_edge():
    # (200, 300) is the center; (200, 400) is on the bottom edge, 100 away
    assert center_distance_normalized((200, 400), (100, 200, 300, 400)) == 100.0


def test_center_distance_is_none_for_an_unparseable_prediction():
    # distinct from "very far away" -- a format failure is a different
    # failure mode than a grounding failure, and averaging them together
    # would hide which one is happening
    assert center_distance_normalized(None, (100, 200, 300, 400)) is None


def _result(distance, hit=False):
    return ArmResult(
        example_id="x",
        app="calculator",
        split="dev",
        predicted_text="",
        predicted_point=None if distance is None else (0, 0),
        ground_truth_bbox_norm=(0, 0, 10, 10),
        hit=hit,
        center_distance=distance,
    )


def test_summarize_diagnostics_reports_parse_rate_over_all_results():
    results = [_result(10.0), _result(None), _result(30.0), _result(50.0)]
    assert summarize_diagnostics(results)["parse_rate"] == 0.75


def test_summarize_diagnostics_excludes_unparseable_from_the_distance_stats():
    # the None must not be counted as a zero-distance (perfect) prediction
    results = [_result(10.0), _result(None), _result(30.0)]
    diagnostics = summarize_diagnostics(results)
    assert diagnostics["median_center_distance"] == 20.0
    assert diagnostics["mean_center_distance"] == 20.0


def test_summarize_diagnostics_handles_an_all_unparseable_arm():
    # a plausible real zero-shot outcome -- must not divide by zero
    diagnostics = summarize_diagnostics([_result(None), _result(None)])
    assert diagnostics["parse_rate"] == 0.0


def test_summarize_diagnostics_handles_no_results():
    assert summarize_diagnostics([])["parse_rate"] == 0.0
