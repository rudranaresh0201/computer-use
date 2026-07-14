"""
Dataset collector tests. The two calls that touch the real OS/UIA
(perception.screenshot.capture, perception.uia.get_foreground_window_tree)
are mocked wherever a test exercises collect_from_current_window, matching
the patching convention in tests/test_nodes.py (patch the name as imported
into the module under test). Everything else here — filtering, dedup,
instruction templates, JSONL writing — is pure and tested directly, no
mocking needed.
"""

import json
import random
from pathlib import Path
from unittest.mock import patch

from computeruse.dataset.collector import (
    LabeledSample,
    collect_from_current_window,
    filter_elements,
    generate_instruction,
    write_samples,
)
from computeruse.perception.screenshot import Screenshot
from computeruse.perception.uia import UIElement


# ---------------------------------------------------------------------------
# element filtering
# ---------------------------------------------------------------------------


def test_filter_drops_elements_without_a_name():
    elements = [
        UIElement(name="", control_type="Button", rect=(0, 0, 10, 10)),
        UIElement(name="   ", control_type="Button", rect=(20, 20, 30, 30)),
        UIElement(name="Save", control_type="Button", rect=(40, 40, 50, 50)),
    ]
    result = filter_elements(elements, real_size=(1920, 1080))
    assert [e.name for e in result] == ["Save"]


def test_filter_drops_off_screen_elements():
    elements = [
        UIElement(name="Offscreen", control_type="Button", rect=(-500, -500, -480, -480)),
        UIElement(name="Onscreen", control_type="Button", rect=(10, 10, 30, 30)),
    ]
    result = filter_elements(elements, real_size=(1920, 1080))
    assert [e.name for e in result] == ["Onscreen"]


def test_filter_keeps_elements_that_are_only_partially_on_screen():
    # right/bottom past the screen edge is fine as long as it overlaps at all
    elements = [
        UIElement(name="Edge", control_type="Button", rect=(1900, 1060, 1950, 1100)),
    ]
    result = filter_elements(elements, real_size=(1920, 1080))
    assert [e.name for e in result] == ["Edge"]


def test_filter_drops_control_types_without_a_real_instruction_template():
    # a window, a text label, and a button can share the same UIA name --
    # only the button has a real template, so only it should survive.
    elements = [
        UIElement(name="Calculator", control_type="Window", rect=(0, 0, 500, 800)),
        UIElement(name="Calculator", control_type="Text", rect=(140, 60, 220, 85)),
        UIElement(name="Calculator", control_type="Button", rect=(10, 10, 30, 30)),
    ]
    result = filter_elements(elements, real_size=(1920, 1080))
    assert [(e.name, e.control_type) for e in result] == [("Calculator", "Button")]


# ---------------------------------------------------------------------------
# deduplication
# ---------------------------------------------------------------------------


def test_dedup_collapses_near_identical_rects_keeping_the_later_node():
    # simulates a container walked before its near-identical single child
    elements = [
        UIElement(name="Panel", control_type="Pane", rect=(10, 10, 100, 40)),
        UIElement(name="Save", control_type="Button", rect=(11, 11, 101, 41)),  # within tolerance
    ]
    result = filter_elements(elements, real_size=(1920, 1080))
    assert len(result) == 1
    assert result[0].name == "Save"


def test_dedup_keeps_rects_beyond_tolerance_as_distinct():
    elements = [
        UIElement(name="Save", control_type="Button", rect=(0, 0, 20, 20)),
        UIElement(name="Cancel", control_type="Button", rect=(100, 100, 120, 120)),
    ]
    result = filter_elements(elements, real_size=(1920, 1080))
    assert {e.name for e in result} == {"Save", "Cancel"}


# ---------------------------------------------------------------------------
# element filtering: control-type cap (rule 4)
# ---------------------------------------------------------------------------


def test_filter_caps_elements_per_control_type():
    buttons = [
        UIElement(name=f"btn{i}", control_type="Button", rect=(i * 100, 0, i * 100 + 20, 20))
        for i in range(10)
    ]
    checkbox = UIElement(name="Enabled", control_type="CheckBox", rect=(0, 200, 20, 220))

    result = filter_elements(buttons + [checkbox], real_size=(2000, 400), max_per_control_type=3)

    button_names = [e.name for e in result if e.control_type == "Button"]
    assert len(button_names) == 3
    assert any(e.control_type == "CheckBox" for e in result)


# ---------------------------------------------------------------------------
# instruction generation
# ---------------------------------------------------------------------------


def test_generate_instruction_fills_in_the_element_name():
    element = UIElement(name="Save", control_type="Button", rect=(0, 0, 10, 10))
    instruction = generate_instruction(element, rng=random.Random(0))
    assert "Save" in instruction


def test_generate_instruction_uses_a_template_matching_control_type():
    tab = UIElement(name="Settings", control_type="TabItem", rect=(0, 0, 10, 10))
    instruction = generate_instruction(tab, rng=random.Random(0))
    assert "tab" in instruction.lower()


def test_generate_instruction_falls_back_for_unknown_control_type():
    weird = UIElement(name="Widget", control_type="SomeUnmappedType", rect=(0, 0, 10, 10))
    instruction = generate_instruction(weird, rng=random.Random(0))
    assert instruction == "Click Widget"


def test_generate_instruction_is_deterministic_for_a_seeded_rng():
    element = UIElement(name="Save", control_type="Button", rect=(0, 0, 10, 10))
    first = generate_instruction(element, rng=random.Random(42))
    second = generate_instruction(element, rng=random.Random(42))
    assert first == second


# ---------------------------------------------------------------------------
# labeled sample schema (collect_from_current_window, OS calls mocked)
# ---------------------------------------------------------------------------


def _fake_screenshot() -> Screenshot:
    return Screenshot(
        real_size=(1920, 1080),
        scaled_size=(1568, 882),
        scale_x=1.2244897959183674,
        scale_y=1.2244897959183674,
        png_bytes=b"fake-png-bytes",
    )


@patch("computeruse.dataset.collector.get_foreground_window_tree")
@patch("computeruse.dataset.collector.capture")
def test_collect_produces_one_sample_per_surviving_element(mock_capture, mock_tree, tmp_path):
    mock_capture.return_value = _fake_screenshot()
    mock_tree.return_value = [
        UIElement(name="Save", control_type="Button", rect=(10, 10, 30, 30), automation_id="SaveBtn"),
        UIElement(name="", control_type="Button", rect=(40, 40, 60, 60)),  # filtered out
    ]

    samples = collect_from_current_window(
        app="notepad",
        app_richness="rich",
        split="train",
        session_id="sess1",
        screenshot_seq=7,
        dataset_root=tmp_path,
        rng=random.Random(0),
    )

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, LabeledSample)
    assert sample.id == "notepad_0007_e0"
    assert sample.screenshot_id == "notepad_0007"
    assert sample.app == "notepad"
    assert sample.app_richness == "rich"
    assert sample.split == "train"
    assert sample.session_id == "sess1"
    assert sample.source == "uia_auto_label"


@patch("computeruse.dataset.collector.get_foreground_window_tree")
@patch("computeruse.dataset.collector.capture")
def test_collect_sample_element_and_coordinate_fields_match_schema(mock_capture, mock_tree, tmp_path):
    shot = _fake_screenshot()
    mock_capture.return_value = shot
    mock_tree.return_value = [
        UIElement(name="Save", control_type="Button", rect=(10, 10, 30, 30), automation_id="SaveBtn"),
    ]

    samples = collect_from_current_window(
        app="notepad",
        app_richness="rich",
        split="train",
        session_id="sess1",
        screenshot_seq=1,
        dataset_root=tmp_path,
        rng=random.Random(0),
    )

    sample = samples[0]
    assert sample.real_size == shot.real_size
    assert sample.scaled_size == shot.scaled_size
    assert sample.scale_x == shot.scale_x
    assert sample.scale_y == shot.scale_y
    assert sample.element == {
        "name": "Save",
        "control_type": "Button",
        "automation_id": "SaveBtn",
        "bbox_real": [10, 10, 30, 30],
    }
    assert sample.instruction  # non-empty, template-generated


@patch("computeruse.dataset.collector.get_foreground_window_tree")
@patch("computeruse.dataset.collector.capture")
def test_collect_screenshot_path_is_relative_to_dataset_root(mock_capture, mock_tree, tmp_path):
    mock_capture.return_value = _fake_screenshot()
    mock_tree.return_value = [
        UIElement(name="Save", control_type="Button", rect=(10, 10, 30, 30)),
    ]

    samples = collect_from_current_window(
        app="notepad",
        app_richness="rich",
        split="train",
        session_id="sess1",
        screenshot_seq=3,
        dataset_root=tmp_path,
        rng=random.Random(0),
    )

    assert samples[0].screenshot_path == "images/notepad/notepad_0003.png"


@patch("computeruse.dataset.collector.get_foreground_window_tree")
@patch("computeruse.dataset.collector.capture")
def test_collect_passes_the_image_save_path_to_capture(mock_capture, mock_tree, tmp_path):
    mock_capture.return_value = _fake_screenshot()
    mock_tree.return_value = []

    collect_from_current_window(
        app="notepad",
        app_richness="rich",
        split="train",
        session_id="sess1",
        screenshot_seq=1,
        dataset_root=tmp_path,
        rng=random.Random(0),
    )

    _, kwargs = mock_capture.call_args
    assert kwargs["save_path"] == tmp_path / "images" / "notepad" / "notepad_0001.png"


# ---------------------------------------------------------------------------
# JSONL writing
# ---------------------------------------------------------------------------


def _sample(sample_id: str) -> LabeledSample:
    return LabeledSample(
        id=sample_id,
        screenshot_id="notepad_0001",
        screenshot_path="images/notepad/notepad_0001.png",
        real_size=(1920, 1080),
        scaled_size=(1568, 882),
        scale_x=1.22,
        scale_y=1.22,
        element={"name": "Save", "control_type": "Button", "automation_id": None, "bbox_real": [10, 10, 30, 30]},
        instruction="Click Save",
        app="notepad",
        app_richness="rich",
        split="train",
        session_id="sess1",
    )


def test_write_samples_creates_one_json_line_per_sample(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    write_samples([_sample("a"), _sample("b")], labels_path)

    lines = labels_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["id"] for p in parsed] == ["a", "b"]
    assert parsed[0]["element"]["name"] == "Save"


def test_write_samples_appends_across_calls_rather_than_overwriting(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    write_samples([_sample("a")], labels_path)
    write_samples([_sample("b")], labels_path)

    lines = labels_path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["a", "b"]


def test_write_samples_creates_parent_directories(tmp_path):
    labels_path = tmp_path / "nested" / "dir" / "labels.jsonl"
    write_samples([_sample("a")], labels_path)
    assert labels_path.exists()
