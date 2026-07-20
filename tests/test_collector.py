"""
Dataset collector tests. The two calls that touch the real OS/UIA
(perception.screenshot.capture, perception.uia.get_foreground_window_info)
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

import pytest

from computeruse.dataset.collector import (
    LabeledSample,
    _disambiguated_names,
    WindowMismatchError,
    collect_from_current_window,
    filter_elements,
    generate_instruction,
    write_samples,
)
from computeruse.perception.screenshot import Screenshot
from computeruse.perception.uia import UIElement, WindowTree


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


def test_generate_instruction_uses_the_name_override_when_given():
    element = UIElement(name="Windows PowerShell", control_type="TabItem", rect=(0, 0, 10, 10))
    instruction = generate_instruction(element, rng=random.Random(0), name="first Windows PowerShell")
    assert "first Windows PowerShell" in instruction


# ---------------------------------------------------------------------------
# duplicate-name disambiguation
# ---------------------------------------------------------------------------


def test_disambiguated_names_leaves_unique_names_untouched():
    elements = [
        UIElement(name="Save", control_type="Button", rect=(0, 0, 10, 10)),
        UIElement(name="Cancel", control_type="Button", rect=(20, 0, 30, 10)),
    ]
    names = _disambiguated_names(elements)
    assert names[id(elements[0])] == "Save"
    assert names[id(elements[1])] == "Cancel"


def test_disambiguated_names_adds_ordinal_for_same_name_and_control_type():
    # two Windows Terminal tabs, same name, different positions -- the real
    # case that produced identical instructions for different pixels
    left_tab = UIElement(name="Windows PowerShell", control_type="TabItem", rect=(100, 140, 220, 160))
    right_tab = UIElement(name="Windows PowerShell", control_type="TabItem", rect=(300, 140, 420, 160))

    names = _disambiguated_names([left_tab, right_tab])

    assert names[id(left_tab)] == "first Windows PowerShell"
    assert names[id(right_tab)] == "second Windows PowerShell"


def test_disambiguated_names_orders_by_position_not_input_order():
    right_tab = UIElement(name="Tab", control_type="TabItem", rect=(300, 140, 420, 160))
    left_tab = UIElement(name="Tab", control_type="TabItem", rect=(100, 140, 220, 160))

    # passed in right-to-left order; leftmost should still be "first"
    names = _disambiguated_names([right_tab, left_tab])

    assert names[id(left_tab)] == "first Tab"
    assert names[id(right_tab)] == "second Tab"


def test_disambiguated_names_does_not_collide_across_control_types():
    # same name, different control_type -- not the same ambiguity, no ordinal
    button = UIElement(name="Settings", control_type="Button", rect=(0, 0, 10, 10))
    tab = UIElement(name="Settings", control_type="TabItem", rect=(50, 0, 60, 10))

    names = _disambiguated_names([button, tab])

    assert names[id(button)] == "Settings"
    assert names[id(tab)] == "Settings"


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


def _always_visible(rect):
    """Stub for uia.is_visible_at_center -- the real one hit-tests the live
    desktop, which no unit test has. Occlusion policy itself is tested
    directly against drop_occluded below."""
    return True


def _fake_window(elements, rect=(0, 0, 1920, 1080), title="Untitled - Notepad", class_name="Notepad"):
    return WindowTree(title=title, class_name=class_name, rect=rect, elements=elements)


@patch("computeruse.dataset.collector.get_foreground_window_info")
@patch("computeruse.dataset.collector.capture")
def test_collect_produces_one_sample_per_surviving_element(mock_capture, mock_window, tmp_path):
    mock_capture.return_value = _fake_screenshot()
    mock_window.return_value = _fake_window([
        UIElement(name="Save", control_type="Button", rect=(10, 10, 30, 30), automation_id="SaveBtn"),
        UIElement(name="", control_type="Button", rect=(40, 40, 60, 60)),  # filtered out
    ])

    samples = collect_from_current_window(
        app="notepad",
        app_richness="rich",
        split="train",
        session_id="sess1",
        screenshot_seq=7,
        dataset_root=tmp_path,
        visibility_check=_always_visible,
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


@patch("computeruse.dataset.collector.get_foreground_window_info")
@patch("computeruse.dataset.collector.capture")
def test_collect_sample_element_and_coordinate_fields_match_schema(mock_capture, mock_window, tmp_path):
    shot = _fake_screenshot()
    mock_capture.return_value = shot
    mock_window.return_value = _fake_window([
        UIElement(name="Save", control_type="Button", rect=(10, 10, 30, 30), automation_id="SaveBtn"),
    ])

    samples = collect_from_current_window(
        app="notepad",
        app_richness="rich",
        split="train",
        session_id="sess1",
        screenshot_seq=1,
        dataset_root=tmp_path,
        visibility_check=_always_visible,
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


@patch("computeruse.dataset.collector.get_foreground_window_info")
@patch("computeruse.dataset.collector.capture")
def test_collect_screenshot_path_is_relative_to_dataset_root(mock_capture, mock_window, tmp_path):
    mock_capture.return_value = _fake_screenshot()
    mock_window.return_value = _fake_window([
        UIElement(name="Save", control_type="Button", rect=(10, 10, 30, 30)),
    ])

    samples = collect_from_current_window(
        app="notepad",
        app_richness="rich",
        split="train",
        session_id="sess1",
        screenshot_seq=3,
        dataset_root=tmp_path,
        visibility_check=_always_visible,
        rng=random.Random(0),
    )

    assert samples[0].screenshot_path == "images/notepad/notepad_0003.png"


@patch("computeruse.dataset.collector.get_foreground_window_info")
@patch("computeruse.dataset.collector.capture")
def test_collect_passes_the_image_save_path_to_capture(mock_capture, mock_window, tmp_path):
    mock_capture.return_value = _fake_screenshot()
    mock_window.return_value = _fake_window([])

    collect_from_current_window(
        app="notepad",
        app_richness="rich",
        split="train",
        session_id="sess1",
        screenshot_seq=1,
        dataset_root=tmp_path,
        visibility_check=_always_visible,
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


# --- window verification and cropping (added 2026-07-19 after the audit) ---


@patch("computeruse.dataset.collector.get_foreground_window_info")
@patch("computeruse.dataset.collector.capture")
def test_collect_raises_when_the_foreground_window_is_a_different_app(
    mock_capture, mock_window, tmp_path
):
    # THE regression this guards: focus was stolen mid-run and the collector
    # labeled the code editor's elements as the target app -- 196 of 542 rows
    # in the v2 dataset, including 100% of one held-out app. Must raise.
    mock_capture.return_value = _fake_screenshot()
    mock_window.return_value = _fake_window(
        [UIElement(name="Explorer", control_type="Button", rect=(10, 10, 30, 30))],
        title="labels.jsonl - project computer use - Visual Studio Code",
        class_name="Chrome_WidgetWin_1",
    )

    with pytest.raises(WindowMismatchError):
        collect_from_current_window(
            app="file_explorer",
            app_richness="rich",
            split="train",
            session_id="sess1",
            screenshot_seq=0,
            dataset_root=tmp_path,
        visibility_check=_always_visible,
            expect_title_contains="File Explorer",
        )


@patch("computeruse.dataset.collector.get_foreground_window_info")
@patch("computeruse.dataset.collector.capture")
def test_collect_does_not_even_screenshot_on_mismatch(mock_capture, mock_window, tmp_path):
    # a wrong-window PNG left on disk is how 8 unreviewed screenshots ended
    # up inside the Kaggle upload zip
    mock_capture.return_value = _fake_screenshot()
    mock_window.return_value = _fake_window([], title="Visual Studio Code", class_name="X")

    with pytest.raises(WindowMismatchError):
        collect_from_current_window(
            app="audacity",
            app_richness="weak",
            split="test_held_out_app",
            session_id="sess1",
            screenshot_seq=0,
            dataset_root=tmp_path,
        visibility_check=_always_visible,
            expect_title_contains="Audacity",
        )
    mock_capture.assert_not_called()


@patch("computeruse.dataset.collector.get_foreground_window_info")
@patch("computeruse.dataset.collector.capture")
def test_collect_accepts_a_class_match_when_the_title_is_unreliable(
    mock_capture, mock_window, tmp_path
):
    # Windows Terminal reports the active shell's name, never "Terminal"
    mock_capture.return_value = _fake_screenshot()
    mock_window.return_value = _fake_window(
        [UIElement(name="Save", control_type="Button", rect=(10, 10, 30, 30))],
        title="Windows PowerShell",
        class_name="CASCADIA_HOSTING_WINDOW_CLASS",
    )

    samples = collect_from_current_window(
        app="windows_terminal",
        app_richness="weak",
        split="train",
        session_id="sess1",
        screenshot_seq=0,
        dataset_root=tmp_path,
        visibility_check=_always_visible,
        rng=random.Random(0),
        expect_title_contains="Terminal",
        expect_class_contains="CASCADIA_HOSTING_WINDOW_CLASS",
    )
    assert len(samples) == 1


@patch("computeruse.dataset.collector.get_foreground_window_info")
@patch("computeruse.dataset.collector.capture")
def test_collect_crops_to_the_window_and_rebases_boxes(mock_capture, mock_window, tmp_path):
    # window at (200,100); a button at screen (250,150) must be stored at
    # (50,50) -- relative to the cropped image, which is what the model sees.
    window_rect = (200, 100, 1000, 700)
    mock_capture.return_value = Screenshot(
        real_size=(800, 600), scaled_size=(800, 600), scale_x=1.0, scale_y=1.0,
        png_bytes=b"fake", origin=(200, 100),
    )
    mock_window.return_value = _fake_window(
        [UIElement(name="Save", control_type="Button", rect=(250, 150, 290, 180))],
        rect=window_rect,
    )

    samples = collect_from_current_window(
        app="notepad", app_richness="rich", split="train", session_id="sess1",
        screenshot_seq=0, dataset_root=tmp_path, rng=random.Random(0),
        visibility_check=_always_visible,
        expect_title_contains="Notepad", crop_to_window=True,
    )

    assert samples[0].element["bbox_real"] == [50, 50, 90, 80]
    _, kwargs = mock_capture.call_args
    assert kwargs["region"] == window_rect


@patch("computeruse.dataset.collector.get_foreground_window_info")
@patch("computeruse.dataset.collector.capture")
def test_collect_drops_elements_outside_the_target_window(mock_capture, mock_window, tmp_path):
    # a dropdown hosted in its own top-level window would land outside the
    # cropped image entirely -- an unlearnable target
    mock_capture.return_value = Screenshot(
        real_size=(800, 600), scaled_size=(800, 600), scale_x=1.0, scale_y=1.0,
        png_bytes=b"fake", origin=(200, 100),
    )
    mock_window.return_value = _fake_window(
        [
            UIElement(name="Inside", control_type="Button", rect=(250, 150, 290, 180)),
            UIElement(name="Outside", control_type="Button", rect=(1500, 900, 1560, 930)),
        ],
        rect=(200, 100, 1000, 700),
    )

    samples = collect_from_current_window(
        app="notepad", app_richness="rich", split="train", session_id="sess1",
        screenshot_seq=0, dataset_root=tmp_path, rng=random.Random(0),
        visibility_check=_always_visible,
        expect_title_contains="Notepad", crop_to_window=True,
    )

    assert [s.element["name"] for s in samples] == ["Inside"]


def test_collect_defaults_to_a_full_desktop_frame():
    # crop_to_window defaults off so a partial top-up of an existing
    # full-desktop dataset can't silently introduce a second image format
    import inspect

    from computeruse.dataset.collector import collect_from_current_window as fn

    assert inspect.signature(fn).parameters["crop_to_window"].default is False


# --- occlusion (added 2026-07-19 after calculator_0022 labeled four buttons
# hidden behind an open navigation flyout) ---


def test_drop_occluded_keeps_visible_elements():
    from computeruse.dataset.collector import drop_occluded

    visible = UIElement(name="Save", control_type="Button", rect=(10, 10, 30, 30))
    assert drop_occluded([visible], lambda rect: True) == [visible]


def test_drop_occluded_removes_elements_hidden_behind_a_flyout():
    from computeruse.dataset.collector import drop_occluded

    covered = UIElement(name="Plus", control_type="Button", rect=(10, 10, 30, 30))
    assert drop_occluded([covered], lambda rect: False) == []


def test_drop_occluded_is_applied_before_instructions_are_generated(tmp_path):
    # an occluded element must never reach labels.jsonl at all -- not be
    # written and filtered later
    hidden = {(100, 100, 140, 130)}

    with patch("computeruse.dataset.collector.capture") as mock_capture, patch(
        "computeruse.dataset.collector.get_foreground_window_info"
    ) as mock_window:
        mock_capture.return_value = _fake_screenshot()
        mock_window.return_value = _fake_window([
            UIElement(name="Visible", control_type="Button", rect=(10, 10, 40, 30)),
            UIElement(name="BehindMenu", control_type="Button", rect=(100, 100, 140, 130)),
        ])
        samples = collect_from_current_window(
            app="calculator",
            app_richness="rich",
            split="train",
            session_id="s",
            screenshot_seq=0,
            dataset_root=tmp_path,
            rng=random.Random(0),
            visibility_check=lambda rect: tuple(rect) not in hidden,
        )

    assert [s.element["name"] for s in samples] == ["Visible"]
    assert all("BehindMenu" not in s.instruction for s in samples)
