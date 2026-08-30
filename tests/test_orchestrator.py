"""
Dataset orchestrator tests. Everything that touches the real OS/UIA (window
launch, window enumeration/focus, input dispatch, the collector's own
capture/UIA calls) is mocked, matching the patching convention in
tests/test_nodes.py and tests/test_collector.py — patch the name as imported
into the module under test.
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from computeruse.dataset.orchestrator import (
    OrchestrationError,
    ProgressLedger,
    UnexpectedSessionStateError,
    _drive_step,
    _find_element,
    _verify_clean_single_instance_session,
    run,
)
from computeruse.dataset.registry import AppConfig, AppLaunch, AppState, DriveStep
from computeruse.perception.uia import UIElement, WindowTree


# ---------------------------------------------------------------------------
# ProgressLedger: persistence + resume
# ---------------------------------------------------------------------------


def test_ledger_starts_empty_when_no_file_exists(tmp_path):
    ledger = ProgressLedger.load(tmp_path / "progress.json")
    assert ledger.completed == set()
    assert ledger.peek_seq("notepad") == 0


def test_ledger_mark_done_persists_and_reloads(tmp_path):
    path = tmp_path / "progress.json"
    ledger = ProgressLedger.load(path)
    ledger.mark_done("notepad", "empty_document")

    reloaded = ProgressLedger.load(path)
    assert reloaded.is_done("notepad", "empty_document")
    assert not reloaded.is_done("notepad", "with_text")


def test_ledger_increments_next_seq_per_app_independently(tmp_path):
    ledger = ProgressLedger.load(tmp_path / "progress.json")
    ledger.mark_done("notepad", "empty_document")
    ledger.mark_done("notepad", "with_text")
    ledger.mark_done("calculator", "default")

    assert ledger.peek_seq("notepad") == 2
    assert ledger.peek_seq("calculator") == 1


# ---------------------------------------------------------------------------
# _drive_step dispatch
# ---------------------------------------------------------------------------


def test_drive_step_key_dispatches_key_action():
    controller = MagicMock()
    _drive_step(DriveStep(kind="key", text="ctrl+f"), controller)

    action = controller.execute.call_args[0][0]
    assert action.type.value == "key"
    assert action.text == "ctrl+f"


def test_drive_step_type_dispatches_type_action():
    controller = MagicMock()
    _drive_step(DriveStep(kind="type", text="hello"), controller)

    action = controller.execute.call_args[0][0]
    assert action.type.value == "type"
    assert action.text == "hello"


def test_drive_step_wait_does_not_touch_the_controller():
    controller = MagicMock()
    _drive_step(DriveStep(kind="wait", duration_seconds=0), controller)
    controller.execute.assert_not_called()


@patch("computeruse.dataset.orchestrator.get_foreground_window_tree")
def test_drive_step_click_element_clicks_the_matching_elements_center(mock_tree):
    mock_tree.return_value = [
        UIElement(name="Save", control_type="Button", rect=(10, 10, 30, 30)),
    ]
    controller = MagicMock()
    _drive_step(DriveStep(kind="click_element", name="Save", control_type="Button"), controller)

    action = controller.execute.call_args[0][0]
    assert action.type.value == "left_click"
    assert action.coordinate == (20, 20)


@patch("computeruse.dataset.orchestrator.get_foreground_window_tree")
def test_drive_step_click_element_matches_by_automation_id_over_name(mock_tree):
    # name is a placeholder like "\r" for some native controls (Control
    # Panel's search box) -- automation_id must win when both are given.
    mock_tree.return_value = [
        UIElement(name="\r", control_type="Edit", rect=(10, 10, 30, 30), automation_id="SearchEditBox"),
    ]
    controller = MagicMock()
    _drive_step(DriveStep(kind="click_element", automation_id="SearchEditBox"), controller)

    action = controller.execute.call_args[0][0]
    assert action.type.value == "left_click"
    assert action.coordinate == (20, 20)


@patch("computeruse.dataset.orchestrator.get_foreground_window_tree")
def test_drive_step_click_element_raises_when_not_found(mock_tree):
    mock_tree.return_value = []
    controller = MagicMock()
    with pytest.raises(OrchestrationError):
        _drive_step(DriveStep(kind="click_element", name="Missing"), controller)


def test_drive_step_unknown_kind_raises():
    with pytest.raises(OrchestrationError):
        _drive_step(DriveStep(kind="teleport"), MagicMock())


@patch("computeruse.dataset.orchestrator.get_foreground_window_tree")
def test_find_element_matches_on_name_and_control_type(mock_tree):
    mock_tree.return_value = [
        UIElement(name="Save", control_type="Button", rect=(0, 0, 10, 10)),
        UIElement(name="Save", control_type="MenuItem", rect=(0, 0, 10, 10)),
    ]
    found = _find_element("Save", "MenuItem")
    assert found.control_type == "MenuItem"


# ---------------------------------------------------------------------------
# run(): registry walking, resume-skip, per-app/per-state error isolation
# ---------------------------------------------------------------------------


def _app(name: str, states: list[AppState], single_instance: bool = False) -> AppConfig:
    return AppConfig(
        name=name,
        pool="train",
        richness="rich",
        launch=AppLaunch(method="exe", target=f"{name}.exe"),
        window_title_contains=name,
        states=states,
        single_instance=single_instance,
    )


@patch("computeruse.dataset.orchestrator.write_samples")
@patch("computeruse.dataset.orchestrator.collect_from_current_window")
@patch("computeruse.dataset.orchestrator._find_and_focus_window")
@patch("computeruse.dataset.orchestrator._launch")
@patch("computeruse.dataset.orchestrator.load_registry")
def test_run_collects_every_pending_state_and_marks_them_done(
    mock_load, mock_launch, mock_focus, mock_collect, mock_write, tmp_path
):
    mock_load.return_value = [
        _app("notepad", [AppState(name="empty", split="train"), AppState(name="with_text", split="train")])
    ]
    mock_collect.return_value = [MagicMock()]

    ledger = run(
        registry_path=tmp_path / "apps.yaml",
        dataset_root=tmp_path,
        controller=MagicMock(),
        session_id="sess1",
    )

    assert ledger.is_done("notepad", "empty")
    assert ledger.is_done("notepad", "with_text")
    assert mock_collect.call_count == 2
    assert mock_write.call_count == 2


@patch("computeruse.dataset.orchestrator.write_samples")
@patch("computeruse.dataset.orchestrator.collect_from_current_window")
@patch("computeruse.dataset.orchestrator._find_and_focus_window")
@patch("computeruse.dataset.orchestrator._launch")
@patch("computeruse.dataset.orchestrator.load_registry")
def test_run_skips_already_completed_states_on_resume(
    mock_load, mock_launch, mock_focus, mock_collect, mock_write, tmp_path
):
    mock_load.return_value = [
        _app("notepad", [AppState(name="empty", split="train"), AppState(name="with_text", split="train")])
    ]
    mock_collect.return_value = []

    ledger_path = tmp_path / "progress.json"
    ProgressLedger.load(ledger_path).mark_done("notepad", "empty")

    run(
        registry_path=tmp_path / "apps.yaml",
        dataset_root=tmp_path,
        controller=MagicMock(),
        session_id="sess2",
        resume=True,
    )

    # only "with_text" should have been driven/collected this run
    assert mock_collect.call_count == 1
    assert mock_collect.call_args.kwargs["screenshot_seq"] == 1  # notepad already used seq 0


@patch("computeruse.dataset.orchestrator.write_samples")
@patch("computeruse.dataset.orchestrator.collect_from_current_window")
@patch("computeruse.dataset.orchestrator._find_and_focus_window")
@patch("computeruse.dataset.orchestrator._launch")
@patch("computeruse.dataset.orchestrator.load_registry")
def test_run_skips_app_entirely_when_all_states_already_done(
    mock_load, mock_launch, mock_focus, mock_collect, mock_write, tmp_path
):
    mock_load.return_value = [_app("notepad", [AppState(name="empty", split="train")])]

    ledger_path = tmp_path / "progress.json"
    ProgressLedger.load(ledger_path).mark_done("notepad", "empty")

    run(
        registry_path=tmp_path / "apps.yaml",
        dataset_root=tmp_path,
        controller=MagicMock(),
        session_id="sess2",
    )

    mock_launch.assert_not_called()
    mock_collect.assert_not_called()


@patch("computeruse.dataset.orchestrator.write_samples")
@patch("computeruse.dataset.orchestrator.collect_from_current_window")
@patch("computeruse.dataset.orchestrator._find_and_focus_window")
@patch("computeruse.dataset.orchestrator._launch")
@patch("computeruse.dataset.orchestrator.load_registry")
def test_run_continues_to_next_app_when_launch_fails(
    mock_load, mock_launch, mock_focus, mock_collect, mock_write, tmp_path
):
    mock_load.return_value = [
        _app("broken_app", [AppState(name="default", split="train")]),
        _app("notepad", [AppState(name="empty", split="train")]),
    ]
    mock_launch.side_effect = [RuntimeError("not installed"), None]
    mock_collect.return_value = []

    ledger = run(
        registry_path=tmp_path / "apps.yaml",
        dataset_root=tmp_path,
        controller=MagicMock(),
        session_id="sess1",
    )

    assert not ledger.is_done("broken_app", "default")
    assert ledger.is_done("notepad", "empty")


# ---------------------------------------------------------------------------
# _verify_clean_single_instance_session: catches a restored real session
# that _launch's "is a process already running" check can't see (2026-07-19
# .env-tab incident -- see docstring on the function under test)
# ---------------------------------------------------------------------------


def _window(elements: list[UIElement]) -> WindowTree:
    return WindowTree(title="Notepad", class_name="Notepad", rect=(0, 0, 800, 600), elements=elements)


@patch("computeruse.dataset.orchestrator.get_foreground_window_info")
def test_verify_clean_session_skips_check_for_non_single_instance_apps(mock_info):
    config = _app("paint", [], single_instance=False)
    _verify_clean_single_instance_session(config)
    mock_info.assert_not_called()


@patch("computeruse.dataset.orchestrator.get_foreground_window_info")
def test_verify_clean_session_passes_with_only_a_fresh_untitled_tab(mock_info):
    mock_info.return_value = _window(
        [UIElement(name="Untitled", control_type="TabItem", rect=(0, 0, 10, 10))]
    )
    config = _app("notepad", [], single_instance=True)
    _verify_clean_single_instance_session(config)  # must not raise


@patch("computeruse.dataset.orchestrator.get_foreground_window_info")
def test_verify_clean_session_raises_on_a_real_restored_tab(mock_info):
    mock_info.return_value = _window(
        [UIElement(name=".env. Unmodified.", control_type="TabItem", rect=(0, 0, 10, 10))]
    )
    config = _app("notepad", [], single_instance=True)
    with pytest.raises(UnexpectedSessionStateError):
        _verify_clean_single_instance_session(config)


@patch("computeruse.dataset.orchestrator.get_foreground_window_info")
def test_verify_clean_session_raises_on_more_than_one_tab_even_if_all_untitled(mock_info):
    mock_info.return_value = _window(
        [
            UIElement(name="Untitled", control_type="TabItem", rect=(0, 0, 10, 10)),
            UIElement(name="Untitled", control_type="TabItem", rect=(10, 0, 20, 10)),
        ]
    )
    config = _app("notepad", [], single_instance=True)
    with pytest.raises(UnexpectedSessionStateError):
        _verify_clean_single_instance_session(config)


@patch("computeruse.dataset.orchestrator.write_samples")
@patch("computeruse.dataset.orchestrator.collect_from_current_window")
@patch("computeruse.dataset.orchestrator.get_foreground_window_info")
@patch("computeruse.dataset.orchestrator._find_and_focus_window")
@patch("computeruse.dataset.orchestrator._launch")
@patch("computeruse.dataset.orchestrator.load_registry")
def test_run_skips_app_when_session_state_is_unexpected(
    mock_load, mock_launch, mock_focus, mock_info, mock_collect, mock_write, tmp_path
):
    mock_load.return_value = [_app("notepad", [AppState(name="empty", split="train")], single_instance=True)]
    mock_info.return_value = _window(
        [UIElement(name=".env. Unmodified.", control_type="TabItem", rect=(0, 0, 10, 10))]
    )
    mock_collect.return_value = []

    ledger = run(
        registry_path=tmp_path / "apps.yaml",
        dataset_root=tmp_path,
        controller=MagicMock(),
        session_id="sess1",
    )

    assert not ledger.is_done("notepad", "empty")
    mock_collect.assert_not_called()


@patch("computeruse.dataset.orchestrator.write_samples")
@patch("computeruse.dataset.orchestrator.collect_from_current_window")
@patch("computeruse.dataset.orchestrator._find_and_focus_window")
@patch("computeruse.dataset.orchestrator._launch")
@patch("computeruse.dataset.orchestrator.load_registry")
def test_run_continues_to_next_state_when_one_state_fails(
    mock_load, mock_launch, mock_focus, mock_collect, mock_write, tmp_path
):
    mock_load.return_value = [
        _app(
            "notepad",
            [
                AppState(
                    name="broken",
                    split="train",
                    steps=[DriveStep(kind="teleport")],  # unknown kind -> raises
                ),
                AppState(name="ok", split="train"),
            ],
        )
    ]
    mock_collect.return_value = []

    ledger = run(
        registry_path=tmp_path / "apps.yaml",
        dataset_root=tmp_path,
        controller=MagicMock(),
        session_id="sess1",
    )

    assert not ledger.is_done("notepad", "broken")
    assert ledger.is_done("notepad", "ok")


@patch("computeruse.dataset.orchestrator.load_registry")
def test_run_apps_filter_restricts_to_named_apps(mock_load, tmp_path):
    mock_load.return_value = [
        _app("notepad", [AppState(name="empty", split="train")]),
        _app("calculator", [AppState(name="default", split="train")]),
    ]

    with patch("computeruse.dataset.orchestrator._launch") as mock_launch, patch(
        "computeruse.dataset.orchestrator._find_and_focus_window"
    ), patch("computeruse.dataset.orchestrator.collect_from_current_window", return_value=[]), patch(
        "computeruse.dataset.orchestrator.write_samples"
    ):
        run(
            registry_path=tmp_path / "apps.yaml",
            dataset_root=tmp_path,
            controller=MagicMock(),
            session_id="sess1",
            apps_filter=["notepad"],
        )
        assert mock_launch.call_count == 1


# ---------------------------------------------------------------------------
# geometry variants (added 2026-07-19)
# ---------------------------------------------------------------------------


def test_registry_parses_geometry_variants_inherited_from_the_app():
    from computeruse.dataset.registry import load_registry

    apps = load_registry(Path("data/gui_grounding/apps.yaml"))
    notepad = next(a for a in apps if a.name == "notepad")
    # declared once at app level, inherited by every state
    assert len(notepad.states[0].geometry_variants) == 3
    assert {v.name for v in notepad.states[0].geometry_variants} == {
        "compact",
        "medium",
        "large",
    }


def test_all_variants_of_a_state_share_that_states_split():
    # two window sizes of the same screen are near-duplicates -- putting one
    # in train and one in dev would be leakage dressed up as more data.
    # The schema enforces this structurally (split lives on the state, not
    # the variant); this test pins that so a future refactor can't move it.
    from computeruse.dataset.registry import AppState, GeometryVariant

    state = AppState(
        name="s",
        split="train",
        geometry_variants=[GeometryVariant(name="a"), GeometryVariant(name="b")],
    )
    assert not hasattr(state.geometry_variants[0], "split")
    assert state.split == "train"


def test_registry_real_file_produces_the_expected_screenshot_count():
    # guards against an accidental edit gutting the state list again: the
    # 2026-07-19 expansion took this from 32 to 212 screenshots, and 32 was
    # the single biggest cap on model performance.
    from computeruse.dataset.registry import load_registry

    apps = load_registry(Path("data/gui_grounding/apps.yaml"))
    shots = sum(
        max(1, len(s.geometry_variants)) for a in apps for s in a.states
    )
    assert shots >= 200, f"registry only yields {shots} screenshots"


def test_every_state_that_can_inherit_a_dialog_starts_by_dismissing_it():
    # states run sequentially against whatever the previous one left open;
    # without a leading escape a state silently collects the PREVIOUS
    # state's dialog and labels it as this one. See apps.yaml's authoring
    # rule. Checked for states whose own steps open something modal.
    from computeruse.dataset.registry import load_registry

    apps = load_registry(Path("data/gui_grounding/apps.yaml"))
    offenders = []
    for app in apps:
        for i, state in enumerate(app.states):
            if i == 0 or not state.steps:
                continue
            opens_dialog = any(
                s.kind == "key" and s.text and ("ctrl+" in s.text or "alt+" in s.text)
                for s in state.steps
            )
            first_is_escape = state.steps[0].kind == "key" and state.steps[0].text == "escape"
            if opens_dialog and not first_is_escape:
                offenders.append(f"{app.name}:{state.name}")
    # a small allow-list: states that deliberately build on the previous one
    allowed = {
        "calculator:scientific_mode",
        "calculator:programmer_mode",
        "calculator:date_calculation_mode",
        "calculator:navigation_open",
        "task_manager:performance_tab",
        "task_manager:app_history_tab",
        "task_manager:startup_apps_tab",
        "task_manager:details_tab",
        "task_manager:services_tab",
        "windows_terminal:two_tabs",
        "windows_terminal:three_tabs",
        "windows_terminal:split_pane",
        "windows_terminal:command_palette",
        "control_panel:appearance",
    }
    assert not (set(offenders) - allowed), f"states missing an escape reset: {sorted(set(offenders) - allowed)}"
