from unittest.mock import Mock, patch

from computeruse.action.schema import Action, ActionResult, ActionType
from computeruse.orchestrator.nodes import make_act_node, make_decide_node
from computeruse.perception.screenshot import Screenshot


def make_screenshot(png_bytes: bytes) -> Screenshot:
    return Screenshot(
        real_size=(100, 100),
        scaled_size=(100, 100),
        scale_x=1.0,
        scale_y=1.0,
        png_bytes=png_bytes,
    )


def base_state(**overrides) -> dict:
    state = {
        "goal": "test",
        "screenshot": make_screenshot(b"before"),
        "ui_elements": [],
        "action_history": [],
        "done": False,
        "step_count": 1,
        "consecutive_failures": 0,
    }
    state.update(overrides)
    return state


def test_decide_node_appends_chosen_action_to_history():
    action = Action(type=ActionType.LEFT_CLICK, coordinate=(1, 1))
    brain = Mock()
    brain.decide.return_value = action
    decide = make_decide_node(brain)

    state = base_state()
    result = decide(state)

    assert result == {"action_history": [action]}
    brain.decide.assert_called_once_with(
        goal="test",
        screenshot=state["screenshot"],
        ui_elements=[],
        action_history=[],
    )


def test_act_node_short_circuits_on_done_without_executing():
    controller = Mock()
    act = make_act_node(controller)

    state = base_state(action_history=[Action(type=ActionType.DONE)])
    result = act(state)

    assert result == {"done": True}
    controller.execute.assert_not_called()


@patch("computeruse.orchestrator.nodes.get_foreground_window_tree", return_value=[])
@patch("computeruse.orchestrator.nodes.screenshot_mod.capture")
def test_act_node_resets_consecutive_failures_when_screen_changes(mock_capture, mock_tree):
    controller = Mock()
    controller.execute.return_value = ActionResult(success=True)
    mock_capture.return_value = make_screenshot(b"after")
    act = make_act_node(controller)

    action = Action(type=ActionType.LEFT_CLICK, coordinate=(1, 1))
    state = base_state(action_history=[action], consecutive_failures=2)

    result = act(state)

    assert result["consecutive_failures"] == 0
    assert result["step_count"] == 2
    controller.execute.assert_called_once_with(action)


@patch("computeruse.orchestrator.nodes.get_foreground_window_tree", return_value=[])
@patch("computeruse.orchestrator.nodes.screenshot_mod.capture")
def test_act_node_increments_consecutive_failures_when_screen_unchanged(mock_capture, mock_tree):
    controller = Mock()
    controller.execute.return_value = ActionResult(success=True)
    mock_capture.return_value = make_screenshot(b"before")  # identical to state's current shot
    act = make_act_node(controller)

    action = Action(type=ActionType.LEFT_CLICK, coordinate=(1, 1))
    state = base_state(action_history=[action])

    result = act(state)

    assert result["consecutive_failures"] == 1


@patch("computeruse.orchestrator.nodes.get_foreground_window_tree", return_value=[])
@patch("computeruse.orchestrator.nodes.screenshot_mod.capture")
def test_act_node_increments_consecutive_failures_on_controller_failure_even_if_screen_changed(
    mock_capture, mock_tree
):
    controller = Mock()
    controller.execute.return_value = ActionResult(success=False, error="boom")
    mock_capture.return_value = make_screenshot(b"after")
    act = make_act_node(controller)

    action = Action(type=ActionType.LEFT_CLICK, coordinate=(1, 1))
    state = base_state(action_history=[action])

    result = act(state)

    assert result["consecutive_failures"] == 1
