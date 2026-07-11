import pytest

from computeruse.action.schema import Action, ActionType, ActionValidationError


def test_left_click_requires_coordinate():
    action = Action(type=ActionType.LEFT_CLICK)
    with pytest.raises(ActionValidationError):
        action.validate()


def test_left_click_with_coordinate_is_valid():
    Action(type=ActionType.LEFT_CLICK, coordinate=(10, 20)).validate()


def test_negative_coordinate_rejected():
    action = Action(type=ActionType.LEFT_CLICK, coordinate=(-1, 20))
    with pytest.raises(ActionValidationError):
        action.validate()


def test_type_requires_text():
    action = Action(type=ActionType.TYPE)
    with pytest.raises(ActionValidationError):
        action.validate()


def test_drag_requires_both_coordinates():
    action = Action(type=ActionType.LEFT_CLICK_DRAG, coordinate=(10, 10))
    with pytest.raises(ActionValidationError):
        action.validate()

    Action(
        type=ActionType.LEFT_CLICK_DRAG,
        start_coordinate=(0, 0),
        coordinate=(10, 10),
    ).validate()


def test_scroll_requires_valid_direction():
    action = Action(type=ActionType.SCROLL, scroll_direction="sideways")
    with pytest.raises(ActionValidationError):
        action.validate()

    Action(type=ActionType.SCROLL, scroll_direction="down", scroll_amount=3).validate()


def test_hold_key_requires_text_and_duration():
    with pytest.raises(ActionValidationError):
        Action(type=ActionType.HOLD_KEY, text="shift").validate()

    Action(type=ActionType.HOLD_KEY, text="shift", duration=1.0).validate()


def test_wait_and_screenshot_and_done_need_nothing():
    Action(type=ActionType.WAIT, duration=0.5).validate()
    Action(type=ActionType.SCREENSHOT).validate()
    Action(type=ActionType.DONE).validate()


def test_to_dict_round_trips_through_from_anthropic_tool_input():
    action = Action(type=ActionType.LEFT_CLICK, coordinate=(500, 300), text="shift")
    tool_input = action.to_dict()

    rebuilt = Action.from_anthropic_tool_input(tool_input)

    assert rebuilt.type == ActionType.LEFT_CLICK
    assert rebuilt.coordinate == (500, 300)
    assert rebuilt.text == "shift"


def test_from_anthropic_tool_input_matches_docs_example():
    # Straight from docs/research/anthropic-computer-use.md "Example actions"
    tool_input = {"action": "scroll", "coordinate": [500, 400], "scroll_direction": "down", "scroll_amount": 3}

    action = Action.from_anthropic_tool_input(tool_input)

    assert action.type == ActionType.SCROLL
    assert action.coordinate == (500, 400)
    assert action.scroll_direction == "down"
    assert action.scroll_amount == 3
