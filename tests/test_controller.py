"""
Controller tests mock pyautogui entirely — this suite must never move the
real mouse or type on the real keyboard. See scripts/phase1_demo.py for the
supervised, real-OS verification instead.
"""

from unittest.mock import patch

import pyautogui
import pytest

from computeruse.action.controller import ActionController
from computeruse.action.schema import Action, ActionType
from computeruse.safety import ActionLogger, ActionNotAllowed, SafetyConfig, SafetyGuard


def make_controller(tmp_path, allowed=None):
    config = SafetyConfig(allowed_actions=allowed or set(ActionType))
    guard = SafetyGuard(config, ActionLogger(tmp_path / "log.jsonl"))
    return ActionController(guard), guard


def test_left_click_dispatches_to_pyautogui(tmp_path):
    controller, _ = make_controller(tmp_path)
    with patch("computeruse.action.controller.pyautogui.click") as mock_click:
        result = controller.execute(Action(type=ActionType.LEFT_CLICK, coordinate=(100, 200)))

    assert result.success is True
    mock_click.assert_called_once_with(100, 200)


def test_type_dispatches_to_pyautogui(tmp_path):
    controller, _ = make_controller(tmp_path)
    with patch("computeruse.action.controller.pyautogui.typewrite") as mock_type:
        result = controller.execute(Action(type=ActionType.TYPE, text="hello"))

    assert result.success is True
    mock_type.assert_called_once_with("hello", interval=0.01)


def test_key_splits_hotkey_combo(tmp_path):
    controller, _ = make_controller(tmp_path)
    with patch("computeruse.action.controller.pyautogui.hotkey") as mock_hotkey:
        result = controller.execute(Action(type=ActionType.KEY, text="ctrl+s"))

    assert result.success is True
    mock_hotkey.assert_called_once_with("ctrl", "s")


def test_disallowed_action_never_reaches_pyautogui(tmp_path):
    controller, _ = make_controller(tmp_path, allowed={ActionType.SCREENSHOT})
    with patch("computeruse.action.controller.pyautogui.click") as mock_click:
        result = controller.execute(Action(type=ActionType.LEFT_CLICK, coordinate=(1, 1)))

    assert result.success is False
    assert "not in the allow-list" in result.error
    mock_click.assert_not_called()


def test_failsafe_exception_is_caught_and_logged_not_raised(tmp_path):
    controller, _ = make_controller(tmp_path)
    with patch(
        "computeruse.action.controller.pyautogui.click",
        side_effect=pyautogui.FailSafeException,
    ):
        result = controller.execute(Action(type=ActionType.LEFT_CLICK, coordinate=(1, 1)))

    assert result.success is False
    assert "kill-switch" in result.error


def test_invalid_action_is_rejected_before_dispatch(tmp_path):
    controller, _ = make_controller(tmp_path)
    with patch("computeruse.action.controller.pyautogui.click") as mock_click:
        with pytest.raises(Exception):
            controller.execute(Action(type=ActionType.LEFT_CLICK))  # missing coordinate

    mock_click.assert_not_called()
