"""
Turns a validated Action into a real mouse/keyboard event via pyautogui,
routed through the SafetyGuard first. `screenshot` and `done` are no-ops
here by design: the orchestrator (Phase 2) owns taking screenshots and
ending the loop, this module only ever touches the OS.
"""

from __future__ import annotations

import time

import pyautogui

from ..safety import SafetyGuard
from .schema import Action, ActionResult, ActionType

# Kill-switch: slam the mouse into any screen corner to raise
# pyautogui.FailSafeException and abort the in-flight action. See safety.py.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


class ActionController:
    def __init__(self, safety: SafetyGuard) -> None:
        self.safety = safety

    def execute(self, action: Action) -> ActionResult:
        action.validate()

        try:
            self.safety.check(action)
        except Exception as exc:
            result = ActionResult(success=False, error=str(exc))
            self.safety.log(action, result)
            return result

        try:
            self._dispatch(action)
            result = ActionResult(success=True)
        except pyautogui.FailSafeException:
            result = ActionResult(
                success=False,
                error="kill-switch triggered: mouse moved to a screen corner",
            )
        except Exception as exc:
            result = ActionResult(success=False, error=str(exc))

        self.safety.log(action, result)
        return result

    def _dispatch(self, action: Action) -> None:
        t = action.type

        if t in (ActionType.SCREENSHOT, ActionType.DONE):
            return  # owned by the orchestrator, not the controller

        if t == ActionType.MOUSE_MOVE:
            x, y = action.coordinate
            pyautogui.moveTo(x, y, duration=0.2)

        elif t == ActionType.LEFT_CLICK:
            x, y = action.coordinate
            pyautogui.click(x, y)

        elif t == ActionType.RIGHT_CLICK:
            x, y = action.coordinate
            pyautogui.click(x, y, button="right")

        elif t == ActionType.MIDDLE_CLICK:
            x, y = action.coordinate
            pyautogui.click(x, y, button="middle")

        elif t == ActionType.DOUBLE_CLICK:
            x, y = action.coordinate
            pyautogui.doubleClick(x, y)

        elif t == ActionType.TRIPLE_CLICK:
            x, y = action.coordinate
            pyautogui.tripleClick(x, y)

        elif t == ActionType.LEFT_CLICK_DRAG:
            sx, sy = action.start_coordinate
            ex, ey = action.coordinate
            pyautogui.moveTo(sx, sy, duration=0.1)
            pyautogui.dragTo(ex, ey, duration=0.3, button="left")

        elif t == ActionType.TYPE:
            pyautogui.typewrite(action.text, interval=0.01)

        elif t == ActionType.KEY:
            keys = [k.strip() for k in action.text.split("+")]
            pyautogui.hotkey(*keys)

        elif t == ActionType.HOLD_KEY:
            pyautogui.keyDown(action.text)
            try:
                time.sleep(action.duration)
            finally:
                pyautogui.keyUp(action.text)

        elif t == ActionType.SCROLL:
            if action.coordinate is not None:
                x, y = action.coordinate
                pyautogui.moveTo(x, y, duration=0.1)
            amount = action.scroll_amount if action.scroll_amount is not None else 3
            clicks = amount if action.scroll_direction == "up" else -amount
            if action.scroll_direction in ("left", "right"):
                pyautogui.hscroll(clicks if action.scroll_direction == "right" else -clicks)
            else:
                pyautogui.scroll(clicks)

        elif t == ActionType.WAIT:
            time.sleep(action.duration if action.duration is not None else 1.0)

        else:
            raise ValueError(f"Unhandled action type: {t}")
