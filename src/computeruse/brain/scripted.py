"""
A fixed-playback Brain: returns Actions from a predetermined list, in
order, ignoring the actual screen/goal entirely. Exists to unit-test the
LangGraph orchestrator's wiring (state flow, edges, termination) without
depending on a real model — the same role a stubbed LLM call plays when
testing a graph in PRGuard/Aria.
See docs/decisions/0002-brain-abstraction-and-cost-strategy.md.
"""

from __future__ import annotations

from ..action.schema import Action
from ..perception.screenshot import Screenshot
from ..perception.uia import UIElement
from .base import Brain


class ScriptExhaustedError(RuntimeError):
    """Raised when decide() is called more times than the script provides for."""


class ScriptedBrain(Brain):
    def __init__(self, script: list[Action]) -> None:
        self._script = script
        self._index = 0

    def decide(
        self,
        goal: str,
        screenshot: Screenshot,
        ui_elements: list[UIElement],
        action_history: list[Action],
    ) -> Action:
        if self._index >= len(self._script):
            raise ScriptExhaustedError(
                f"ScriptedBrain.decide() called {self._index + 1} times "
                f"but script only has {len(self._script)} actions"
            )
        action = self._script[self._index]
        self._index += 1
        return action
