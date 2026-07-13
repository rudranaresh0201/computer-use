"""
LangGraph node bodies. Each function takes the current AgentState and
returns a dict of the fields it updates — the standard LangGraph node
shape. Built and tested one at a time before any graph wires them together
(see docs/decisions/0004-langgraph-orchestrator.md and the 2026-07-11
journal entry on going slow with checkpoints).
"""

from __future__ import annotations

from typing import Callable

from ..action.controller import ActionController
from ..action.schema import ActionType
from ..brain.base import Brain
from ..perception import screenshot as screenshot_mod
from ..perception.uia import get_foreground_window_tree
from .state import AgentState


def perceive(state: AgentState) -> dict:
    """The 'eyes' node: look at the screen, both ways (screenshot + UIA),
    same hybrid perception as ADR-0001. Doesn't touch action_history or
    done — just refreshes what the agent can currently see."""
    shot = screenshot_mod.capture()
    elements = get_foreground_window_tree()
    return {
        "screenshot": shot,
        "ui_elements": elements,
        "step_count": state["step_count"] + 1,
    }


def make_decide_node(brain: Brain) -> Callable[[AgentState], dict]:
    """The 'choose' node: ask the given Brain what to do next, given the
    current state, and record its answer. Which Brain answers (scripted
    today, a real model later) is bound once here via closure, since
    LangGraph node functions only ever receive (state,) — the same shape
    you'd inject a bound LLM client into a node in PRGuard/Aria."""

    def decide(state: AgentState) -> dict:
        action = brain.decide(
            goal=state["goal"],
            screenshot=state["screenshot"],
            ui_elements=state["ui_elements"],
            action_history=state["action_history"],
        )
        return {"action_history": state["action_history"] + [action]}

    return decide


def make_act_node(controller: ActionController) -> Callable[[AgentState], dict]:
    """The 'hands' node: execute the action decide() just chose, then
    re-perceive and compare before/after — "no exception" and "had the
    intended effect" are different claims (the 2026-07-10 Phase 1 incident).
    Which ActionController to use is bound once via closure, same reasoning
    as make_decide_node's Brain."""

    def act(state: AgentState) -> dict:
        action = state["action_history"][-1]

        if action.type == ActionType.DONE:
            return {"done": True}

        before = state["screenshot"]
        result = controller.execute(action)

        shot = screenshot_mod.capture()
        elements = get_foreground_window_tree()
        screen_changed = shot.png_bytes != before.png_bytes

        if result.success and screen_changed:
            consecutive_failures = 0
        else:
            consecutive_failures = state["consecutive_failures"] + 1

        return {
            "screenshot": shot,
            "ui_elements": elements,
            "step_count": state["step_count"] + 1,
            "consecutive_failures": consecutive_failures,
        }

    return act
