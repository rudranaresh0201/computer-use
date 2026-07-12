"""
LangGraph node bodies. Each function takes the current AgentState and
returns a dict of the fields it updates — the standard LangGraph node
shape. Built and tested one at a time before any graph wires them together
(see docs/decisions/0004-langgraph-orchestrator.md and the 2026-07-11
journal entry on going slow with checkpoints).
"""

from __future__ import annotations

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
