"""
The shared State that flows through the Phase 2 LangGraph orchestrator.
Same job as the state schema at the top of any LangGraph build (PRGuard,
Aria): every node reads from it and returns updates to it, LangGraph
merges those updates in. See docs/decisions/0004-langgraph-orchestrator.md.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from ..action.schema import Action
from ..perception.screenshot import Screenshot
from ..perception.uia import UIElement


class AgentState(TypedDict):
    goal: str
    screenshot: Optional[Screenshot]
    ui_elements: list[UIElement]
    action_history: list[Action]
    done: bool
    step_count: int
    consecutive_failures: int
