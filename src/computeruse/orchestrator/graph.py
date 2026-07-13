"""
Wires perceive/decide/act into the actual LangGraph StateGraph Anthropic's
own reference loop doesn't have (see docs/research/agent-loop-architecture.md):
one entry perceive, then a decide/act loop with two independent stop
conditions — the Brain's self-declared DONE, and a circuit breaker that
overrides it regardless of what the Brain says (the lesson from browser-use
issue #191: a self-declared "done" with no independent check is how you get
an agent stuck scrolling 40+ times).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..action.controller import ActionController
from ..brain.base import Brain
from .nodes import make_act_node, make_decide_node, perceive
from .state import AgentState

MAX_STEPS = 50  # matches the order of magnitude of OSWorld-style step caps
MAX_CONSECUTIVE_FAILURES = 3  # matches browser-use's default


def _route_after_act(state: AgentState) -> str:
    if state["done"]:
        return END
    if state["step_count"] >= MAX_STEPS:
        return END
    if state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
        return END
    return "decide"


def build_graph(brain: Brain, controller: ActionController) -> CompiledStateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("perceive", perceive)
    builder.add_node("decide", make_decide_node(brain))
    builder.add_node("act", make_act_node(controller))

    builder.add_edge(START, "perceive")
    builder.add_edge("perceive", "decide")
    builder.add_edge("decide", "act")
    builder.add_conditional_edges("act", _route_after_act, {"decide": "decide", END: END})

    return builder.compile()
