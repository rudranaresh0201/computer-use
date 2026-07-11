# ADR-0004: Orchestrator Built as a LangGraph StateGraph

Date: 2026-07-10
Status: Accepted

## Context

Anthropic's own reference implementation runs the agent loop as a plain `while True:` Python loop with no framework (see `docs/research/anthropic-computer-use.md`). The Phase 2 orchestrator (perceive → decide → act, looping until done) needs to be built somehow, and there's a real choice between matching that minimal reference or using an agent framework.

The user has deep, production-level fluency with LangGraph specifically (PRGuard: an 8-node LangGraph autonomous PR review pipeline; Aria: a 10+ agent LangGraph assistant with human-in-the-loop approval) — earned through real production debugging, not just familiarity. A stated goal for this project is active ownership and understanding, not passive delegation (see the project journal, 2026-07-10 Phase 1 entries) — building the orchestrator in an unfamiliar shape would work against that goal for no real technical benefit.

## Decision

Build the Phase 2+ orchestrator as an actual **LangGraph `StateGraph`**: a `State` object (goal, screenshot, ui_elements, action_history, done), three nodes (`perceive`, `decide`, `act`) wired with a conditional edge (loop back to `perceive`, or route to `END` when `done`), rather than a hand-rolled loop.

The three Phase 1 modules already map directly onto tool functions a node would call: `perception/screenshot.py` + `perception/uia.py` → `perceive` node body; `action/controller.py` (wrapped with `safety.py`'s `SafetyGuard`) → `act` node body. The `decide` node is the `Brain` interface from ADR-0002 — swappable between a scripted fake (Phase 2, free testing), a local VLM, and Claude.

Built incrementally, one node at a time, each understood and checked before the next is wired in and before the full graph is assembled — not built as one large drop.

## Consequences

- One additional dependency (`langgraph`) versus Anthropic's zero-dependency reference loop — accepted, since the ownership/familiarity benefit outweighs the minor dependency cost.
- The orchestrator will look structurally identical to the user's existing production agents (PRGuard, Aria), which is the point — it lowers the barrier to genuinely understanding and extending it, rather than maintaining an unfamiliar hand-rolled loop.
- LangGraph's state-passing needs to handle a non-trivial `Screenshot` object (contains raw image bytes) — worth watching for if/when checkpointing or persistence is added later; not a blocker for the in-memory Phase 2 build.
