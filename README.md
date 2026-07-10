# computer-use

An autonomous desktop agent for Windows: give it a goal (optionally starting from a screenshot), and it loops on its own — perceive the screen → decide an action → execute it (mouse/keyboard) → perceive again — until the task is done. Modeled on Anthropic's computer-use architecture, scoped to full OS control rather than just a browser.

This is a long-running, seriously documented project, not a weekend build. See `docs/` for the reasoning behind every major decision.

## Status

Phase 0 (research & foundations) — see `docs/roadmap.md` for the full phased plan.

## Project structure

```
docs/
  roadmap.md        — living roadmap, phases and status
  research/          — notes on Anthropic's computer-use architecture and prior art
  decisions/         — ADRs: the "why" behind architecture choices
  journal.md         — running dev log, one entry per session
  learning.md        — zero-background math/ML reading track, sequenced to the roadmap
src/computeruse/     — the package
scripts/              — standalone proof-of-concept / one-off scripts
tests/                — unit tests
```

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -e .
```

## Design principles

1. **Hybrid perception**: Windows UI Automation for structured, precise element targeting, vision/screenshots as fallback and ground truth. See `docs/decisions/0001-architecture.md`.
2. **Pluggable brain**: the decision-making backend (local free model during development, Claude in production) sits behind one interface. See `docs/decisions/0002-brain-abstraction-and-cost-strategy.md`.
3. **No paid API spend until the harness is proven.** Every dollar spent later is logged in `docs/journal.md`.
4. **No architectural decision without an ADR, no research without a note, no session without a journal entry.**
