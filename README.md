# computer-use

An autonomous desktop agent for Windows: give it a goal (optionally starting from a screenshot), and it loops on its own — perceive the screen → decide an action → execute it (mouse/keyboard) → perceive again — until the task is done. Modeled on Anthropic's computer-use architecture, scoped to full OS control rather than just a browser.

This is a long-running, seriously documented project, not a weekend build. See `docs/` for the reasoning behind every major decision.

## Status

Phases 0–2 complete; Phase 3 (the research core) in progress. See
`docs/roadmap.md` for the full phased plan — it is the authority, this is a summary.

- **Phase 0 — research & foundations** ✅ prior art, ADRs, hypotheses pre-registered
- **Phase 1 — core harness ("hands and eyes")** ✅ UIA perception + input control
- **Phase 2 — agent loop skeleton** ✅ perceive → decide → act, with `act` re-perceiving
  and diffing before/after; the decision backend is still mocked
- **Phase 3 — GUI grounding model** 🔬 in progress. Dataset collected and audited
  (v3: 200 screenshots, 1,572 train / 555 dev), LoRA training pipeline built, and the
  four-arm ablation harness (fine-tuned grounder / zero-shot VLM / UIA-only / hybrid)
  complete and dry-run against the real dataset. The UIA baseline arm has its first
  real export: 1,591 examples, 1,588 resolved, 99.8%. **The model arms have not been
  scored yet, so there is no accuracy claim for the fine-tuned grounder** — per the
  rigor bar below, an honest negative result is acceptable and an unverified one is not.
- **Phase 4+ — brain integration, hardening, eval suite** ⏳ not started

Training moved off free-tier notebooks to a rented pod after five runs died with
their session holding a healthy loss curve and no weights; the pod also has bf16,
which retires the fp16 gradient-underflow class outright rather than working around it.

The agent does not yet drive a real desktop end-to-end with a real model: the loop
is real, the grounding model it will call is still being trained.

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
