# computer-use: Roadmap

Living document. Update the status line per phase as we go instead of just adding new files elsewhere. See `docs/decisions/` for the *why* behind key choices and `docs/journal.md` for the day-to-day log.

## Vision

An autonomous desktop agent: give it a goal (optionally starting from a screenshot), and it loops — perceive screen → decide action → execute (mouse/keyboard) → perceive again — until done, with no further input from the user. Scoped to full Windows desktop control, not just a browser. Built to run for months, documented and evaluated like a real system, not a weekend demo.

## Confirmed foundational decisions
- **Scope**: full OS control (any Windows app)
- **Stack**: Python
- **Interface**: CLI first, web dashboard later
- **Budget**: $0 until the harness is proven; Brain is pluggable (see ADR-0002)
- **Perception**: hybrid — Windows UI Automation + vision (see ADR-0001)

## Phases

- [x] **Phase 0 — Research & foundations** (2026-07-10)
  - Research notes: `docs/research/anthropic-computer-use.md`, `docs/research/prior-art.md`
  - ADR-0001 (hybrid perception), ADR-0002 (brain abstraction + cost strategy)
  - Repo scaffolded, Python project initialized
  - No-LLM PoC: screenshot capture (mss) + programmatic mouse move (pyautogui) confirmed working
- [ ] **Phase 1 (weeks 2–3) — Core harness: "hands and eyes"**
  - Screenshot module w/ resize + coordinate scale-back
  - Input controller (mouse/keyboard)
  - Windows UI Automation integration (pywinauto)
  - Typed `Action` schema
  - Safety layer: allow-list, rate limit, kill-switch, action logging
  - Unit tests, no LLM involved
- [ ] **Phase 2 (weeks 4–5) — Agent loop skeleton, brain mocked**
  - `Brain` interface defined
  - Scripted/fake Brain for free integration testing
  - Orchestrator loop with max-steps/timeout guardrails
  - CLI: `computeruse run "<goal>"`
- [ ] **Phase 3 (weeks 6–8) — Real brain, budget-aware**
  - Local free VLM backend (Ollama)
  - Claude backend (computer-use tool), used sparingly, cost logged
  - First mini-benchmark (handful of tasks, pass/fail)
- [ ] **Phase 4 (weeks 9–10) — Reliability & safety hardening**
  - Error/stuck-state recovery
  - Human-in-the-loop confirmation for risky actions
  - Structured run logs + replay tool
- [ ] **Phase 5 (weeks 11–14) — Web dashboard**
  - FastAPI + WebSocket backend, React frontend with live screenshot/reasoning/timeline
- [ ] **Phase 6 (weeks 15+) — Eval suite, polish, funding pitch**
  - OSWorld-style scored benchmark suite
  - Demo video + architecture writeup
  - Funding ask backed by real cost/eval data

## Reading list

See `docs/research/*.md` for notes already written. Still to read as we approach the relevant phase:
- pywinauto docs (Phase 1)
- Anthropic "Computer Use Best Practices" quickstart — trajectory recording, server-side compaction (Phase 3-4)
- OSWorld task/scoring format in detail (Phase 6)

## Learning track (parallel, not blocking)

See `docs/learning.md` for the zero-background math/ML reading track, sequenced to match these phases.
