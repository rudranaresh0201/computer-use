# computer-use: Roadmap

Living document. Update the status line per phase as we go instead of just adding new files elsewhere. See `docs/decisions/` for the *why* behind key choices and `docs/journal.md` for the day-to-day log.

## Vision

An autonomous desktop agent: give it a goal (optionally starting from a screenshot), and it loops — perceive screen → decide action → execute (mouse/keyboard) → perceive again — until done, with no further input from the user. Scoped to full Windows desktop control, not just a browser. Built to run for months, with a genuine trained-and-evaluated research contribution at its core (see ADR-0003), not just an integration of existing APIs — documented and evaluated like a real system, not a weekend demo.

## Confirmed foundational decisions
- **Scope**: full OS control (any Windows app)
- **Stack**: Python
- **Interface**: CLI first, web dashboard later
- **Budget**: $0 until the harness is proven; Brain is pluggable (ADR-0002); model training uses free-tier cloud GPU quota (ADR-0003)
- **Perception**: hybrid — Windows UI Automation + vision (ADR-0001)
- **Orchestrator**: built as a LangGraph StateGraph, not a hand-rolled loop (ADR-0004)
- **Research core**: a GUI grounding model fine-tuned on a UI-Automation-auto-labeled dataset of native Windows apps — a real gap in the published literature (SeeClick/UGround/OS-Atlas are all web/mobile-sourced), not just a wrapper around a pretrained model (ADR-0003)
- **Repo**: public on GitHub from Day 0 — https://github.com/rudranaresh0201/computer-use — commit history is itself part of the record, not just the final code dump

## Rigor bar (the actual "not a weekend project" contract)

Modeled on how SRA VJTI's Eklavya mentorship program runs real student flagship projects — literature review before building, milestone checkpoints, public repo, final report — not just "write documented code." Concretely, for every phase below:

1. **No phase is checked off without a working demo of that phase's capability** — a recorded gif/screenshot sequence or terminal transcript proving it actually runs, attached to that phase's journal entry. A checked box with no evidence doesn't count as done.
2. **Tests are a gate, not an afterthought** — a phase with planned unit/integration tests doesn't get marked complete until they exist and pass.
3. **Every phase gets a short written wrap-up in `docs/journal.md`** (what was built, what was learned, what broke) — not just one giant report saved for the very end.
4. **Every commit that completes a meaningful chunk of work gets pushed** — the public commit history should read as a continuous build log, not a single dump.
5. **Empirical claims get benchmarked, not asserted** — the grounding model phase (Phase 3) reports real ablation numbers, including if the trained model doesn't clearly beat baselines. An honest negative result is acceptable; an unverified claim is not.

## Phases

- [x] **Phase 0 — Research & foundations** (2026-07-10)
  - Research notes: `docs/research/anthropic-computer-use.md`, `docs/research/prior-art.md`
  - ADR-0001 (hybrid perception), ADR-0002 (brain abstraction + cost strategy)
  - Repo scaffolded, Python project initialized
  - No-LLM PoC: screenshot capture (mss) + programmatic mouse move (pyautogui) confirmed working
- [x] **Phase 1 (2026-07-10) — Core harness: "hands and eyes"**
  - Screenshot module w/ resize + coordinate scale-back (`perception/screenshot.py`)
  - Input controller (`action/controller.py`, pyautogui)
  - Windows UI Automation integration (`perception/uia.py`, pywinauto)
  - Typed `Action` schema (`action/schema.py`)
  - Safety layer: allow-list, rate limit, kill-switch, action logging (`safety.py`)
  - 28 unit tests, all passing
  - Real end-to-end demo (`scripts/phase1_demo.py`) — see the 2026-07-10 incident + fix writeup in `docs/journal.md`, a real lesson on verifying targets before acting, not just a formality
- [ ] **Phase 2 (weeks 2–3) — Agent loop skeleton, brain mocked**
  - Built as a **LangGraph StateGraph** (`perceive` → `decide` → `act`, conditional loop/end) rather than a hand-rolled loop — deliberate choice to match existing production LangGraph experience (PRGuard, Aria). See ADR-0004.
  - `Brain` interface defined (the `decide` node)
  - Scripted/fake Brain for free integration testing
  - Max-steps/timeout guardrails
  - CLI: `computeruse run "<goal>"`
  - Built one node at a time, each checked before wiring the next — not assembled in one shot
- [ ] **Phase 3 (weeks 4–9) — GUI Grounding Model: the research core** (ADR-0003)
  - Literature review: SeeClick, UGround, OS-Atlas, GUI-Actor — done, `docs/research/gui-grounding-research.md`
  - Extend `perception/uia.py` into a dataset-collection tool: walk a deliberate spread of common native Windows apps, auto-label (screenshot, element, bounding box, instruction) triples from UIA, filter for actually-visible elements (not just tree presence — a documented failure mode in the literature)
  - Curate a held-out split: some apps/element-types excluded from training, reserved for evaluating generalization, not memorization
  - Fine-tune a small open VLM (candidate: Qwen2-VL-2B class) via LoRA on the curated dataset, using free-tier cloud GPU quota (Kaggle/Colab)
  - Evaluation: a real ablation — our fine-tuned grounder vs. zero-shot VLM prompting vs. UIA-only vs. the hybrid — on the held-out set, reported honestly including if it doesn't clearly win
  - Write-up of the method and results as its own document, not folded silently into code comments
- [ ] **Phase 4 (weeks 10–11) — Brain integration, budget-aware**
  - Local free VLM backend (Ollama) for the general-purpose `decide` node
  - The Phase 3 grounding model plugged in as a specialized perception component the Brain can call for native-app click accuracy
  - Claude backend (computer-use tool) as the production-grade Brain option, used sparingly, cost logged
  - First mini-benchmark (handful of end-to-end tasks, pass/fail)
- [ ] **Phase 5 (weeks 12–13) — Reliability & safety hardening**
  - Error/stuck-state recovery
  - Human-in-the-loop confirmation for risky actions
  - Structured run logs + replay tool
- [ ] **Phase 6 (weeks 14–17) — Web dashboard**
  - FastAPI + WebSocket backend, React frontend with live screenshot/reasoning/timeline
- [ ] **Phase 7 (weeks 18+) — Eval suite, polish, funding pitch**
  - OSWorld-style scored benchmark suite (methodology already studied in Phase 0)
  - Demo video + full architecture + research writeup
  - Funding ask backed by real cost/eval data, including Phase 3's grounding-model results as evidence of genuine technical depth

## Reading list

See `docs/research/*.md` for notes already written. Still to read as we approach the relevant phase:
- pywinauto docs, as needed (ongoing)
- LoRA fine-tuning basics + a small-VLM fine-tuning walkthrough (Qwen2-VL or similar) — before Phase 3's training step
- Anthropic "Computer Use Best Practices" quickstart — trajectory recording, server-side compaction (Phase 4-5)
- OSWorld task/scoring format in detail (Phase 7)

## Learning track

`docs/learning.md` was originally written assuming zero ML background — that assumption was wrong (see the 2026-07-10 journal entries and project memory) and the file needs a rewrite matched to the user's actual production-level fluency. Until rewritten, treat the reading list above as the live one.
