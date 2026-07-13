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

Modeled on how the strongest mentor-guided student flagship projects actually run — literature review before building, milestone checkpoints, public repo, final report — not just "write documented code." Concretely, for every phase below:

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
- [x] **Phase 2 (2026-07-12) — Agent loop skeleton, brain mocked**
  - Built as a **LangGraph StateGraph** (`perceive` → `decide` → `act`, conditional loop/end) rather than a hand-rolled loop — deliberate choice to match existing production LangGraph experience (PRGuard, Aria). See ADR-0004.
  - `Brain` interface defined (the `decide` node) + `ScriptedBrain` for free integration testing
  - `act` executes via the Phase 1 controller, then re-perceives and diffs before/after screenshots — the direct structural fix for the Phase 1 incident (see `docs/research/agent-loop-architecture.md`)
  - Two independent stop conditions wired as the conditional edge after `act`: the Brain's self-declared `DONE`, and a circuit breaker (`step_count` ceiling, `consecutive_failures` ceiling) that overrides it regardless — never trusting self-declared "done" alone
  - CLI: `computeruse run "<goal>"` — runs the real graph end-to-end (verified: `computeruse run "test goal"` → 1 step, done=True)
  - 11 new unit tests (39 total passing) covering `ScriptedBrain`, `decide`, `act`, and the routing logic
  - Built one node at a time, each checked before wiring the next — not assembled in one shot
- [ ] **Phase 3 (weeks 4–9) — GUI Grounding Model: the research core** (ADR-0003)
  - Literature review: SeeClick, UGround, OS-Atlas, GUI-Actor — done, `docs/research/gui-grounding-research.md`
  - Pre-registered hypotheses H1/H2/H3 — done, `docs/research/gui-grounding-hypothesis.md`
  - Dataset design (14-app registry, 4 splits, schema, filtering rules) — done, `docs/research/gui-grounding-dataset-design.md`
  - Collector (`dataset/collector.py`), app registry (`dataset/registry.py`), driving orchestrator (`dataset/orchestrator.py`), `computeruse collect-dataset` CLI — built and tested (2026-07-12/13)
  - Dataset collection: **8/14 registry apps collected**, **frozen as v1 on 2026-07-13** — dataset card at `data/gui_grounding/README.md` (528 examples, 21 screenshots, 0 integrity issues, split sizes, and the 6 blocked apps documented as a known limitation, not silently dropped). Held-out pool is thin (1 app, all rich-tree) — v1 held-out numbers are provisional, not a completed H3 test. Revisit the 6 blocked apps in a later session before the final H1-H3 evaluation.
  - Curate a held-out split: some apps/element-types excluded from training, reserved for evaluating generalization, not memorization
  - Training pipeline built and verified (2026-07-13): `training/prepare_dataset.py` (labels -> normalized coordinate targets, verified against real collected data), `training/chat_format.py` (Qwen2-VL's real chat template, verified against the live tokenizer), `training/lora_config.py` (real target_modules verified against the actual model architecture on the meta device — 0.829% trainable params), `training/dataset.py` (real processor-produced tensors + loss masking, verified end-to-end: decoded target span exactly matches the coordinate string, nothing else leaks into the training signal), `training/train_lora.py` (Trainer wiring). 97 tests passing. Fine-tune a small open VLM (candidate: Qwen2-VL-2B class) via LoRA on the curated dataset, using free-tier cloud GPU quota (Kaggle/Colab) — notebook built, first real run not yet executed (this machine has no GPU).
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
