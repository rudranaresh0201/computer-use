# Dev Journal

One entry per working session. Short is fine — the point is a record of what happened and why, not prose.

---

## 2026-07-10 — Day 0: kickoff, research, harness proof-of-concept

**Goal for today**: go from empty folder to a scaffolded, documented project with the lowest-level primitives (screenshot + mouse control) proven to work.

**Decisions made** (full detail in `docs/decisions/`):
- Scope: full Windows OS control, not browser-only.
- Stack: Python. Interface: CLI first, web dashboard later.
- Budget: $0 for now — no paid Claude API calls until the harness works. Brain (the decision-making backend) is built as a pluggable interface from the start so we can dev against a free/local model and swap in Claude later without a rewrite.
- Perception architecture: hybrid — Windows UI Automation (via `pywinauto`) as the primary, precise signal, with vision/screenshot as fallback and ground truth. This came directly out of research (Microsoft's UFO2 project does the same thing on Windows for the same reason: pure vision is the dominant failure mode per the OSWorld benchmark, and UIA sidesteps a lot of that on native apps).

**Research done**: read Anthropic's official computer-use tool docs + their reference agent loop implementation (`loop.py`), and surveyed four prior-art projects (Microsoft UFO2, Microsoft OmniParser, OthersideAI's self-operating-computer, the OSWorld benchmark). Full notes in `docs/research/anthropic-computer-use.md` and `docs/research/prior-art.md`.

**Key thing learned**: the "agent loop" has no magic autonomy in it — the model never touches the OS. It's a plain multi-turn tool-use conversation where *our application* is responsible for screenshotting, executing actions, and reporting results back. All the hard engineering (this whole project) is in that application layer, not in prompting cleverness. Also learned that grounding (clicking the right pixel) rather than planning is the benchmark-proven hard problem in this space — that reprioritizes our own effort: get perception right before investing in fancy reasoning prompts.

**Built today**:
- Repo initialized, `docs/` structure set up (research notes, ADRs, roadmap, this journal, learning track).
- Python project scaffolded (`pyproject.toml`, venv, `src/computeruse/` package).
- `scripts/poc_hands_and_eyes.py` — no-LLM proof of concept. Confirmed on this machine: `mss` captures a real screenshot (~300KB PNG saved to `runs/poc/screenshot.png`), `pyautogui` moves the mouse programmatically. Both core primitives work before any LLM is involved.

**Also started**: a zero-background learning track (`docs/learning.md`) sequenced to the roadmap phases, since building this project doubles as learning the ML/agent-systems space from scratch.

**Next session**: start Phase 1 — build the real screenshot module (with the resize + coordinate-scaling logic that Anthropic's docs flag as the easiest thing to get subtly wrong), the input controller, and a first pass at the `pywinauto` UI Automation integration.

---

## 2026-07-10 — Phase 1: hands, eyes, UIA, safety layer

**Built**: the full Phase 1 harness — `action/schema.py` (typed `Action`/`ActionType`, mirrors Anthropic's own action vocabulary, includes `from_anthropic_tool_input` for painless Phase 3 integration), `perception/screenshot.py` (mss capture + resize + coordinate scale-back, the exact detail flagged in Day 0's research as easy to get subtly wrong), `perception/uia.py` (pywinauto foreground-window element tree — the ADR-0001 differentiator), `safety.py` (allow-list, rate limiter, JSONL action logger; kill-switch is pyautogui's own built-in FAILSAFE, not reimplemented), `action/controller.py` (executes an `Action` via pyautogui, routed through `SafetyGuard`). 28 unit tests, all passing — pyautogui mocked for controller tests, real (but side-effect-free) calls for screenshot/UIA tests.

**Incident**: the first real end-to-end demo (open Notepad, click into it via UIA, type) went wrong. `subprocess.Popen(["notepad.exe"])` was assumed to always open a fresh blank window; instead Windows 11's single-instance, multi-tab Notepad silently refocused an *existing* tab — first attempt raced the window launch and grabbed the user's Chrome window instead (typed a test string into it with nothing focused; screenshot confirmed no visible effect); second attempt correctly found Notepad but it turned out to be a pre-existing tab with the user's real file "Book1_cleaned (1).xlsx" open, not a blank one. No confirmed harm (the process closed before its buffer could be re-verified), but this was a real near-miss on the user's own data from an autonomous action.

**Root cause**: the demo script executed `type` without ever verifying its target actually had focus — exactly the failure mode Anthropic's own docs warn about ("Claude sometimes assumes outcomes of its actions without explicitly checking results"). The safety layer built earlier (allow-list, rate limit, logging) had no "verify before you act" check, because that's a Phase 2 orchestrator concern in the original plan — but this demo script itself should never have skipped it.

**Fix**: `scripts/phase1_demo.py` now creates and targets a file it owns (`runs/phase1_demo/demo_target.txt`) instead of guessing at a generic "Untitled" window title, polls until a window whose title contains that exact filename has focus, and aborts loudly (non-zero exit, explicit message) rather than acting on an unverified target — including a second verification pass between the click and the type. Re-run confirmed clean: UIA correctly found the target `[Document]` element among 50 elements in the real Notepad window, listed (and left untouched) the user's actual other open tabs (`.env`, `app.py`, `Book1_cleaned (1).xlsx`), clicked and typed successfully.

**Known issue, logged not fixed**: the demo's typed text landed as "computer-use hase 1 demo..." — missing the leading "P" of "Phase," almost certainly a click-to-type race (typed immediately after the click, before Windows fully registered focus). `success=True` was reported by the controller because pyautogui didn't error — a good concrete reminder that "the action executed without error" and "the action had the intended effect" are different claims, and the loop (Phase 2+) needs to verify outcomes, not just check for exceptions. Left unfixed deliberately for now; candidate fix is a short delay between click and type, or an explicit focus-confirmation step.

**Decision**: Phase 2's orchestrator will be built as an actual **LangGraph StateGraph** (`perceive` → `decide` → `act` nodes, conditional edge back to `perceive` or to `END`), not a hand-rolled `while True` loop, despite Anthropic's own reference using the latter. Chosen deliberately to map onto the user's existing production LangGraph experience (PRGuard, Aria) rather than build in an unfamiliar shape — this is a real architecture decision, not just a convenience call, and will get its own ADR when Phase 2 starts.

**Phase 1: done.** All roadmap checkboxes for it are satisfied: demo evidence captured (`runs/phase1_demo/`), tests passing, this write-up done.
