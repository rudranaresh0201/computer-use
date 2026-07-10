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
