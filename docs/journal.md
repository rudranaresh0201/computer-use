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

---

## 2026-07-11 — Scope decision: adding a real trained-and-benchmarked research core

**Context**: compared computer-use honestly against harder peer student projects seen elsewhere — specifically a bipedal-locomotion RL project (benchmarks PPO vs SAC empirically) and from-scratch systems/hardware projects (a custom RTOS, a custom parallel processor). The honest gap identified wasn't subject matter, it was rigor: as scoped through Phase 0-2, computer-use only ever *calls* a pretrained model, it never trains or empirically benchmarks anything of its own.

**Decision made, and why not a domain pivot instead**: considered switching to RL or embedded/hardware work to chase that same rigor, rejected it — starting cold in a discipline with zero prior experience (no RL, no embedded systems background beyond the EE degree) would set the project back, not forward, and would break coherence with the rest of the portfolio (RAG, LangGraph, agentic AI — all genuinely earned expertise). Chose instead to raise computer-use's own rigor ceiling without leaving its lane.

**The actual idea**: literature review (`docs/research/gui-grounding-research.md`) found that "GUI grounding" — predicting exact click coordinates from a screenshot + instruction — is an active, real research area (SeeClick 2024, UGround ICLR'25 Oral, OS-Atlas, Microsoft's GUI-Actor), all using the same recipe: auto-label training data from structural ground truth (HTML/DOM for web, Android accessibility for mobile), then fine-tune a small VLM via LoRA. Every one of those datasets is web or mobile sourced — **none uses Windows UI Automation, none targets native desktop apps.** Phase 1's `perception/uia.py` already produces exactly the (screenshot, element, bounding box) triples this literature auto-labels from HTML — for the exact platform the literature doesn't cover. That's a real, citable, non-hand-wavy novel angle, not manufactured ambition.

**What changed**: ADR-0003 (train and benchmark a native-Windows GUI grounding model) and ADR-0004 (LangGraph orchestrator, formalizing yesterday's decision) written. Roadmap restructured: new Phase 3 (weeks 4-9) is the grounding-model research core — dataset collection tooling, curation with a held-out generalization split, LoRA fine-tuning of a small open VLM on free-tier cloud GPU quota, and a real ablation study (fine-tuned grounder vs. zero-shot VLM vs. UIA-only vs. hybrid), reported honestly even if results are mixed. Everything after shifted out accordingly (now 7 phases total, ~18+ weeks). `docs/learning.md` rewritten to point at what's actually new (LoRA mechanics, practical VLM fine-tuning) instead of the incorrect zero-background assumption it started with.

**Explicit acknowledgment of scope growth**: this roughly doubles the project's timeline. Accepted deliberately — the stated goal from the start of this project was months of real depth, not a week of API-calling, and this is what actually closes that gap rather than just asserting it.

---

## 2026-07-12/13 — Phase 3: dataset collection tooling, first collection run

**Built (2026-07-12/13)**: the dataset collection pipeline — `dataset/collector.py` (auto-labels one already-open window: filter UIA elements, generate a templated instruction, emit a `LabeledSample`), `dataset/registry.py` (parses `data/gui_grounding/apps.yaml`, the declarative 14-app spec), `dataset/orchestrator.py` (walks the registry, launches each app, drives it through registry-defined states via the Phase 1 controller, calls the collector, persists resumable progress in `progress.json`). `computeruse collect-dataset` CLI wired in `cli.py`. 73 unit tests passing, including a dedicated `test_orchestrator.py` with everything OS/UIA-facing mocked.

**2026-07-13 collection run**: ran the orchestrator against the live registry to fill in the 8 apps that had never been collected. Two real bugs found and fixed via live inspection before they could collect wrong data:
- Settings' search box reports an empty `name` (only `automation_id: CommandSearchTextBox` is stable) — same placeholder-name pattern already seen with Control Panel's search box.
- GIMP's real window is titled "GNU Image Manipulation Program", not "GIMP" — the registry's `window_title_contains: "GIMP"` was only ever matching its ~20s "GIMP Startup" splash window, a different top-level window entirely. Fixed the title match and added a per-app `launch_timeout_seconds` field to `AppConfig`/`registry.py` (default 15s, GIMP set to 30s) since a global timeout can't fit every app's startup time.

**Six apps hit real environment constraints that code fixes can't close**, and were left as documented, unretried known limitations in `apps.yaml` rather than worked around:
- **Task Manager, Windows Terminal**: launch (subprocess starts fine) but never produce a foreground window matching their expected title in this automation session — both are COM/MSIX-activated Windows 11 apps, unlike every plain Win32 app in the registry (Notepad, Calculator, Control Panel, all of which worked). Windows Terminal specifically shows the *shell's* title ("Windows PowerShell"), not "Terminal", by design — but even after ruling that out as "just" a title bug, no window at all shows up for either app reliably.
- **Device Manager**: fails outright, `WinError 740: requires elevation` — this collection account is a standard (non-admin) user, and `devmgmt.msc` needs an admin token. Deliberately not bypassed (that would mean silently working around UAC from a script).
- **GIMP**: even after the title/timeout fix above, the process exits cleanly (exit code 0) 5-15s after launch, confirmed over two separate attempts — once after briefly showing its real window, once without showing it at all. Looks like a GTK rendering/display-access issue specific to this automation session rather than a registry config problem.
- **Audacity, Notepad++**: never got installed — `winget install` triggered a UAC elevation prompt with nobody present to click "Yes", so Windows cancelled both installs after they timed out (VLC and GIMP's own installs happened to get through the same prompt cleanly; not a reliable pattern to lean on).

**Result**: 8/14 registry apps collected — notepad, calculator, file_explorer, control_panel, settings, paint, vlc, character_map. 528 labeled examples across 21 screenshots, verified via `scripts/inspect_dataset.py`: 0 integrity issues, all 4 splits populated (train 266 / dev 158 / test_same_app 84 / test_held_out_app 20), and the richness sanity check passes directionally (rich apps average 25.1 elements/screenshot vs. weak apps' 20.8).

**Honest readiness call**: not training-ready yet. `inspect_dataset.py`'s own pre-registered volume target is 210 screenshots; 21 collected is 10% of that. More importantly for the hypothesis doc specifically, the held-out pool (needed for H3, generalization to unseen apps) is down to a single app — character_map — because device_manager, audacity, and notepad_plus_plus are all currently blocked. A held-out split with one app and zero weak-richness coverage can't support a real generalization claim; training now would mean reporting H3 against essentially no held-out diversity. Unblocking the remaining apps needs either an elevated/admin collection session (Device Manager, and to let the two installs complete) or accepting a reduced-scope v1 dataset and revisiting held-out coverage before the ablation study, not before the LoRA fine-tuning code itself.

**Next session**: either re-run collection from an admin session to close the gap, or proceed to LoRA fine-tuning groundwork on the current 8-app dataset while treating the held-out numbers as provisional until the pool is filled out.

---

## 2026-07-14 — Training pipeline, a real data bug, first training run

**Built**: `training/prepare_dataset.py` (labels.jsonl -> image/prompt/target triples, normalized 0-1000 click coordinates, splits by the `split` field already on each sample), `training/chat_format.py` (Qwen2-VL chat-template wrapping), `training/lora_config.py` (verified LoRA config against the real architecture, 0.829% trainable params), `training/dataset.py` (`GroundingDataset`/`collate_fn`, plus `resolve_path` to handle Kaggle's flattened uploads), `training/train_lora.py` (`build_trainer`). Fixed a CUDA OOM on Kaggle's T4 (fp16 weights, gradient checkpointing, smaller batch) and a Qwen2-VL-specific gradient-checkpointing incompatibility (needed reentrant mode). Windows Terminal's window-matching bug (title shows the shell name, never "Terminal") was fixed for real this session — `window_class_contains` matching on `CASCADIA_HOSTING_WINDOW_CLASS`.

**Real data bug found via the first training run's own sanity check**: a window, its panes, and a text label can share one UIA `name` (e.g. "Calculator"), and `Window`/`Pane`/`Text` had no real instruction template so they all fell back to the same generic `"Click {name}"` phrasing while pointing at different boxes. Confirmed via spot-check: three different "Click Calculator" screenshots produced the identical prediction — the model learned to average the contradiction instead of resolving it. Fix: `filter_elements` now only keeps control types with a real template (interactive elements only). Applied retroactively to the 8 collected apps (528 -> 291 rows, original preserved as a `.bak`).

**First real training run**: LoRA fine-tune of Qwen2-VL-2B-Instruct, 1 epoch, Kaggle T4, on the cleaned 8-app/291-example v1 split. Train loss 0.92-1.06, dev loss 1.066 — sane numbers, and critically, a 5-example dev spot-check showed every prediction now differed by instruction (no more repeated identical output), direct evidence the control-type fix worked. Explicitly **not** an accuracy claim — spot-check accuracy was mixed (8px off, 170px off) — this run's only job was proving the training loop and the data fix both work end-to-end. v1 frozen as of this session; full writeup in `data/gui_grounding/README.md`.

---

## 2026-07-15 — All 14/14 apps collected

**Root-caused Task Manager for real**: not a COM/broker activation failure (07-13's theory) or DWM cloaking (this session's first theory, implemented then disproven) — it's a UIPI/process-integrity mismatch. `taskmgr.exe` carries an `autoElevate=true` manifest flag and silently runs at High integrity with no UAC prompt; our collector runs at Medium, and Windows enforces UIPI as a hard boundary regardless of the window's actual visibility. Same bucket as Device Manager: both need an elevated collection session, not a code fix.

**Lesson learned the hard way**: `Start-Process powershell -Verb RunAs` from inside a non-elevated shell does **not** elevate that shell — verified directly via `IsInRole(Administrator)` printing `False` even after the prompt appeared, and independently by Device Manager still throwing `WinError 740` in that "elevated" session. The real fix was opening a terminal via Start -> right-click -> "Terminal (Admin)", confirming `IsInRole(Administrator)` is `True`, *then* running the collector. **Always verify elevation directly before trusting it, never infer it from a UAC prompt having appeared somewhere upstream.**

**Result**: Windows Terminal re-collected clean (confirming the 07-14 fix works live, not just in theory). GIMP needed no further fix — the collector is fast enough to grab a screenshot before its 5-15s crash window. Task Manager and Device Manager collected from the elevated session. Audacity just needed a longer timeout on a cold `winget install` first-run. Notepad++ worked first try. **14/14 apps, 427 examples, 31 screenshots, 0 integrity issues** — held-out pool now covers all 4 held-out apps, so H3 finally has real diversity behind it, still pending the actual ablation study.

The DWM-cloak change (`_is_cloaked()`, `visible_only=False` in `orchestrator.py`) stayed in the tree uncommitted — it didn't end up being Task Manager's fix, but it's not wrong code, and turned out to matter for a different reason the next session (see below).

---

## 2026-07-16 — Second PII incident, four real pipeline bugs, v2 dataset (570 examples, 14/14 apps)

**Incident**: a full re-collection run hit two real PII leaks. Windows 11 Notepad's single-instance tab persistence reused an already-running window again (same failure class as the very first 07-10 incident), and the bare `ms-settings:` Home page renders the signed-in account name and the currently connected Wi-Fi SSID directly on screen — both got captured into a labeled screenshot before anyone noticed.

**Root-caused and fixed four real bugs, not just the two symptoms**:
1. `_sample_by_control_type` took the *first* N elements of each control type instead of sampling — UIA tree-walk order visits window chrome (Minimize/Close) before real content on every app, so capped types were silently collapsing to chrome-only data across the whole v1 dataset. Fixed: sample randomly within the cap.
2. `max_depth=6` in `perception/uia.py` was tuned against classic Win32 apps; packaged/WinUI3 apps (Paint's ribbon) nest real controls at depth 7-8 while chrome sits at 2-4 — the old cutoff collected chrome-only data for those apps too. Raised to `max_depth=12`, `max_elements` 200->400.
3. Packaged apps can report a genuinely-focused, on-screen state via UIA before their accessibility tree finishes laying out — confirmed directly (Paint's rect read `(0,0,0,0)` right after `set_focus()`, then the real rect moments later). `orchestrator.py` now polls (`_wait_for_stable_rect`) until the rect is non-degenerate before handing off to the collector.
4. The PII leaks themselves: `registry.py` gained a `single_instance` flag (Notepad set `true`) — the orchestrator now refuses to launch into an already-running single-instance app rather than silently attaching to its real content. `apps.yaml`'s settings entry now launches `ms-settings:personalization` instead of the bare home page. `collector.py` also gained a generic PII filter (`_pii_terms`/`_contains_pii`, keyed on the logged-in username plus an optional `COMPUTERUSE_DATASET_REDACT_TERMS` env var) as defense in depth.

**Re-collection**: backed up the pre-fix 14-app/427-example dataset to `_pre_v2_backup_20260716T012300Z/` before touching anything. Re-ran the full registry unelevated first (12/14 apps, 447 examples — file_explorer alone went 77 -> 157 examples thanks to the deeper tree walk), then from an elevated admin terminal for the two apps that still need it (`task_manager`, `device_manager` — same UIPI finding as 07-15, still requires elevation, not fixable in code). **Final v2 state: 14/14 apps, 570 labeled examples, 32 screenshots, 0 integrity issues** — every state currently defined in `apps.yaml`. Splits regenerated (train 252 / dev 146 / test_same_app 45 / test_held_out_app 127); `test_held_out_app` now spans all 4 held-out apps. Full writeup in `data/gui_grounding/README.md`. All 103 tests still pass with the fixes in place. Rebuilt `gui_grounding_upload.zip` and updated `notebooks/phase3_lora_training.ipynb` (title, dataset slug, example counts, next-steps) to point at v2 instead of v1.

**Honest note on readiness**: 32/210 screenshots against `inspect_dataset.py`'s pre-registered volume target — that target assumed a denser per-app state list than what's actually defined in `apps.yaml` today. Hitting 100% of currently-defined states is real, but isn't the same claim as hitting the original volume target; more states per app is a lever still on the table.

**Next session**: run this v2 dataset through the Kaggle notebook (second real training run, still an engineering validation not a final result), then start the 4-arm ablation (UIA-only / zero-shot VLM / fine-tuned grounder / hybrid) that H1-H3 actually depend on.
