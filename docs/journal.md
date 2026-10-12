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

---

## 2026-07-19 — Pre-rerun audit: a real fp16 bug, and a biased spot-check

**Context**: the v2 training run kept producing bad predictions on the notebook's 5-example dev spot-check, and the 07-19 rerun landed noticeably worse than the run before it on identical data. That was initially attributed to `lr=5e-4` being too aggressive (lowered to 1e-4, epochs 1 -> 3, seed pinned). This session audited the pipeline before spending another GPU run.

**Bug 1 (the real one): training ran in raw fp16 with no loss scaling.** `train_lora.build_trainer` loaded the model with `torch_dtype=torch.float16`, but `build_training_args` never set `fp16=True`. Two consequences, neither of which errors:
- The trainable LoRA params were themselves fp16, so Adam's state and updates were fp16. Update magnitudes here are ~1e-4 against weights of ~1e-2; fp16's smallest normal value is ~6e-5, so a large share of updates rounded to zero.
- HF `Trainer` only attaches a `GradScaler` when `fp16=True`. Without it the backward pass ran unscaled and small gradients underflowed silently.

The failure mode is nasty because it looks like success: the loss is dominated by the frozen base model, so it stays in a sane range while the adapter barely moves. And *which* updates survive depends on batch order — which is exactly the run-to-run instability that got blamed on the learning rate. Note that lowering lr to 1e-4 makes underflow strictly worse, not better.

**Fix**: `cast_trainable_params_to_fp32` (keeps the frozen base in fp16 so it still fits a 15GB T4, promotes only the trainable adapter), called after `get_peft_model` and before the `Trainer` is constructed so the optimizer is built over fp32 params; plus `fp16=True` in `TrainingArguments` so autocast + `GradScaler` are actually active. This is the standard PEFT mixed-precision recipe; we had neither half of it. bf16 was considered and rejected — the T4 has no bf16 tensor cores. Guarded by tests, including one asserting the cast doesn't drop `requires_grad` (which would silently freeze the whole adapter — a worse version of the same bug).

**Also fixed**: `load_best_model_at_end` on `eval_loss` (a 3-epoch run over 252 examples was keeping epoch 3 even when epoch 2 was better — i.e. reporting the overfit checkpoint), and `warmup_ratio=0.03` (a LoRA adapter starts at exactly zero, so step 1 takes full lr against an unmoved model; the whole run is only ~189 steps).

**Bug 2: the spot-check was measuring the worst possible five examples.** `dev.jsonl` is grouped by app and the cell took `[:5]`, so every run was judged on the *same* five — three of which are Calculator's Minimize/Restore/Close titlebar buttons, adjacent targets 36 normalized units apart, visually near-identical, and exactly the window chrome the 07-16 sampling-bias fix was about. "Predictions look bad" was partly a measurement artifact.

**Fix**: the notebook now runs the **full 146-example dev split** through `eval/vlm_grounder.evaluate_arm` and reports click accuracy (the H1-H3 metric), parse rate, and median center distance, plus a *seeded random* 8-example qualitative sample instead of the first five. Added `center_distance_normalized` and `summarize_diagnostics` for this. The reasoning for distance: the median target box in this dataset is 37x44 units, 0.16% of screen area, so binary hit-rate is a nearly-all-zeros signal early in training — a model that has learned "Close lives in the top right" and one pointing at the taskbar both score 0.0. Distance separates them, so it's the right thing to steer dev decisions by. It stays a diagnostic; accuracy remains the reported metric.

**Notebook hygiene**: cell 3's robust `rglob("labels.jsonl")` dataset finder had been authored as a **markdown** cell, so it never ran — the cell that actually executed hardcoded `/kaggle/input/gui-grounding-v2`. Merged into one real code cell that also asserts exactly one `labels.jsonl` is mounted and that the split sizes are 252/146. This is not hypothetical: a screenshot of the live Kaggle session shows two datasets attached (`final_gui_grounding` and `rudragui`), which means a run could have silently trained against the stale v1 upload with nothing in the logs to say so.

**Dataset scale, stated plainly** (audited, not fixed — no code change can fix it): train is 252 rows but only **14 unique screenshots**; dev is 9. Median target is 0.16% of screen area, and a trivial always-click-center baseline scores 0.4%. This is SeeClick's task attempted on 14 images. The fixes above remove real bugs and should make runs comparable to each other, but the honest ceiling here is set by screenshot count, and more states per app in `apps.yaml` remains the highest-leverage lever left.

**Next session**: rerun with the fixed precision path, read the dev accuracy/distance numbers (not the old spot-check), and only then decide whether lr needs to go back up — the original reason for lowering it no longer applies.

---

## 2026-07-19 (later) — The actual reason the numbers were bad: 36% of the dataset was the wrong application

**How it was found**: rendering every label's bounding box onto its screenshot and *looking at the images*. That had never been done — every previous check was numeric (`inspect_dataset.py` reported "0 integrity issues" on this exact data, because it verifies schema and file existence, not whether the picture shows the app the label claims).

**The bug**: `collector.collect_from_current_window` called `screenshot.capture()` (whole desktop) and `uia.get_foreground_window_tree()` (whatever is active *right now*), with nothing tying the two together or to the app being collected. The orchestrator focused each app **once**, before its state loop, then never re-checked. So when focus was lost mid-run — a human typing in another window, GIMP/Audacity exiting seconds after launch, a drive step opening something — the collector walked the **code editor's** UIA tree and wrote its buttons to `labels.jsonl` under the target app's name, with no error.

**Damage**: 196 of 542 rows (36%), across 5 screenshots:
- all four `file_explorer` states (157 rows, 29% of the entire dataset) — boxes on VS Code's File/Edit/Selection menu and Explorer sidebar
- `audacity_0000` (39 rows) — VS Code again, with our own Claude Code session visible in frame. This was the *only* Audacity screenshot, so **100% of that held-out app was wrong**, and H3's generalization claim had been resting on it.

Three training runs were spent tuning hyperparameters against this.

**Also found in the same audit**:
- 6 screenshots contain nothing but Minimize/Restore/Close (the app's content never rendered). One of them, `calculator_0002`, supplied three of the five examples the notebook's spot-check judged every run on — so "the predictions look bad" was partly measuring three near-identical titlebar buttons 36 normalized units apart.
- 2 boxes out of frame entirely (`task_manager_0003_e25` extends to y=1102 on a 1080px screen).
- The whole dataset is **32 screenshots**. 542 "examples" is 32 images with many elements labeled on each; median target is 0.16% of screen area, and a trivial always-click-center baseline scores 0.4%.

**Fixes**:
1. `uia.get_foreground_window_info()` returns the window's title/class/rect alongside its elements, so the caller can *verify* what it got. `collector` now checks it against the expected app and raises `WindowMismatchError` rather than labeling. The orchestrator re-verifies before every state, not once per app.
2. `screenshot.capture(region=...)` + `collector`'s `crop_to_window` (opt-in, default off) crops to the target window and rebases boxes into that frame. Default off deliberately: mixing cropped and full-desktop images in one dataset is a second bug, so it's only for a full re-collection.
3. `scripts/validate_dataset.py` — a hard gate that exits non-zero on wrong-window / out-of-bounds / ambiguous / leaked / PII rows. It reproduces every finding above from `labels.jsonl` alone. Run before every upload.
4. `scripts/quarantine_bad_screenshots.py` — removes named screenshots + their rows and resets the matching ledger entries, with a backup, so only the broken apps re-collect.
5. `registry.GeometryVariant` + `apps.yaml` expansion: 2-3 states per app became 6-10, each captured at 2-3 window sizes. **32 screenshots -> 212**, which finally meets `inspect_dataset.py`'s pre-registered 210 target. Variants of one state share that state's split (two sizes of one screen are near-duplicates; splitting them would be leakage). Geometry is applied *before* the drive steps, so a resize doesn't strand an already-open dialog.
6. `run_dataset_collection.py` gained `--apps`, a hands-off countdown, and a raised rate limit — the 60/min default *raises* on breach, and at 212 screenshots it would have started silently skipping states inside the first app.

**Lesson, and it's the same one as 2026-07-10 and 2026-07-16 in a new costume**: "the code ran without error" is not "the code did the right thing". `inspect_dataset.py` passed, 103 tests passed, the loss looked sane, and a third of the data was of the wrong application. The check that found it was looking at a picture. Any pipeline that produces images should have a render-and-eyeball step wired in from day one, not reached for after three wasted GPU runs.

**Next session**: quarantine the 5 bad screenshots, re-collect (unelevated pass, then an elevated one for task_manager/device_manager), `validate_dataset.py` must pass, rebuild the upload zip, then train with the fp16 fix from earlier today.

---

## 2026-07-20 — Elevated pass closes 14/14, then a third real Notepad PII incident, root-caused and fixed

**Elevated pass**: `task_manager` + `device_manager` collected from an admin terminal (same UIPI constraint as every prior session — `IsInRole(Administrator)` verified `True` before running, not just inferred from a UAC prompt). 14/14 apps, 173 screenshots, 2653 rows, `validate_dataset.py` passed 0 errors.

**Then, rebuilding the upload zip for Kaggle, a third real PII incident was caught before it left the machine**: 21 of the 24 Notepad screenshots (`notepad_0004`–`0024`) opened with a real `.env` tab restored from a previous real session, and its tab name (`.env. Unmodified.`) got written into `labels.jsonl`. The 2026-07-16 `single_instance` guard (`_process_already_running`, checks `tasklist` for `notepad.exe`) did not fire — confirmed directly that `notepad.exe` was *not* running beforehand, so the guard had nothing to catch. Root cause: Windows 11 Notepad separately persists session state to `%LOCALAPPDATA%\Packages\Microsoft.WindowsNotepad_8wekyb3d8bbwe\LocalState\TabState` and restores it on a **cold** launch — confirmed by a `TabState` `.bin` file timestamped to the exact minute of the collection run. This is a different mechanism than "attach to an already-running window" and no existing guard checked for it.

**Blast radius, stated plainly**: never reached GitHub (the dataset — `labels.jsonl`, `images/`, the zip — has never been git-tracked at all, only `apps.yaml`/`README.md` are; the commit holding the v3 code fixes was local-only besides). The Kaggle upload zip built from the tainted data was caught and deleted before upload. The 21 screenshots and their rows were quarantined (backed up, then permanently deleted once confirmed) — which means it's no longer possible to check whether the actual `.env` file content was visible in the deleted screenshot pixels (not just the tab name), since that check wasn't done before approving the backup's deletion. Worth naming as a process mistake, separate from the original bug: verify full exposure *before* destroying the evidence, not after.

**Fix**: `orchestrator._verify_clean_single_instance_session`, called right after every `single_instance` app is launched and focused. Walks the actually-focused window's `TabItem` elements via `get_foreground_window_info()` and raises `UnexpectedSessionStateError` if there's more than one tab, or any tab whose name doesn't start with `Untitled` — i.e. it checks what got focused, not what was launched. Deliberately does not close the unexpected tab itself; programmatically dismissing an unknown real (possibly unsaved) file is worse than just refusing to collect. 4 new tests in `tests/test_orchestrator.py`; 157/157 total passing.

**The fix generalized on the very first real retry**: after the user manually closed the `.env` tab, the next collection attempt hit a *second*, different leftover tab (`demo_target.txt. Modified.` — likely residue from the Phase 1 demo scripts) and correctly refused rather than collecting anything wrong. Notepad is still only 3/24 states (the clean screenshots from before the original incident) pending a fully clean session.

**Also found during this cleanup**: 9 image files on disk with no corresponding `labels.jsonl` row — stale leftovers from earlier collection attempts (dated 2026-07-16 and 2026-07-19) that `validate_dataset.py` never catches, since it only checks referenced rows, not stray files. Deleted, and the zip-building step now writes strictly from `labels.jsonl`'s referenced paths instead of globbing `images/`, so a future orphan can't silently ride along into an upload.

**Lesson, and it's the fourth time a version of it has been written in this file**: a safety guard built for one attack path (an *already-running* process) doesn't cover a sibling path (a *cold-launch* restore) just because they produce the same symptom. When fixing a leak, ask what the *general* mechanism is, not just what would have stopped this specific instance.

**Notepad closed out same day**: two more retries hit real blockers before succeeding — a second leftover tab (`demo_target.txt`, correctly refused) and then `notepad.exe` genuinely still running (the *original*, pre-existing guard caught that one, working as designed). Once Notepad was actually closed with nothing left open, re-collection succeeded clean: 24/24 states, no session-restore recurrence. **Final v3 state: 3163 rows / 200 screenshots, `validate_dataset.py` passes 0 errors** (2 expected GIMP chrome-only warnings). 200/210 against the pre-registered volume target, with the remaining 10 being entirely GIMP's deliberately-capped scope, not a new gap. `gui_grounding_upload.zip` rebuilt clean, strictly from `labels.jsonl`'s referenced paths so orphaned files can't ride along. Full writeup in `data/gui_grounding/README.md`'s v3 section.

**Next session**: the actual Kaggle training run — dataset is validated and the fp16/eval fixes from earlier today are in place. Still an engineering validation of the fixed pipeline, not the final H1-H3 ablation (the 4-arm evaluation still doesn't exist as a completed run).

---

## 2026-08-30 — Leaving the free tier: RunPod path, bf16, and two latent bugs that would have burned the run

**Context**: five training runs lost to session death (four Kaggle, one Colab), every one with a healthy loss curve and no weights. The diagnosis in the 2026-08-10 commit was right — Draft sessions don't persist `OUTPUT_DIR` — but the mitigation (short committed chunks of 100 steps) means ~12 separate commits to finish 1,179 steps, each repeating the clone/install/model-download overhead. Decision: stop paying in wall-clock and pay ~$4 instead. Full runbook in `docs/runpod-setup.md`.

**bf16, detected not hardcoded**: `train_lora.resolve_precision()` returns bf16 when `torch.cuda.is_bf16_supported()`, else fp16. This isn't a speed tweak — it retires the entire failure class from 2026-07-19. Kaggle's T4 is Turing with no bf16 tensor cores, which forced fp16, and fp16 without a `GradScaler` is what silently underflowed the adapter updates to zero. bf16 carries fp32's exponent range, so there is nothing to underflow and no scaler in the loop at all. Detected rather than constant so the Kaggle path keeps working unchanged; `TrainingArguments(bf16=True)` refuses to construct on non-Ampere hardware, so a wrong pod fails loudly at args-construction rather than silently running emulated. `build_training_args`/`build_trainer` also take `per_device_batch_size`/`gradient_accumulation_steps`, with a guard that their product stays at 4 — the effective batch every hyperparameter, `lr=1e-4` especially, was reasoned about at.

**Two latent bugs in the Kaggle notebook, both found by static checking before spending anything, both of which would have crashed *after* the model download**:
1. `vlm_grounder.MODEL_ID` did not exist, and cell 21 (the zero-shot arm) imports it. Added it to `vlm_grounder`, with `test_model_id_matches_the_trained_base_model` guarding it against drifting from `train_lora.MODEL_ID` — if those two ever disagree, the zero-shot arm scores a different base model than the adapter was trained on and H1 silently stops meaning anything.
2. Cell 21 also calls `load_registry(KAGGLE_DATASET_ROOT / "apps.yaml")`, but the upload zip contains no `apps.yaml` — verified directly: 205 entries, `images/` + `labels.jsonl` + `splits/` only. That's correct behaviour from the 2026-07-20 fix that builds the zip strictly from `labels.jsonl`'s referenced paths, but it means the registry has to come from the repo clone. `runpod_eval.resolve_apps_yaml` looks in the repo first, dataset root second.

**`hybrid` decoupled from Windows**: `eval/hybrid.py` imported `UiaArmResult` from `uia_only`, which reaches `dataset.collector` → `perception.uia` → `pywinauto`. That chain cannot import on Linux, so the pure combiner — whose own docstring calls it deliberately GPU-free and fixture-testable — was unimportable on the machine meant to run the other three arms. Now a `TYPE_CHECKING`-only import; `from __future__ import annotations` was already there, so nothing evaluates it at runtime.

**Consequence for the arm split**: `uia_only` genuinely cannot run on the pod, so it now runs where it can (`scripts/export_uia_arm.py`, Windows) and travels to the pod as JSON. Side benefit: the arm becomes a versioned artifact instead of something recomputed differently on two machines, and `runpod_eval.py` validates the exported splits match before joining.

**Exported the UIA arm for real**: 1,591 examples across dev/test_same_app/test_held_out_app, UIA resolved a candidate on 1,588 of them, **click accuracy 99.8%**. Also dry-ran the entire non-GPU half of `runpod_eval.py` against the real dataset with a stubbed grounder — example-id sets from the UIA side and the model side are identical (1,591 = 1,591), `hybrid.combine` joins without a KeyError, and `build_report` produces all three required slices. The whole join is proven before any GPU is rented.

**Honest read on what this run can and cannot show, written before seeing the numbers**: UIA-only at 99.8% with 1,588/1,591 resolved is a ceiling by construction — `eval/uia_only.py`'s docstring already said so, and the export confirms it numerically. Hybrid defers to the grounder on 3 examples out of 1,591, so **hybrid will come out ≈ UIA-only and H2 will hold in a near-trivial way**. The load-bearing results of this run are H1 (fine-tuned vs zero-shot) and H3 (fine-tuned vs zero-shot on the four held-out apps). The write-up leads with those, and says plainly that H2's arm design measures "can text matching pick the right element from a list that definitely contains it", not "can UIA find click targets in the wild."

**Next session**: rent the pod, run it, read H1/H3.

---

## 2026-10-11 — Back on Kaggle, with checkpoints that leave the machine

**Context**: the 2026-08-30 plan was a rented pod; it never ran, and the project was parked 2026-09-13. Restarting on Kaggle's free T4s instead, which means facing the failure that killed five runs head-on rather than paying to avoid it.

**Two problems in the chunked Kaggle notebook, found by reading it before running anything**:
1. **The LR schedule was broken by design.** Each chunk set `max_steps = current_step + 400`. HF's linear schedule is built from `max_steps`, so chunk 1 would decay the learning rate to zero at step 400, and each later chunk would rebuild a different curve. No completed run ever exposed it because no run ever completed.
2. **Checkpoints only survived a fully completed notebook.** `/kaggle/working` persists only when a commit finishes, so the 2026-08-30 guard cells reduced the risk without removing it.

**Fix**: `training/hub_checkpoint.py`. Every Trainer save is pushed to a private Hugging Face repo as one atomic commit (`last-checkpoint/` for resume, `adapters/step-N/` snapshots every 100 steps, `run_state.json`); the next session pulls it back. Sessions keep the full 1179-step schedule and stop on a wall-clock budget (11 h, save every 30 min). A NaN guard stops a non-finite fp16 run without overwriting the last good checkpoint. `load_best_model_at_end` is off for this run: the best step will be chosen afterwards by dev click accuracy, because dev loss scores "137" and "927" as equally wrong for a target of "127". New notebook: `notebooks/phase3_train_kaggle.ipynb`, pinned to code commit `b525d0e`.

**Proof before spending GPU time**: a tiny real model, trained straight through vs stopped at step 3 / pushed / pulled / resumed by a fresh Trainer, ends with identical LoRA weights (atol 1e-6) and the same LR at step 4, on transformers 5.0.0 (the pinned version) and 5.13.1. The notebook's own cells 7-10 were executed locally across two simulated sessions against a fake Hub. 201 tests pass.

**Still unverified, and what checks it**: whether Kaggle exposes secrets inside a committed run, real T4 peak memory, and real s/step. Smoke commit 1 checks all three; smoke commit 2 checks resume on Kaggle itself. The token is checked in the first minute, so a failure costs nothing.

**Also found today**: the local `.venv`'s editable install still pointed at the folder's old location (`Desktop/project computer use`); reinstalled. And 98 of 3163 labels (3%) carry sentence-like UIA names ("Switch to the There's no history yet. History tab"), found by rendering a screenshot's labels. Logged as known label noise, not fixed: the v3 upload is frozen.

**Next session**: smoke ×2, then the real run.

**Same day, first smoke commits on Kaggle**: commit 1 stopped in cell 3 in 9.6 s -- the `HF_TOKEN` secret was not attached to this notebook (fail-fast working as designed). Commit 2 stopped at cell 6 -- no dataset attached. Commit 3 confirmed on Kaggle itself what local tests could not: the secret is readable inside a committed run, the token writes to the Hub (preflight commit landed), the pinned stack installs on Kaggle's Python 3.13 (torch 2.11, transformers 5.0.0, peft 0.20.0, hub 1.29.0), dataset 1572/555. Then it **OOM'd on step 1**, asking for 3.06 GiB with 11.27 GiB already allocated.

3.06 GiB is exactly 7168^2 × 16 heads × 4 bytes: one vision layer's full attention matrix. Two causes:
1. **The frozen vision tower was in the backward graph.** `gradient_checkpointing_enable()` (the Trainer calls it) and PEFT both call `enable_input_require_grads()`, which in transformers 5.0.0 hooks every sub-model's input embeddings -- including `visual.patch_embed`. A 32-layer tower with no LoRA was recomputed and back-propagated every step. Removing our own call to it changed nothing (verified: the library re-adds the hook), so the vision forward now runs under `no_grad`. Shown on a tiny real Qwen2-VL with the same LoRA config and reentrant checkpointing: 2 backward calls into the vision blocks before, 0 after, 14/14 `lora_B` still get a gradient. This is also the most likely explanation for the ~57 s/step of every earlier run -- to be confirmed by the next smoke's s/step.
2. **`precision: bf16 (device: Tesla T4)`.** `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True` and answers yes on a T4. Emulated bf16 has no tensor cores and sends SDPA to its math kernel, which materializes the full attention matrix. `resolve_precision()` now checks compute capability.

Lesson, again in a new costume: "the library function I called did what its name says" was an assumption. Both bugs were in calls whose names promised exactly what we wanted (`enable_input_require_grads`, `is_bf16_supported`); both did something broader. Code commit `d1e135a`.

**Same day, smoke run done live in the editor** (the 4th commit sat "Running" 30+ min without ever reaching the Hub preflight; run interactively, every setup cell passed in minutes -- most likely the commit was waiting for a GPU machine, not hung in our code; unconfirmed). Results on the real T4 with `d1e135a`: `precision: fp16`, 4.5 GB after model load, **8.5 GB peak (9.7 reserved) of 15.6**, **18.5 s/step** against ~57 s/step for every earlier run -- consistent with the vision tower having been in backward all along. 222 MB checkpoint pushed in 6 s. A one-off check after ~4 steps: total |lora_B| went from exactly 0 to 1213.7 and the GradScaler was still at 65536 (no skipped steps), so the adapter really trains on the real model. At this speed the full 1179 steps are ~6 h of training, so the run should fit in one session.

Running cell 9 twice in one session exposed a callback bug: `WallClockCallback.stop_reason` survived into the second `train()` call and ended it after step 1 (the log said "smoke stop at step 3" next to "step 1/6" -- the contradiction gave it away). All three callbacks now reset per-run state in `on_train_begin`. Committed runs were never affected (fresh kernel, one `train()` call); the real run was launched on `d1e135a`.

## 2026-10-11 (night) — The real run, and the eval protocol fixed before any number

**The run** (`rudra201/computer-use-grounder`, code `d1e135a`): one Kaggle session, fp16 on the T4, GPU peak flat at 8.5 GB the whole way, 20.8 s/step of training plus ~6 min per 100-step eval. Dev loss on the 120-example eval subset: train loss 1.45 at step 10, eval 0.625 at step 400 (end of epoch 1), 0.389 at step 1000 -- `best_global_step` moved with every eval through epoch 3, so no sign of overfitting. The 30-minute timer save fired as designed (a checkpoint at step 487 and 687). A `torch.utils.checkpoint` "None of the inputs have requires_grad=True" warning appeared after the step-687 save: reproduced on a tiny Qwen2-VL, it comes only from the vision blocks (run under no_grad on purpose since `d1e135a`), every LoRA matrix still gets a gradient, and it re-prints because something reset Python's warning registry -- noise, not a fault.

**Eval decisions, written down before seeing any v3 zero-shot or fine-tuned number:**
1. **A box is scored at its center.** Qwen2-VL was pretrained to ground with `<|box_start|>(x1,y1),(x2,y2)<|box_end|>`; decoded with special tokens skipped that is `(x1,y1),(x2,y2)`, and `parse_point` used to take the first pair -- scoring the zero-shot arm by the box's top-left corner, a miss on most targets for a reason unrelated to grounding. H1 would have been inflated by our parser. `answer_format` now records point/box/none per answer so the report shows which format the base model actually used.
2. **`max_new_tokens` 16 -> 48**, so a native answer that names the object before the box is not cut off mid-box. The fine-tuned arm stops at EOS after ~10 tokens; no cost to it.
3. **Greedy decoding stated explicitly** (`do_sample=False`), so a re-scored example is identical -- which is what makes the eval resumable.
4. **The adapter is chosen on dev only**, by click accuracy on a fixed seeded 200-example dev subset over every Hub snapshot (`step-00100` ... `final`), zero-shot included as step 0 of the learning curve. The choice is pushed to the Hub as `eval/selection.json` *before* any test example is scored; a resumed session must reproduce it or stop.
5. **Every accuracy gets a 95% bootstrap interval; H1 and H3 are paired differences** (same examples, both arms, resampled together). "Supported" means the interval excludes 0.

**Resumable eval** (`eval/resumable.py`): each scored example goes to `eval/results/<arm>.jsonl` on the Hub at most every 5 minutes and at the end of every arm; a new session pulls and skips what is there. Tested with FakeHub: a hard-killed session loses only its unpushed tail and the resumed results equal an uninterrupted run. The whole notebook (cells 6-10) was dry-run on CPU with a tiny Qwen2-VL, the real processor, real v3 rows and the real UIA export: smoke, eval, then a second eval session that rescored nothing and produced an identical report. Also checked there: zero-shot through `disable_adapter()` gives bit-identical logits to the base model, and switching adapters really switches them.

**The UIA arm export** moves from the ignored `runs/` to tracked `results/uia_arm_results.json` (no PII: ids, app, split, available, hit) so the Kaggle notebook can join it.

**Next**: smoke the eval notebook, then the full eval (~5,400 generations, estimate 4-6 h on a T4 -- the smoke prints the real number).

## 2026-10-12 — Eval smoke on Kaggle: the parser was still unfair to zero-shot

Smoke (4 dev examples, dev only, no test example scored) passed end to end: fp16, 4.6 GB with the adapters loaded, ~2.4 s/example including Hub pushes, full eval estimated at 3.7 h. Training finished the night before: 1179/1179 in 8.0 h, one session, last eval_loss 0.376 at step 1100 (best of the run); grad_norm rose to 3-10 over the last 70 steps at a learning rate of ~1e-6, with no loss spike -- noted, harmless under clipping at that LR.

**The smoke caught one more scoring bug before it could touch a number that matters.** Zero-shot answered 3 of 4 times with four numbers in one bracket -- `(333, 333, 400, 399)`, `(779,207,841,250)` -- a box written differently from the `(x1,y1),(x2,y2)` the 2026-10-11 fix handled. All three parsed as "no answer" and scored as misses. One of them, `(779,207,841,250)` on `settings_0013_e6`, centers at (810, 228), next to the fine-tuned model's hit at (811, 245): zero-shot was being marked wrong for its syntax, not its grounding. `parse_point` now reads `(x1, y1, x2, y2)` and `[x1, y1, x2, y2]` as a box scored at its center -- the same rule decided 2026-10-11, applied to the syntax the model actually used. The four smoke answers are now test cases, verbatim.

Also seen: the Hub holds both `adapters/step-01179` (the callback's end-of-run snapshot) and `final/` -- the same checkpoint. Both are scored on the dev subset (~8 extra minutes) and show identical results; ties go to the first, `step-01179`.

## 2026-10-12 (morning) — The ablation: H1 and H3 supported

Full eval finished in 4.9 h on one T4 (`ad1446a`). Adapter chosen on dev only (200-example seeded subset, choice pushed to the Hub before any test example): **step 1179** (identical to `final`).

**Dev click accuracy by training step** (the learning curve; zero-shot is step 0): 4.0% → 42.5% (100) → 45.5 → 53.0 → 55.0 (400) → 61.0 → 63.0 → 64.5 → 65.5 (800) → 69.5 → 68.0 → 72.5 (1100) → **73.0% (1179)**. Median miss distance 297 → 2 units (of 1000). Still rising at the last step: more steps or more data may help.

**Test, scored once** (95% bootstrap CI):

| arm | dev (555) | test_same_app (246) | test_held_out_app (790) |
|---|---|---|---|
| uia_only | 100.0% | 100.0% | 99.6% |
| zero_shot | 6.3% [4,8] | 14.2% [10,19] | 1.5% [1,2] |
| fine_tuned | 77.8% [74,81] | 64.6% [59,70] | 46.5% [43,50] |
| hybrid | 100.0% | 100.0% | 99.7% |

- **H1 supported:** trained apps, new screenshots: +50.4 pts [+43.5, +57.3], n=246.
- **H3 supported:** four apps never trained on: +44.9 pts [+41.4, +48.5], n=790; and held-out (46.5%) < same-app (64.6%), as predicted.
- Zero-shot answered in box form on 165/555, 104/246, 247/790 examples and failed to parse on 1 of 1591 -- the 2026-10-11/12 parser fixes are what made it a fair baseline.

**Read honestly:** dev 77.8% includes the 200 examples used for selection, so the test columns are the headline. UIA-only and hybrid are ~100% by construction (labels come from UIA), as written down before this run -- H2 needs a test set from apps where UIA is blind. One training run, one seed. The held-out apps still expose UIA; whether this transfers to custom-drawn UIs is the next experiment.

## 2026-10-12 — ScreenSpot, decided before any number

Next Kaggle job: the public benchmark. ScreenSpot (SeeClick, 2024): 1,272 human-written instructions on iOS/Android, macOS/Windows and four websites; checked on the real parquet (the repo moved to bevaya/ScreenSpot, the old name redirects): bbox is normalized [x1, y1, x2, y2]. Fixed before running: the adapter is the dev-chosen `step-01179` (read from `eval/selection.json`, nothing re-chosen on ScreenSpot); instructions verbatim (same shape as our prompts, "Select Nine" vs "close"); images above the training pixel budget (1568x882) scaled down, aspect kept, never up; same prediction/parsing/hit rule as the v3 ablation; both arms see identical pixels. The notebook stops itself if the fine-tuned model's first 20 answers don't parse. Dry-run on CPU with real ScreenSpot images (incl. a 1170x2532 iPhone shot) and a resumed second session: identical reports, nothing rescored.

## 2026-10-12 (afternoon) — ScreenSpot: +46.6 points, and the biggest gain is on phones

Full run, 2.1 h, adapter `step-01179`, all 1,272 examples. Click accuracy:

| arm | mob/text | mob/icon | desk/text | desk/icon | web/text | web/icon | 6-cell avg | per-example |
|---|---|---|---|---|---|---|---|---|
| zero-shot | 16.1 | 7.0 | 26.3 | 9.3 | 25.7 | 19.9 | 17.4 | 17.6 |
| fine-tuned | 82.4 | 57.6 | 69.6 | 40.0 | 75.7 | 46.1 | **61.9** | 64.2 |

Paired, fine-tuned minus zero-shot (95% CI): mobile +59.2 [54.6, 63.5] (n=502), desktop +38.0 [32.3, 43.4] (n=334), web +38.8 [33.3, 44.3] (n=436), overall +46.6 [43.6, 49.7].

Papers report the plain mean of the six cells, so **61.9** is the comparable figure. For orientation only (each paper has its own prompts, resolution and parsing): SeeClick (9.6B, ~1M examples) 53.4; ShowUI-2B (same Qwen2-VL-2B base, 256K examples) 75.1; the ShowUI paper measured zero-shot Qwen2-VL-2B at 9.3, below our 17.4 -- our box-center parsing is, if anything, generous to the baseline, so the gain is not an artifact of an under-measured zero-shot.

Read: trained only on 10 Windows apps (~1.5k examples), the adapter improved every platform, phones most (12.0 -> 71.1) -- what it learned transfers beyond Windows. Text targets beat icons everywhere (82/58, 70/40, 76/46): finding written labels transferred better than knowing what icons mean. Desktop icons are weakest (40.0) -- ScreenSpot's Windows shots are 960x540 and were not upscaled; a hypothesis, untested. Caveats: ScreenSpot v1 has known label noise (v2 exists); one training run; cross-paper numbers are approximate.

## 2026-10-12 (evening) — Test B: real Windows screens, UIA damaged on purpose

The grounder never reads the tree, so its 1,591 v3 eval answers (`eval/results/step-01179.jsonl`) can be re-scored against real UIA trees under imposed conditions, on CPU (`eval/real_tree.py`; rules fixed in its docstring before running). Tree = the named UIA elements stored per screenshot (median 13; raw trees with unnamed/decorative elements were not saved, so this tree is cleaner than a real one). The vlm column reproduces the eval exactly (64.6 / 46.5 / 77.8), a consistency check.

| test_held_out_app (n=790) | tree | vlm | fallback | snap | snap − vlm |
|---|---|---|---|---|---|
| intact | 92.8 | 46.5 | 93.0 | 58.0 | **+11.5 [9.4, 13.8]** |
| target nameless | 1.0 | 46.5 | **37.0** | 58.0 | +11.5 |
| target absent | 1.0 | 46.5 | 37.0 | 24.8 | **−21.6 [−24.7, −18.7]** |

test_same_app (n=246): tree 85.0 / vlm 64.6 / snap 67.9 intact (+3.3 [1.2, 5.7]); absent: snap −17.1.

What it says:
1. **Snapping vision onto the tree's boxes helps when the target is in the tree, named or not**: unseen apps 46.5 → 58.0. The VLM gets close; UIA supplies the exact rectangle.
2. **When the target is not in the tree, snapping hurts badly** (it moves a near-correct click onto a neighbour). Knowing whether the tree contains the target is the open problem a good hybrid must solve.
3. **Naive "tree first, else vision" is worse than vision alone once the tree is incomplete** (37.0 vs 46.5): the name matcher confidently picks a wrong element instead of saying "not here".
4. Even an intact tree is not 100% for name matching: 23.2% of same-app test targets share their exact name with another element on the screen (3.2% on held-out apps) -- real-world C5.

Not yet allowed: tuning the snap rule on these test numbers. Any new rule (e.g. snap only when the point is inside a box) is designed on dev and confirmed on test.

**Same evening: a hybrid that doesn't break when the tree is blind.** Five hybrid rules were compared on the v3 **dev** split under the three tree conditions (radius snap, inside-only snap, strict-subset tree, agreement snap, and "guarded": the tree only when exactly one element's name is exactly what was asked, else vision). Guarded won on dev (93.0 / 76.9 / 76.9 vs vision 77.8), was committed (`d665c19`), and only then run once on the test splits:

| guarded − vision (95% CI) | test_same_app (n=246) | test_held_out_app (n=790) |
|---|---|---|
| tree intact | **+20.3 [15.4, 25.6]** (85.0 vs 64.6) | **+31.4 [28.2, 34.7]** (77.8 vs 46.5) |
| target nameless / absent | −0.8 [−2.0, 0.0] | −1.1 [−1.9, −0.4] |

So on real screens: when the tree has the target, the guarded hybrid recovers most of the tree's advantage; when it doesn't, it costs about one point against vision alone -- where naive tree-first cost 9.5 and radius snap 21.6. Its intact score on unseen apps (77.8) is still below the tree alone (92.8): exact-name agreement declines many correct tree matches, so a better-calibrated guard is the obvious next step (designed on dev again).
