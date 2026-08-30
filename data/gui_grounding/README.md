# GUI Grounding Dataset (v3, current as of 2026-07-20; v2/v1 history below preserved as-is)

## v3 (2026-07-19/20): wrong-window bug, occlusion filter, geometry expansion, a third Notepad PII incident

Rendering label boxes onto their screenshots (never done before) showed 196
of 542 v2 rows (36%) were the code editor's UI mislabeled as `file_explorer`
or `audacity` — `collector.collect_from_current_window` had nothing tying
"the window it walked" to "the app it believed it was collecting," so any
focus loss mid-run (typing elsewhere, an app crashing seconds after launch)
silently redirected labeling. 100% of the dataset's only `audacity`
screenshot was wrong, which meant H3's generalization claim rested on
nothing. Full root-cause writeup: `docs/journal.md`, 2026-07-19.

**Fixes**: `uia.get_foreground_window_info()` reports the walked window's
own title/class/rect so the collector can verify it and raise
`WindowMismatchError` instead of labeling (checked before every state, not
once per app). `uia.is_visible_at_center()` hit-tests via
`UIAElementInfo.from_point` so controls hidden behind an open menu/dialog
are dropped (`calculator_0022` had 4 buttons labeled behind a flyout).
`scripts/validate_dataset.py` is a hard gate (wrong-window / out-of-bounds /
ambiguous / leaked / PII rows) — run before every upload, non-negotiable.
Registry expanded from 2-3 states/app to 6-10, each captured at 2-3 window
sizes (`registry.GeometryVariant`); variants of one state share its split so
resizes of near-duplicate screens can't leak across train/dev.

**Training-side fixes, same session**: `train_lora` loaded fp16 weights but
never set `fp16=True`, so no `GradScaler` was attached and small LoRA
gradient updates were silently underflowing to zero — the actual cause of
run-to-run instability previously (wrongly) blamed on the learning rate.
Fixed with `cast_trainable_params_to_fp32` + `fp16=True`, `warmup_ratio`,
and `load_best_model_at_end`. The notebook now evaluates the full dev split
via `eval/vlm_grounder` instead of eyeballing the same 5 examples.

**A third real PII incident (2026-07-20), and what it cost**: the 2026-07-16
`single_instance` guard only checks whether `notepad.exe` is *already
running* (via `tasklist`) before launching. Windows 11 Notepad separately
persists its own on-disk session state (`LocalState/TabState`) and restores
it on a **cold** launch — a mechanism the "already running" check cannot
see. 21 of the 24 Notepad screenshots this session (`notepad_0004`–`0024`)
opened with a real `.env` tab restored from a previous real session and
captured its tab name (`.env. Unmodified.`) into `labels.jsonl`; because the
driven states type placeholder text with no explicit "new tab" step first,
it's also possible that text was typed into that real tab's in-memory
buffer if it was the active one. Never reached GitHub or any third party —
the dataset itself (`labels.jsonl`, `images/`, the upload zip) has never
been git-tracked, and the Kaggle upload zip built from this data was caught
and deleted before upload. Quarantined (21 screenshots / 276 rows removed,
backed up then permanently deleted once confirmed) and the affected images
are gone — leaving genuine uncertainty about whether real file content was
visible in the deleted screenshots, not just the tab name, since that
wasn't verified before the backup was cleared.

**Real fix**: `orchestrator._verify_clean_single_instance_session` now
checks what actually got focused after launch, not just what was launched —
any tab whose name isn't a fresh `Untitled` document means the session was
restored, and it raises rather than closing the tab itself (closing an
unknown real file programmatically would be a worse action than refusing to
collect against it). 4 new tests in `tests/test_orchestrator.py`. Verified
live: the very next re-collection attempt hit a *second*, different
leftover tab (`demo_target.txt. Modified.`, likely residue from the Phase 1
demo scripts) and correctly refused instead of mislabeling anything —
proof the guard generalizes past the specific `.env` case it was written
for. Also found and removed during this cleanup: 9 orphaned image files on
disk with no corresponding `labels.jsonl` row (stale leftovers from earlier
collection attempts, dated 2026-07-16 and 2026-07-19) — `validate_dataset.py`
only ever checks referenced rows, not stray files, so the upload zip is now
built strictly from `labels.jsonl`'s referenced paths rather than a glob of
`images/`, to stop any future orphan from silently riding along.

**Update, same day**: Notepad re-collected clean after a full manual
close/reopen cycle — 24/24 states, no session-restore recurrence.
**Final v3 state: 3163 rows across 200 screenshots, `validate_dataset.py`
passes with 0 errors** (2 expected warnings: GIMP's 2 chrome-only shots, a
known consequence of its 5-15s crash window, not a new issue). 14/14 apps,
every currently-defined state collected. 200/210 against the pre-registered
volume target — the remaining 10 are entirely GIMP's deliberately-capped
2-state scope (see its `apps.yaml` comment), not a gap. `gui_grounding_upload.zip`
rebuilt strictly from `labels.jsonl`'s referenced paths.

| Split | Examples |
|---|---|
| `train` | 1572 |
| `dev` | 555 |
| `test_same_app` | 246 |
| `test_held_out_app` | 790 |

## v2 (2026-07-16): full 14/14 app coverage, four pipeline bugs fixed

A second PII incident during a full re-collection attempt (Windows Notepad's
single-instance tab reuse leaked real content again, and the bare
`ms-settings:` Home page rendered the signed-in account name + Wi-Fi SSID
into a captured screenshot) triggered a harder look at the whole pipeline,
which surfaced three more real bugs, all fixed before re-collecting:

- **Chrome-vs-content sampling bias**: `_sample_by_control_type` took the
  *first* N elements of each control type. UIA tree-walk order visits window
  chrome (Minimize/Maximize/Close) before real content on every app, so the
  cap was systematically keeping chrome and dropping the content it was
  meant to sample. Now samples randomly within the cap instead.
- **UIA depth cutoff too shallow for packaged apps**: `max_depth=6` was
  tuned against classic Win32 apps. Packaged/WinUI3 apps (modern Paint's
  ribbon) nest real interactive content at depth 7-8 while chrome sits at
  depth 2-4 -- the old cutoff collected chrome-only data for those apps.
  Raised to `max_depth=12`, `max_elements` 200->400.
- **Packaged-app rect-not-ready race**: a window can report a genuinely
  focused, on-screen state via UIA before its accessibility tree has
  finished laying out -- confirmed directly (Paint's rect read as
  `(0,0,0,0)` immediately after `set_focus()`, then the real rect moments
  later). `orchestrator.py` now polls until the rect is non-degenerate
  before handing off to the collector.

**Fixes for the PII incident itself**: `registry.py` gained a
`single_instance` flag (Notepad set `true`) -- the orchestrator now refuses
to launch into an already-running single-instance app rather than silently
attaching to its real window/tabs. `apps.yaml`'s settings entry now launches
`ms-settings:personalization` (a neutral page) instead of the bare home
page. `collector.py` also gained a generic PII filter (`_pii_terms` /
`_contains_pii`) that drops any element whose name contains the logged-in
username or any term from `COMPUTERUSE_DATASET_REDACT_TERMS` -- a defense
in depth, not a replacement for picking neutral app states.

**Result**: re-ran the full registry twice (once unelevated for 12/14 apps,
once from an elevated session for `task_manager` + `device_manager`, which
need admin per the same UIPI finding as before). **All 14/14 apps collected,
0 integrity issues, 570 labeled examples, 32 screenshots** -- every state
currently defined in `apps.yaml` across every app. The pre-registered 210
screenshot volume target (`scripts/inspect_dataset.py`) was sized for a
denser per-app state list than what's actually defined today; hitting 100%
of *defined* states is real progress, but is not the same claim as hitting
the original volume target -- adding more states per app is a lever still
on the table, not something this run closes.

Splits regenerated via `training/prepare_dataset.py`:

| Split | Examples |
|---|---|
| `train` | 252 |
| `dev` | 146 |
| `test_same_app` | 45 |
| `test_held_out_app` | 127 |

`test_held_out_app` now spans all 4 held-out apps (`character_map`,
`device_manager`, `audacity`, `notepad_plus_plus`) instead of just one --
H3's generalization claim finally has real cross-app diversity behind it.

The pre-v2 dataset (14 apps, 427 examples, the version H1's first training
run below did *not* use -- that one used the earlier 8-app/291-example v1)
is preserved untouched at `_pre_v2_backup_20260716T012300Z/` in case any
comparison against the pre-fix data is ever needed.

## v1 (frozen 2026-07-13, cleaned 2026-07-14)

Auto-labeled from Windows UI Automation (ADR-0003) — no manual annotation.
See `docs/research/gui-grounding-dataset-design.md` for the full design
rationale and `docs/research/gui-grounding-hypothesis.md` for what this
dataset needs to support (H1-H3).

## 2026-07-14 correction: non-instructable control types removed

The first real training run's sanity check surfaced a real data bug: a
window, its sub-panes, and a text label can all share one UIA `name`
(e.g. "Calculator"). `Window`/`Text`/`Pane`/etc. had no real entry in
`collector.py`'s `_INSTRUCTION_TEMPLATES`, so they fell back to the same
generic `"Click {name}"` phrasing while pointing at genuinely different
boxes — the model learned to output one averaged guess for that
instruction instead of resolving the contradiction (confirmed via
side-by-side spot-check: three different "Click Calculator" screenshots
all produced the identical prediction).

`filter_elements` now only keeps control types with a real instruction
template (interactive elements only — `Button`, `MenuItem`, `CheckBox`,
`RadioButton`, `Edit`, `ListItem`, `TreeItem`, `ComboBox`, `TabItem`,
`Hyperlink`). Applied retroactively to the 8 already-collected apps via
`scripts/clean_labels_control_type.py` (no re-collection needed — every
row already carried its element's `control_type`): **528 → 291 rows**.
Original preserved as a timestamped `.bak` before overwrite.

## v1 scope: 8/14 registry apps, cleaned counts

| App | Pool | Richness | States collected |
|---|---|---|---|
| Notepad | train | rich | 4 |
| Calculator | train | rich | 4 |
| File Explorer | train | rich | 4 |
| Control Panel | train | rich | 2 |
| Settings | train | rich | 2 |
| Paint | train | weak | 2 |
| VLC | train | weak | 2 |
| Character Map | held-out | rich | 1 |

**291 labeled examples, 21 screenshots, 0 integrity issues** (verified via
`scripts/inspect_dataset.py`). Control types present: exactly the 9
interactive types listed above — no containers or text labels.

Splits (screenshot-level, per the design doc's leakage-avoidance rule),
regenerated from the cleaned labels:

| Split | Examples |
|---|---|
| `train` | 147 |
| `dev` | 88 |
| `test_same_app` | 44 |
| `test_held_out_app` | 12 |

## First real training run (2026-07-14)

LoRA fine-tune of Qwen2-VL-2B-Instruct, 1 epoch, on the cleaned split
above (`notebooks/phase3_lora_training.ipynb`, run on Kaggle T4).
**Train loss 0.92-1.06, dev loss 1.066** — sane, non-broken numbers,
consistent with the buggy-data run's loss (~1.05), which is the point:
loss alone does not distinguish the two datasets. What does: a 5-example
qualitative spot-check on dev, where every prediction differed by
instruction (no repeated identical output for different instructions/
images, unlike the buggy-data run) — direct evidence the control-type
fix resolved the shortcut-learning behavior it was meant to fix.

**This is not an accuracy claim.** No metric exists yet (that's the
4-arm ablation, still unbuilt) — spot-check accuracy was mixed (one
prediction 8px off, another 170px off), expected for 147 training
examples over 1 epoch. This run's purpose was proving the training loop
and the data fix both work end-to-end, not reporting H1-H3.

## Known limitation: 5/14 apps not yet collected

Documented in `apps.yaml` inline, not silently dropped. Real environment
constraints hit during collection, not code bugs — **Windows Terminal
was fixed 2026-07-14** (its window title showed the active shell name,
never "Terminal"; orchestrator now also matches on window class
`CASCADIA_HOSTING_WINDOW_CLASS`, confirmed against the real OS). The
remaining 5:

- **Task Manager** (train, rich): launches but never produces a matching
  foreground window — COM/broker-activated rather than direct window
  creation, unlike every plain Win32 app in this registry. Unlike
  Windows Terminal, this is a genuine activation gap, not just a title
  mismatch; not yet resolved.
- **Device Manager** (held-out, rich): `WinError 740: requires elevation`
  — needs an admin account, the collection session used a standard user.
- **GIMP** (train, weak): installs and launches but the process exits
  cleanly 5-15s later regardless of a title/timeout fix — looks like a
  GTK rendering/display-access issue specific to that session.
- **Audacity, Notepad++** (held-out, weak): never installed — their
  installers needed a UAC prompt with nobody present to approve it.

**Why this matters for what this dataset can support**: the held-out
pool (`test_held_out_app`, what H3's generalization claim rests on) is
currently **1 app, entirely rich-tree** (Character Map) — zero
weak-tree held-out coverage. A held-out evaluation run on v1 tells you
whether the model generalizes to *that one app*, not whether it
generalizes to unseen apps broadly, and says nothing about weak-tree
generalization specifically. Treat any v1 held-out numbers as
provisional, not a completed H3 test, until the remaining held-out
apps are collected.

## What "frozen" means here

v1 is fixed as the training/eval target for the first real training run
and the learning exercise around it — not because collection is
finished, but to stop the dataset from being a moving target while the
training pipeline gets its first real execution. The 2026-07-14 cleanup
is a correction within that freeze (fixing a labeling bug in already-
collected data), not new collection. Revisit and re-run collection for
the remaining blocked apps in a later session (needs an admin/elevated
session for Device Manager, someone to click through 2 UAC prompts for
Audacity/Notepad++, and further investigation for Task Manager and
GIMP), then re-run `training/prepare_dataset.py` to regenerate `splits/`
before the final held-out evaluation that actually reports H1-H3.
