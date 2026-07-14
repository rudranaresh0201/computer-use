# GUI Grounding Dataset (v1, frozen 2026-07-13, cleaned 2026-07-14)

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
