# GUI Grounding Dataset (v1, frozen 2026-07-13)

Auto-labeled from Windows UI Automation (ADR-0003) — no manual annotation.
See `docs/research/gui-grounding-dataset-design.md` for the full design
rationale and `docs/research/gui-grounding-hypothesis.md` for what this
dataset needs to support (H1-H3).

## v1 scope: 8/14 registry apps

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

**528 labeled examples, 21 screenshots, 0 integrity issues** (verified via
`scripts/inspect_dataset.py`).

Splits (screenshot-level, per the design doc's leakage-avoidance rule):

| Split | Examples |
|---|---|
| `train` | 266 |
| `dev` | 158 |
| `test_same_app` | 84 |
| `test_held_out_app` | 20 |

Richness sanity check passes directionally: rich apps average 25.1
elements/screenshot vs. weak apps' 20.8 — consistent with the design
doc's provisional richness labels, though still a small sample.

## Known limitation: 6/14 apps not yet collected

Documented in `apps.yaml` inline, not silently dropped. Real environment
constraints hit during the 2026-07-13 collection session, not code bugs:

- **Task Manager, Windows Terminal** (train, rich/weak): launch but never
  produce a matching foreground window in the collection session used —
  both are COM/MSIX-activated Windows 11 apps, unlike every plain Win32
  app in this registry, all of which worked.
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
provisional, not a completed H3 test, until the remaining 3 held-out
apps are collected.

## What "frozen" means here

v1 is fixed as the training/eval target for the first real training run
and the learning exercise around it — not because collection is finished,
but to stop the dataset from being a moving target while the training
pipeline gets its first real execution. Revisit and re-run collection
for the 6 blocked apps in a later session (needs an admin/elevated
session for Device Manager and Task Manager, and someone to click through
2 UAC prompts for Audacity/Notepad++), then re-run
`training/prepare_dataset.py` to regenerate `splits/` before the final
held-out evaluation that actually reports H1-H3.
