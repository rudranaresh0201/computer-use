# Dataset Design: Native-Windows GUI Grounding

Date: 2026-07-12
Status: Design complete, precedes collection tooling (Phase 3, ADR-0003)

This document is the concrete specification derived from
`docs/research/gui-grounding-hypothesis.md`. Every choice below exists to keep
H1–H3 testable — see the "why" under each section, not just the "what". No
collection code is written yet; this is the spec that collection code will be
built against next.

## App selection

14 apps total, split into a **train pool** (10 apps) and a **held-out pool**
(4 apps, never touched until the one final evaluation run). Both pools contain
a rich-tree/weak-tree spread — not just the train pool — because H3 needs to
be tested on held-out apps of *both* richness types, otherwise a held-out-app
result would be confounded with richness rather than isolating generalization.

Preference order for picking apps: built-in Windows apps first (zero install,
zero account setup), then free third-party apps that need install but no
login (login-gated apps like Spotify/Discord would complicate a scripted,
repeatable collection tool later, so they're excluded even though they're
realistic weak-tree examples).

| App | Pool | Richness | Why |
|---|---|---|---|
| File Explorer | train | rich | canonical Win32/WinUI hybrid, standard controls |
| Notepad | train | rich | already our Phase 1 demo target, minimal/standard controls |
| Calculator | train | rich | UWP, clean UIA tree, dense button grid (good coordinate variety) |
| Settings | train | rich | WinUI, deep nested navigation (tests tree depth handling) |
| Task Manager | train | rich | standard tabs/listviews, dynamic content (rows change) |
| Control Panel | train | rich | legacy Win32 controls, different visual era than the WinUI apps above |
| Paint | train | weak | canvas-dominant surface; toolbar is rich but canvas is not exposed via UIA |
| Windows Terminal | train | weak | text buffer is not discrete UIA elements; tabs/menu are |
| VLC | train | weak | Qt-rendered custom controls, no native UIA support |
| GIMP | train | weak | GTK toolkit on Windows, minimal/no UIA exposure |
| Device Manager | held-out | rich | standard treeview, unseen app, tests transfer on a rich tree |
| Character Map | held-out | rich | simple standard listbox, unseen app, tests transfer on a rich tree |
| Audacity | held-out | weak | wxWidgets, custom waveform + toolbar rendering, unseen weak app |
| Notepad++ | held-out | weak | Scintilla editing canvas is weak; menus/toolbar are rich — a genuinely mixed case, deliberately held out rather than trained on |

Third-party installs needed before collection starts: VLC, GIMP, Audacity,
Notepad++ (4 one-time installs, all free, no accounts). Flagging this now so
it's a known prerequisite for the next step, not a surprise mid-collection.

**Richness labels are provisional.** Per the hypothesis doc's threats-to-
validity section, "rich" vs. "weak" here is a judgment call made at
design time, not a measured property. The collection tool should record
actual UIA element counts per app so richness can be verified empirically
once real data exists, rather than trusted from this table.

## Split design

Split unit is the **screenshot**, not the individual labeled example — an
app screenshot yields many (element, bbox, instruction) examples, and all
examples from one screenshot must land in the same split. Splitting at the
example level would put two elements from the *same screenshot* in different
splits, leaking layout familiarity across the split boundary.

Four splits, not three, because the hypothesis doc requires three separate
evaluation slices and a same-app held-out slice is different from a
held-out-*app* slice:

| Split | Source | Touched during training? | Answers |
|---|---|---|---|
| `train` | train-pool apps, majority of captured states | yes, directly trained on | — |
| `dev` | train-pool apps, separate states from `train` | yes, repeatedly (hyperparameters, early stopping) | tuning only, never a reported result |
| `test_same_app` | train-pool apps, states not in `train` or `dev` | no, touched once at final eval | H3's "held-out-screenshot" half |
| `test_held_out_app` | held-out-pool apps, entirely | no, touched once at final eval | H3's "held-out-app" half, and H2 across both pools |

Target screenshot counts (targets to plan collection against, not commitments
— adjust once the tool exists and real per-app element yields are known):

- `train`: 10 apps x ~12 states each = ~120 screenshots
- `dev`: 10 apps x ~3 states each = ~30 screenshots
- `test_same_app`: 10 apps x ~2 states each = ~20 screenshots
- `test_held_out_app`: 4 apps x ~10 states each = ~40 screenshots

"States" means deliberately varied app states per app, not repeated
screenshots of one idle window — e.g. for Notepad: empty document, document
with text, Find dialog open, Save As dialog open, right-click context menu
open. This is what gives the dataset instruction and layout diversity, and
what makes `dev`/`test_same_app` a real test of "different state of a known
app" rather than a near-duplicate of a `train` screenshot.

At ~15-30 visible elements per screenshot after filtering, ~210 screenshots
total projects to roughly 3,000-5,000 labeled examples — matching the "few
thousand" scope already committed in ADR-0003 and the research note.

## Sample schema

One example = one (screenshot, element, instruction) triple. Reuses Phase 1's
`Screenshot` (`perception/screenshot.py`) and `UIElement`
(`perception/uia.py`) dataclasses directly rather than inventing a parallel
coordinate representation — bounding boxes are stored in the same real-pixel
space `Screenshot.to_real_coords` already produces, so training/eval code
converts through the exact same path production inference will use.

```jsonc
{
  "id": "notepad_0007_e12",                // {app}_{screenshot_seq}_{element_seq}
  "screenshot_id": "notepad_0007",          // groups examples from one screenshot
  "screenshot_path": "images/notepad/notepad_0007.png",
  "real_size": [1920, 1080],
  "scaled_size": [1568, 882],
  "scale_x": 1.2244897959183674,
  "scale_y": 1.2244897959183674,
  "element": {
    "name": "Save",
    "control_type": "Button",
    "automation_id": "SaveButton",
    "bbox_real": [1802, 12, 1834, 40]       // [left, top, right, bottom], real screen px
  },
  "instruction": "Click the Save button",   // template-generated, see below
  "app": "notepad",
  "app_richness": "rich",                    // provisional, see app table
  "split": "train",
  "session_id": "2026-07-13T10-22-00",       // groups screenshots from one collection run
  "source": "uia_auto_label"                  // leaves room for a future manual/other source without ambiguity
}
```

`screenshot_id` (not just `id`) is what split assignment and dedup logic key
on — it's the unit the split table above is actually defined over.

## Instruction generation

Per ADR-0003's "no manual annotation" commitment, instructions are
template-generated from `element.name` and `element.control_type` — the same
free-label-source principle SeeClick uses HTML `title`/text for, applied to
UIA's `name`/`control_type` instead. Multiple templates per control-type
family, chosen randomly per example, to avoid the model trivially learning
one rigid phrasing:

- Button/MenuItem: "Click {name}", "Press the {name} button", "Select {name}"
- CheckBox/RadioButton: "Check {name}", "Toggle the {name} option"
- Edit/TextBox: "Click the {name} field", "Select the {name} input box"
- ListItem/TreeItem: "Click on {name}", "Select {name} from the list"
- TabItem: "Switch to the {name} tab", "Open the {name} tab"
- Fallback (unmapped control types): "Click {name}"

**Known, stated limitation**: template-generated instructions have far less
paraphrase diversity than SeeClick/UGround's crawled web text (real alt-text,
hover titles, surrounding page context). This caps how well the fine-tuned
model can be expected to generalize to *phrasing* variation — it's being
trained to ground a name+type pair, not to handle open-ended natural
language. Worth stating plainly in the eventual write-up rather than
discovering it as an unexplained result later.

## Filtering rules (what makes a label valid)

Extends the existing `rect.width() > 0 and rect.height() > 0` check in
`perception/uia.py` — that check stays, these are additional gates the
collection tool must apply before an element becomes a labeled example:

1. **On-screen bounds**: bbox must intersect the actual captured monitor
   region — a UIA node can report a rect that's technically nonzero-size but
   off-screen (scrolled away, on another virtual desktop).
2. **Foreground-window only**: only elements belonging to the active,
   foregrounded window — matches `get_foreground_window_tree`'s existing
   scope, avoids labeling elements from occluded background windows.
3. **Dedup near-identical bboxes**: when a UIA tree reports a container and
   its single child at the same (or near-identical, within a few px) rect —
   a common tree redundancy — keep the more specific (deeper/leaf) node and
   drop the container, so the same screen region isn't labeled twice with
   different instructions pointing at the same pixels.
4. **Control-type coverage, not just buttons**: when sampling elements from
   a screenshot for labeling, sample across the available control-type
   vocabulary (Button, MenuItem, Edit, CheckBox, RadioButton, ListItem,
   TreeItem, ComboBox, TabItem, Hyperlink) rather than taking every element
   in tree order — a screenshot dominated by 40 toolbar buttons and 1
   checkbox shouldn't produce 40 button examples and 1 checkbox example.

This is the direct implementation of the "documented noisy-label failure
mode from the lit review" the hypothesis doc calls out — rules 1 and 2 are
the visibility filter, rules 3 and 4 are new and specific to what our own
UIA trees actually look like in practice.

## Dataset format and layout

JSONL for labels (append-friendly — collection will happen across many
separate runs over weeks, not one batch job), images stored separately as
PNG referenced by relative path (not base64-inlined — keeps the label file
small and diffable, matches SeeClick/UGround-style corpus conventions).

```
data/gui_grounding/
  images/
    <app>/<screenshot_id>.png
  labels.jsonl              # one line per example; grows as collection runs happen
  apps.yaml                 # app registry: launch method, richness label, pool (train/held_out)
  README.md                 # dataset card: how it was built, split sizes, known limitations
```

`data/gui_grounding/images/` and the full `labels.jsonl` are **not** meant
for git (binary-heavy, and the point of the public repo per the roadmap's
rigor bar is the code/methodology/results being visible, not raw pixel
data). A small `data/gui_grounding/sample/` subset (a handful of images +
matching JSONL lines) will be committed for reproducibility once collection
exists, with full dataset statistics reported in the dataset card instead of
the raw data itself.

## What's deliberately not decided here

- Exact per-app state list (which dialogs, which menu states) — that's a
  collection-tool implementation detail, not a design commitment.
- LoRA config, training procedure — already scoped at a high level in
  ADR-0003/the research note; concrete hyperparameters are a training-step
  decision, not a dataset-step one.
- Whether `app_richness` labels survive contact with real UIA element counts
  — explicitly deferred to post-collection verification per the hypothesis
  doc.
