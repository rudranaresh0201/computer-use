# When to Trust the Accessibility Tree: Tree, Vision and Hybrid Grounding for Desktop Agents

*Workshop paper draft v0 (2026-10-18). Every number is from `docs/journal.md`. Cells marked
**[PENDING Ex]** wait on the experiment of that id in `docs/research/experiment-plan.md`. The venue is
not chosen yet (ICLR/CVPR 2027 workshops, dates unverified). This goes to Overleaf once the pending cells are filled.*

---

## Abstract

Computer-use agents must turn an instruction such as "click Save" into a point on the screen. Two sources
can supply it: the accessibility tree, which is exact when an element is exposed and named, and a vision
model, which works on any pixels but can miss. Systems increasingly combine the two, but when to trust each
source has not been measured. We train a 2B-parameter vision grounder (Qwen2-VL-2B + LoRA) on 1,572
examples labelled automatically by Windows UI Automation, on one free T4 GPU. On four Windows apps it never
saw, accuracy rises from 1.5% to 46.5%, and on ScreenSpot-v2 from 19.2 to 62.8. We then compare
tree, vision and hybrid methods per condition, with ground truth independent of both, on generated pages
(five conditions) and on real Windows screens with the tree damaged on purpose. Three findings follow. The
tree wins only when the target has a clean name. The common "tree first, else vision" merge is *worse than
vision alone* once the tree is incomplete (−9.5 to −12 points). And a simple gate (an LLM reads the tree,
and its pick counts only when that element has a name, otherwise vision decides) reaches 85.2% against
66.4% for vision, chosen on dev and tested once. [PENDING E4: result on real UIA-blind targets.]
[PENDING E3: how often real apps are blind.]

## 1. Introduction

An agent that operates a GUI needs *grounding*: the map from a description to screen coordinates. Desktop
platforms expose an accessibility tree (UI Automation on Windows), a list of elements with roles, names and
boxes. When the tree has the element and names it, grounding is a lookup. But trees are often incomplete: icon
buttons without names, custom-drawn widgets, canvas content (spreadsheet grids, CAD viewports, drawing
surfaces). Pure-vision grounders (SeeClick, ShowUI, UGround, OS-Atlas) avoid the tree entirely. Agent systems
such as UFO2 merge both sources.

What is missing is a measurement of **when each source should be trusted**. A tree that is present but
incomplete is dangerous in a specific way: a name matcher does not say "not here", it confidently returns a
wrong element. We isolate that effect.

**Contributions.**
1. **A cheap UIA-supervised grounder.** 1,572 automatically labelled examples from 10 Windows apps, one free
   T4. +44.9 points on unseen apps and +46.1 on ScreenSpot-v2, including +57.5 on phones it never trained on (§4.1).
2. **A controlled, per-condition comparison** of tree, vision and hybrids with ground truth from neither
   method: generated pages in five conditions (Test A), and real Windows trees damaged on purpose (Test B) (§4.2-4.3).
3. **The naive hybrid fails.** "Tree first, else vision" is worse than vision alone whenever the tree is
   incomplete, on both test beds (§4.2-4.3).
4. **A gate that works.** Abstention-based guards (`llm_guarded`, pre-registered; `llm_named`, chosen on
   dev) recover the tree's advantage on clean names without paying for its blindness (§4.4).
5. [PENDING E3] **A per-app measurement of UIA blindness** across ~20 Windows apps.

## 2. Related work

**Vision grounders.** SeeClick introduced ScreenSpot. ShowUI-2B shares our base model and uses ~256K
examples (75.1 on ScreenSpot, cross-paper). UGround and Aria-UI scale synthetic data. OS-Atlas synthesises desktop
data from accessibility trees at scale, so *using tree labels to train a grounder is not new*; ours is the
same idea at very small, free scale. GroundCUA provides 56K human-annotated desktop screenshots. WinClick/WinSpot
target Windows specifically. **Hybrid systems.** UFO2 merges UIA and OmniParser detections by IoU. OSWorld
and OS-Harm compare observation types (tree, screenshot, both) at the task level with frontier models.
ScaleCUA notes trees can be present but wrong. **Gap.** None of these, as far as we checked, measures when the
tree's answer should not be trusted, or shows that a naive merge can underperform vision alone.
[TODO before submission: read the UFO2 hybrid ablation and OSWorld's observation ablation in full; check 2026
work on grounding confidence/abstention.]

## 3. Setup

### 3.1 Grounder

**Data.** 14 Windows apps, 200 screenshots, 3,163 targets. Each target is a named UIA element with its box,
plus the instruction "click <name>" generated from the name. Splits: train 1,572 / dev 555 (10 apps),
test_same_app 246 (those apps, new screenshots), test_held_out_app 790 (4 apps never trained on: Audacity,
Character Map, Device Manager, Notepad++). Collection fixes are logged in the journal: wrong-window labels,
occlusion, PII scrubbing.

**Model.** Qwen2-VL-2B-Instruct, LoRA r=16, α=32, dropout 0.1 on the language model's attention and MLP
projections. The vision tower is frozen and run without autograd. lr 1e-4, warmup 3%, effective batch 4, 3
epochs (1,179 steps), fp16 on a T4, ~7 h. Points are predicted in 0-1000 normalised coordinates.

**Metric.** A *hit* is a predicted point inside the target's box. The same rule is used for every method.

**Protocol.** Checkpoints are chosen on a seeded 200-example dev subset, and the choice is pushed before
any test example is scored. Hypotheses are pre-registered in commits before data exists (H1-H6). Each test
set is scored once. Intervals are paired bootstrap 95% CIs, resampled by **screenshot** because targets on one
screen are correlated. [PENDING E1: second training seed.]

### 3.2 Methods compared

| method | rule |
|---|---|
| tree | exact name match in the accessibility tree |
| vision | the fine-tuned grounder's point |
| fallback | tree if a name matches, else vision (the common merge) |
| snap | vision's point, moved into the nearest tree box |
| guarded | tree only if exactly one element's name equals the instruction, else vision |
| tree_llm | an LLM (gpt-oss-120b) reads the tree as text, answers one element or NONE |
| llm_guarded | tree_llm's pick, else vision on NONE (pre-registered) |
| llm_named | tree_llm's pick only if that element has a name, else vision (chosen on dev) |

### 3.3 Test beds

**Test A (generated pages).** Sandbox web pages whose truth comes from the DOM/canvas, not from either method.
There are five conditions, checked on the rendered tree and pixels: C1 named, C2 unnamed icon, C3 reworded
instruction, C4 canvas-drawn, C5 ambiguous name. Dev seeds are 0-11 and 100-159; the frozen test is seeds
1000-1199 (1,000 items, 200 per condition).

**Test B (real screens, damaged tree).** The grounder's saved answers on the v3 test splits, re-scored against
the real UIA tree in three states: intact, target nameless, target absent. Limitation: the stored tree holds
only named elements, so it is cleaner than a live one.

**Test C (real blind targets).** [PENDING E4] 8 apps captured (incl. GIMP, which exposes 7 UIA elements),
~250 boxes drawn by hand.

## 4. Results

### 4.1 UIA-labelled training transfers

| | zero-shot | fine-tuned | Δ [95% CI, screenshot-clustered] |
|---|---|---|---|
| test_same_app (n=246) | 14.2 | 64.6 | +50.4 [43.0, 57.8] |
| test_held_out_app (n=790) | 1.5 | 46.5 | +44.9 [41.1, 48.7] |
| ScreenSpot-v2, 6-cell avg (n=1,272) | 19.2 | 62.8 | +46.1 [42.9, 49.1] (per example) |

The gain holds on every unseen app (Audacity 3.1→48.0, Character Map 1.7→52.1, Device Manager 0.0→37.2,
Notepad++ 1.6→51.0). On ScreenSpot-v2 it holds on every platform: mobile +57.5, desktop +38.3, web +38.9. Text
beats icons everywhere (desktop icons 40.0). The dev learning curve is still rising at the last step (4.0 →
73.0%). [PENDING E1: seed 7 numbers; E6: ShowUI-2B on our splits; E7: data-scaling curve.]
[PENDING E2: same-app result with near-duplicate test screenshots removed.]

### 4.2 Test A: who wins each condition

| method | C1 named | C2 unnamed | C3 reworded | C4 canvas | C5 ambiguous | mean |
|---|---|---|---|---|---|---|
| tree | **100** | 0 | 0 | 0 | 26 | 25.3 |
| vision (zero-shot) | 60 | 8 | 0 | 48 | 40 | 31.2 |
| vision (fine-tuned) | 94 | 40 | 31 | **99** | 65 | 65.8 |
| fallback | 100 | 38 | 30 | 76 | 26 | 53.8 |
| snap | 94 | 41 | 33 | 99 | 70 | 67.6 |

The tree wins only C1 (+6.0 [3.0, 9.5], H4). Vision wins unnamed icons (+40.0) and canvas (+99.0, H5).
**Fallback loses 12.0 points [9.6, 14.7] to vision**: when no name matches exactly it falls back correctly, but on
C4 and C5 a confidently wrong name match overrides a correct vision answer.

### 4.3 Test B: the same failure on real Windows trees

Held-out apps (n=790). Vision is 46.5 in every state because it never reads the tree.

| tree state | tree | fallback | snap | guarded |
|---|---|---|---|---|
| intact | 92.8 | 93.0 | 58.0 | 77.8 (+31.4 [27.1, 35.9]) |
| target nameless | 1.0 | 37.0 | 58.0 | 45.4 (−1.1) |
| target absent | 1.0 | 37.0 | 24.8 (−21.6) | 45.4 (−1.1) |

When the tree is blind, the naive merge falls 9.5 points below vision. Snapping helps only if the target is
in the tree, and costs 21.6 if it is not. `guarded` keeps most of the tree's gain and costs about one point
when blind. Also, 23.2% of same-app targets share their exact name with another element, which is the real-world
version of C5.

### 4.4 A better gate

Same 250-item subset of Test A (first 50 test pages). The LLM uses 138,838 tokens.

| method | C1 | C2 | C3 | C4 | C5 | mean |
|---|---|---|---|---|---|---|
| vision | 98 | 38 | 28 | 100 | 68 | 66.4 |
| tree_llm | 88 | 2 | 88 | 0 | 100 | 55.6 |
| llm_guarded (pre-registered) | 92 | 32 | 92 | 100 | 100 | 83.2 |
| **llm_named** (dev-chosen) | 94 | 38 | 94 | 100 | 100 | **85.2** |

llm_named − vision +18.8 [15.2, 22.4]; − llm_guarded +2.0 [0.4, 4.0]. The LLM fixes wording and ambiguity (C3,
C5). Its NONE hands canvas targets to vision (C4). Requiring a *named* pick removes llm_guarded's one flaw:
the LLM nominating a nameless button for an unnamed icon. llm_named is never below vision on any condition.
[PENDING E5: llm_named on Test B's real screens. PENDING E4: on Test C.]

### 4.5 How the grounder misses

On unseen apps half the misses land within 50 units (of 1,000) of the right box: the right area, the wrong
neighbour. On ScreenSpot two-thirds are far, meaning the instruction was not understood. Target size dominates:
hit rate is 15% for the smallest quarter of boxes and 65% for the largest. [PENDING E10: zoom-in second pass.]
[PENDING E15: named vs nameless targets on Test C, the teacher-blindness gap.]

## 5. Limitations

- One training seed so far [E1]. Only four unseen apps; app-level intervals are wide ([20.9, 41.6] for guarded).
- **The teacher only labels what it names.** Every training target is a named UIA element, yet the
  grounder's role is the nameless ones. Unnamed icons remain vision's weakest cell (38-40) [E15, E16].
- Test A is synthetic. Test B's tree is cleaner than a live tree. Real blind targets are pending [E4].
- The LLM gate was confirmed on 250 synthetic items only [E5]. It costs an LLM call per grounding.
- Element-level only: no task success rates [E13, E14].
- Instructions are template-generated from names; reworded instructions (C3) score 31 for vision [E9].

## 6. Conclusion

The tree is a precise instrument with a blind spot it does not report. Merging it naively with vision imports
that blind spot as confident errors. The fix is not a better merge rule but a better *abstention* signal:
let the tree answer only when it can name what it found, and hand everything else to vision. A small grounder
trained on the tree's own labels, on free hardware, is enough to make the vision side useful.

## Figures to make

1. Dev learning curve (step 0 → 1179), plus seed 7 [E1] and the data-scaling points [E7].
2. ScreenSpot-v2 by platform × text/icon, zero-shot vs fine-tuned.
3. Test A condition × method heatmap.
4. Test B: tree state × method, with the fallback-below-vision gap marked.
5. Hit rate by target-size quartile; miss-distance histogram.
6. [E3] UIA coverage per app.

## Appendix checklist (reproducibility)

Commit hashes for every frozen rule (journal) · notebooks pinned to CODE_REF · adapter on the Hub with a model
card · dataset *not* released until a privacy audit · bootstrap code `eval/report.py`.
