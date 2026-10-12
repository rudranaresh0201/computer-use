# Study guide: the whole project, in the order to learn it

For reading after the last GPU run. Every number here is in `docs/journal.md` with its date and commit;
this file explains what the numbers mean and why each test exists. Work through it top to bottom; each
section ends with questions you should be able to answer out loud without notes.

---

## 0. The story in one paragraph

An agent that uses a computer has to turn "click Save" into a point on the screen. There are two ways
to find that point. **The accessibility tree** (UIA on Windows) is a list of on-screen elements with names and
boxes, provided by the app itself: exact when it's there, but many apps leave things out or unnamed.
**Vision** reads the screenshot like a person does: it works on anything visible, but it can miss.
We trained a small vision model (2B parameters, free GPU) using the tree's own labels as the teacher,
then measured — fairly — where each source wins, and how to combine them. Main finding: **the tree wins only
when the button has a clean name; vision wins everywhere else; a naive "tree first, else vision" is worse
than vision alone; and letting an LLM read the tree and say "NONE" when the target isn't there is the best
combination (83.2% vs vision 66.4% vs tree 26.4%).**

---

## 1. Concepts you need

| concept | in one line | where it lives |
|---|---|---|
| GUI grounding | instruction + screenshot → a click point | `eval/vlm_grounder.py` |
| UIA / accessibility tree | the app's own list of elements: role, name, box | `perception/uia.py`, `realtest/capture.py` |
| 0-1000 coordinates | points scaled so any screen size maps to 0-1000 on each axis | `training/prepare_dataset.py` |
| hit | the predicted point is inside the true box — the only metric we use | `is_hit_normalized` |
| Qwen2-VL-2B | the base vision-language model: a vision encoder + a language model | Hugging Face |
| LoRA | train small add-on matrices instead of all 2B weights; fits a free T4 | `training/train_lora.py` |
| UIA as the teacher | named elements in the tree give (screenshot, "click X", box) pairs for free | `scripts/run_dataset_collection.py` |
| dev / test split | choose things on dev; touch test once, at the end | `eval/ablation.py` |
| zero-shot | the same model without our training — the control | `ZERO_SHOT` arm |
| pre-registration | write the rule and the expected result in a commit *before* seeing the data | journal + commit hashes |
| bootstrap CI | resample examples 2,000 times; the middle 95% of results is the uncertainty | `eval/report.py` |
| paired difference | score both methods on the *same* items and compare item by item — far less noise | `paired_difference` |
| clustered bootstrap | resample whole screenshots, because buttons on one screen aren't independent | `clustered_paired_difference` |

**Check yourself:** Why do we score both arms on the same items? What would happen to the interval if we
resampled single buttons when 15 buttons share a screenshot? Why choose the adapter on dev, not test?

---

## 2. The tests, in the order they were run, and why each one exists

Each test closed a loophole the previous one left open.

### Training (Oct 11)
- **Question:** can a 2B model learn to click on Windows apps from UIA's free labels?
- **How:** 3,163 examples from 14 apps → LoRA on Kaggle T4, 1,179 steps; checkpoints pushed to the Hub so a
  dead session loses nothing.
- **Result:** dev loss fell to 0.376; dev accuracy 4% → 73%, still rising (more data would likely help).

### Test 1 — does training help, and does it transfer? (H1, H3)
- **Why:** a model can memorise its training screens. New screenshots of the same apps test that; four apps it
  never saw test real transfer.
- **Result:** same apps 14.2% → 64.6% (+50.4, CI [43.0, 57.8]); unseen apps 1.5% → 46.5% (+44.9, [41.1, 48.7]).
  Every one of the four unseen apps improved (37-52%).
- **Doesn't prove:** anything about the tree. UIA scored ~100% here only because UIA picked the targets and
  wrote the instructions — it was grading itself.

### Test 2 — ScreenSpot, the public benchmark
- **Why:** our numbers are on our data. A public benchmark lets us compare with other papers, and has phones
  and websites we never trained on.
- **Result:** 17.4 → 61.9 average (+46.6, [43.5, 49.6]). Biggest gain on phones (+59) — training on Windows taught
  a general skill. Context: SeeClick-9.6B 53.4, ShowUI-2B (same base, ~80x our data) 75.1.

### Test A — the fair comparison on generated pages (H4-H6)
- **Why:** to compare tree vs vision fairly, the truth must come from *neither*. We generate pages where we know
  the true box from the page itself, and create each situation on purpose, 200 times each:
  C1 named · C2 unnamed icon · C3 asked in other words · C4 drawn on a canvas (not in the tree) · C5 the same label
  in several panels.
- **Result (1,000 items):** tree 100/0/0/0/26; vision 94/40/31/99/65 (mean 65.8 vs zero-shot 31.2).
  Naive fallback (tree, else vision) **−12 vs vision alone**.
- **Lesson:** the tree is perfect only on clean names. Its failure is not silence — it confidently picks the
  wrong element, which is why "tree first" hurts.

### Test B — the same question on real Windows screens, tree damaged on purpose
- **Why:** generated pages are clean. Our model's real-screen answers were already saved and it never reads the
  tree, so we could blank names or delete the target from the real UIA tree and re-score for free (no GPU).
- **Result (unseen apps):** when the tree is blind, vision 46.5 vs tree 1.0. Naive fallback 37.0 — below vision
  alone again. Snapping vision's point into the tree: +11.5 when the target is in the tree, −21.6 when it isn't.

### Guarded hybrid — fix the failure
- **Rule:** trust the tree only when exactly one element's name equals what was asked; else vision.
  Designed on dev, frozen, tested once.
- **Result:** +31.4 over vision when the tree is intact ([27.1, 35.9]), −1.1 when it is blind. Safe, but too
  cautious: 77.8 vs the tree's 92.8 when the tree works.

### LLM over the tree — the strongest tree baseline
- **Why:** a reviewer would say "your tree loses because name matching is dumb". So we gave the tree its best
  chance: an LLM (gpt-oss-120b) reads the whole tree and picks an element or says NONE.
- **Result:** it fixes wording and ambiguity (C3 88, C5 100) but **not blindness** (C2 2, C4 0). So vision is
  truly needed, not just beating a weak opponent.

### llm_guarded — the best combination (pre-registered)
- **Rule:** the LLM's pick; if it says NONE, vision. Written down in commit 1ad9836 before vision's answers existed.
- **Result (250 items):** 92/32/92/100/100 = **83.2** vs vision 66.4 (+16.8, [12.8, 20.8]) vs LLM alone 55.6.
- **Flaw:** on unnamed icons the LLM sometimes picks *some* nameless button instead of NONE (32 vs vision 38).
  Candidate fixes are frozen in `eval/guards.py`; they get chosen on dev in the last GPU run.

### Honest error bars, and how the model misses
- Every headline interval recomputed with whole screenshots resampled: no conclusion changed.
- Half the misses on unseen apps land within 50 units of the right box (the right area, the wrong neighbour) —
  what snapping repairs. On ScreenSpot two-thirds are far (didn't understand the instruction).
- Small targets are the weakness: hit rate 15% for the smallest quarter vs 65% for the largest.

### Still to come
- **Test C:** genuinely blind buttons in real apps (VLC, GIMP, Audacity, Electron apps), truth = boxes you draw.
- **Test D:** how often UIA is blind, per app — the statistic that says how much vision matters in practice.
- **Last GPU run** (`notebooks/phase3_final_kaggle.ipynb`): dev answers for the better guard, Test C, ScreenSpot-v2.

**Check yourself:** Why was UIA's 99.8% meaningless? Why does a naive fallback lose to vision alone? What does the
LLM baseline prove that Test A alone couldn't? Why is llm_guarded's result trustworthy even though it's the best
number we have? (Pre-registered; one test.)

---

## 3. What a reviewer will attack, and our answers

| attack | answer |
|---|---|
| "Generated pages aren't real apps." | Test B (real trees) and Test C (real blind targets) ask the same question on real screens. |
| "Your tree baseline is weak." | The LLM-over-tree baseline; it still scores ~0 on unnamed and canvas targets. |
| "You tuned on test." | Every rule chosen on dev and frozen in a commit before the test number (hashes in the journal). |
| "Intervals too narrow — buttons on one screen are correlated." | Screenshot-clustered bootstrap; nothing changed. |
| "UIA labels to train a grounder isn't new." | True — OS-Atlas did it at scale. Our claims are about *when to trust which source* (docs/research/related-work.md). |
| "One training run could be lucky." | Open: a second seed costs ~8 GPU-h. Say so in the paper's limitations. |
| "4 unseen apps is few." | Open: app-level intervals are wide ([20.9, 41.6] for the guard). Test C adds more apps. |

---

## 4. Explain-back: answer these out loud before tweeting

1. What is a "hit", and why is it a fair metric for both the tree and vision?
2. How did UIA give us training data for free, and what does that bias toward?
3. What does LoRA change in the model, and why could we train on a free T4?
4. Why were the five conditions chosen? Which method should win each, and why?
5. Why does "tree first, else vision" lose to vision alone?
6. What does the LLM's NONE do in llm_guarded, and where does it fail?
7. What is pre-registration, and which of our results were pre-registered?
8. What does a paired, clustered 95% interval of [12.8, 20.8] actually mean?
9. What is already known in the field (OS-Atlas, UFO2, OSWorld), and what is ours?
10. Where is the grounder weakest (target size, distance of misses), and what would you try next?

---

## 5. Where the code is

`eval/vlm_grounder.py` (predict, parse, hit) · `eval/report.py` (intervals) · `eval/compare.py` (tree, snap,
guarded) · `eval/tree_llm.py` (LLM over tree) · `eval/guards.py` (better-guard candidates) · `eval/real_tree.py`
(Test B) · `eval/screenspot.py` (v1 + v2) · `sandbox/` (Test A pages) · `realtest/` (Tests C/D: capture, label,
classify) · `web/` (the Chrome agent: eyes, hands, Groq brain) · `scripts/rescore_compare.py`,
`scripts/clustered_intervals.py`, `scripts/after_final_run.py` (CPU analyses).
