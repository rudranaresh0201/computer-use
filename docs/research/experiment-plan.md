# Experiment plan: the numbers behind the paper, the blog and the demo

Written 2026-10-12, after the v3 ablation (H1, H3 supported) and ScreenSpot
(17.4 → 61.9). One agent and one task suite feed both outputs: the product
demo is an episode of the same harness the experiments run hundreds of times.

The model of a good study here is a driving-simulator paper: a controlled
environment, scenarios with automatic pass/fail, many runs, and a breakdown
of *why* runs fail -- not one impressive video.

## Done

| study | result |
|---|---|
| S1a v3 ablation (own splits) | unseen apps 1.5% → 46.5% (+44.9, CI [41.4, 48.5]); new screenshots 14.2% → 64.6% |
| S1b ScreenSpot (public) | 6-cell avg 17.4 → 61.9 (+46.6 per-example, CI [43.6, 49.7]); mobile +59 without phone training |

## Next

**S2 -- Tree vs vision vs both, at the task level.** The research question
end to end: same tasks, same brain, perception varied:
(a) accessibility tree only; (b) screenshot + our grounder only; (c) tree,
falling back to the grounder for elements the tree lacks. Tasks run in a
**local sandbox** of small web apps (shop, form, settings, calendar) where a
server-side checker decides pass/fail -- reproducible, safe, free -- and
some targets are drawn into `<canvas>` so the tree is genuinely blind to them
(it is: tests/test_web_browser.py). Metrics: success rate with a 95%
interval, steps, tokens, seconds per task.

**S3 -- How often does the agent say "done" when it isn't?** Every episode
ends with the agent's own claim and the checker's verdict; the gap is the
false-success rate. This is the bridge to verimcp: a verifier that checks
an agent's claim against the world.

**S4 -- How much data does grounding need?** Retrain on 100 / 250 / 500 /
1000 / 1572 examples, and on 2 / 5 / 10 apps; score held-out apps and
ScreenSpot. Answers "quantity or diversity?". ~10-12 Kaggle GPU hours,
spread over weeks inside the free quota.

**S5 -- How blind is UIA in the wild?** On ~20 Windows apps: elements UIA
exposes, the share with no usable name, apps where the tree is empty. No
GPU; runs on the laptop.

## Compute

- Demos: Groq free tier (~8K tokens/minute, ~200K tokens/day per model at
  the time of writing) -- enough for a few dozen episodes a day.
- Experiments: an open model on a Kaggle GPU as the brain -- no rate
  limits, and fixed weights, so a run can be repeated exactly. Episodes
  resume from the Hub like the evals do.

## Rules carried over from H1-H3

Conditions and metrics are fixed in this file before a run; the task suite
is frozen before the comparison runs; every rate gets an interval; failures
are categorized, not only counted.

## Status and the plan to make it excellent (2026-10-12, evening)

**Done:** H1/H3 (v3 ablation), ScreenSpot (17.4 → 61.9), Test B (real screens, tree damaged:
vision 46.5 vs tree 1.0 when blind; naive hybrids fall below vision), guarded hybrid (dev-chosen,
test: +31.4 / −1.1), LLM-over-tree baseline (fixes wording/ambiguity, still ~0 on unnamed icons and
canvas). **Running:** Test A on Kaggle. **Built, waiting on a person:** Tests C/D tooling.

What "excellent" needs, in order:

1. **Finish the comparison.** Test A results; re-score A's 250-item subset with `guarded` and the
   pre-registered `llm_guarded`.
2. **Real blind spots (C) and how common they are (D).** One capture session (~20 real apps:
   Electron, Qt, GTK, wxWidgets, canvas-heavy, plus native ones for contrast), one labelling pass
   (~250 boxes). C needs ~0.5 GPU-h. D's coverage table is the headline statistic.
3. **A better guard**, designed on dev only: use agreement between the tree's pick and the
   grounder's point (or the LLM's NONE) to decide whom to trust. Target: close the 77.8 vs 92.8
   gap when the tree works, keep ~0 cost when it is blind. Confirmed once on B, A and C.
4. **Honest uncertainty.** Items cluster by screenshot and app, so item-level bootstrap intervals
   are too narrow; recompute the headline intervals with a screenshot-clustered bootstrap (CPU).
5. **Robustness, if GPU allows:** ScreenSpot-v2 (corrected labels, ~2 GPU-h); a second training
   seed (~8 GPU-h) to show the result isn't one lucky run.
6. **Related work** before any "first" claim (OS-Atlas, UGround, ShowUI, SeeClick, UFO/UFO2,
   OmniParser, WinSpot, GroundCUA, accessibility-tree agents).
7. **Writing:** blog first (after A), then a 4-page workshop paper. Figures: learning curve,
   ScreenSpot by platform, condition × method table, tree-damage chart, coverage per app.

Product, in parallel:

1. First live `web-run` with the free brain, recorded.
2. A Windows task end to end (UIA eyes + the existing hands + the brain).
3. The guarded hybrid inside the live agent (tree when it's sure, vision otherwise) --
   needs the grounder served from a GPU; CPU is too slow for an interactive demo.
4. A task suite on the sandbox pages for success rate and the "says done but isn't" rate.

## Gaps and the strengthening backlog (2026-10-18, after the seed2 training run)

Written the day the second-seed run finished training (seed 7, all 1,572 examples, 1,179 steps, 7.0 h on a
T4, peak 8.5 GB, eval_loss 0.3586 at step 1100). Each gap is the question a reviewer would ask; each
experiment is the cheapest thing that answers it. Tick boxes as they land; record results in journal.md.

**Where we stand.** Element-level grounding is well supported: H1 64.6 / H3 46.5 (unseen apps, every one of
four improves), ScreenSpot-v2 62.8, Test A (synthetic, 5 conditions), Test B (real screens, tree damaged),
llm_named guard 85.2 vs vision 66.4. What is *not* measured yet: real UIA-blind targets, how common they are,
seed variance, an outside baseline on our splits, data scaling, and anything at the **task** level.

Note on the run log: `torch/utils/checkpoint.py: None of the inputs have requires_grad=True` is expected. It
comes from `keep_vision_tower_out_of_backward` (training/train_lora.py), which runs the frozen vision tower
under no_grad on purpose. LoRA sits only on the language model, so it still gets gradients through the text side.

### Tier 1: closes the claims we already make (cheap, do first)

- [ ] **E1 Seed variance.** `phase3_more_eval_kaggle.ipynb`, RUN = "seed2" (~3.7 GPU-h, full protocol).
  Report H1/H3 for both seeds and the spread; check whether dev picks the same step region. *Gap:* "one lucky run?"
- [ ] **E2 Leakage audit (CPU).** Perceptual-hash nearest neighbour from every test screenshot to train; same-app
  split is by screenshot, so near-duplicates are likely. Re-score H1 without test items whose screenshot has a
  near-twin in train. Also count shared common controls (File menu, title-bar buttons) across "unseen" apps.
  *Gap:* "is 64.6 memorization?"
- [ ] **E3 UIA coverage survey = Test D (CPU, S5).** ~20 apps (Electron, Qt, GTK, wx, canvas-heavy + native):
  elements exposed, share unnamed, empty trees. Headline stat of the paper; it also tells us which side of the
  ~25% blind-share break-even real use sits on. *Gap:* "how often does your hybrid matter?"
- [ ] **E4 Test C, real blind targets.** Label ~250 boxes on the 8 captured apps in runs/realtest (GIMP has only
  7 UIA elements), then ~0.5 GPU-h. *Gap:* the title claim ("ground where UIA is blind") is only shown on synthetic
  canvas (A/C4) and on trees we damaged ourselves (B).
- [ ] **E5 llm_named on real screens (CPU + Groq).** The better guard was confirmed only on 250 synthetic items. Run
  it on Test B's real held-out screens before quoting 85.2 anywhere. *Gap:* "the guard is tuned to the sandbox."

### Tier 2: makes the paper competitive (GPU, schedule against the weekly Kaggle quota)

- [ ] **E6 Outside baseline on our splits.** ShowUI-2B (same Qwen2-VL-2B base, 256K examples) on v3 same-app +
  held-out, and on Test C. ~1-1.5 GPU-h. Replaces the "approximate, cross-paper" ScreenSpot context with a
  direct comparison. *Gap:* "is a Windows-UIA-labelled 1.5K set any better than a public 256K one?"
- [ ] **E7 Data scaling (S4).** RUN = n100 / n200 / n500 / n1000, train + eval (~0.6+1.2+2.8+5.2 h train,
  ~1.5 h eval each; ~16 GPU-h total). Plot held-out accuracy vs examples, both seeds' full-data point on it.
- [ ] **E8 App-diversity scaling.** Same example count, 2 / 5 / 10 training apps (needs a notebook setting, not
  built). With E7, this answers "quantity or diversity?"
- [ ] **E9 Paraphrase robustness.** Vision scores 28-31 on reworded instructions (Test A C3), and two-thirds of
  ScreenSpot misses are "far" (didn't understand the ask). Eval first: LLM-rewrite held-out instructions into
  functional wording ("the button that saves the file"), re-score (~1 GPU-h). If the drop is large, retrain
  with paraphrase-augmented instructions (~8 GPU-h) and re-run A/C3 + ScreenSpot.
- [ ] **E10 Zoom-in second pass for small targets.** Smallest-quartile targets hit 15% vs 65%. Inference only: crop
  around the first guess, ground again on the crop, map back. ~1 GPU-h on held-out + ScreenSpot icons.
- [ ] **E11 Does the grounder know when it's wrong?** Sample k=5 answers; spread of the points as confidence;
  AUROC for hit vs miss. If it separates, it becomes a guard signal (trust the tree when the eyes are unsure)
  and lets the agent ask instead of misclicking.

### Tier 2b: the teacher-blindness gap (found 2026-10-18)

UIA can only label elements it names, so every training example is a *named* element, but the grounder's job
in the product is the *nameless* ones. Test A C2 (unnamed icons) is vision's weak cell (38-40).

- [ ] **E15 Measure it.** On Test C labels, split the hand-drawn targets into "has a UIA name" / "no UIA name /
  not in the tree" and score the grounder on each. A large gap means the training distribution is the bottleneck,
  not model size. Free once E4's labels exist.
- [ ] **E16 Fix it, cheaply.** Make nameless-looking training examples from named ones: mask the visible label
  text, or describe the element by function/position ("the icon left of Save", "the magnifier icon") via an LLM
  rewrite, then retrain (~8 GPU-h) and rescore C2 + Test C. If E9 (paraphrase) runs, combine the two in one retrain.

### Tier 3: agent level, where the research meets the product

- [ ] **E12 CPU eyes: latency vs accuracy (CPU).** `scripts/bench_grounder_cpu.py --n 12 --edges 1568,1024,768`,
  int8 on and off. Decides whether the live demo can use the eyes on the laptop.
- [ ] **E13 Hero-task bench + false-success rate (S2/S3).** `tasks.py` checkers (Paint PNG, Sheets CSV, Docs,
  Tinkercad STL) are independent of the agent's own receipts. 10 runs per task: pass@10 *and* pass-all-10, steps,
  tokens, seconds; and the gap between "agent said done" and "checker agrees" (the verimcp bridge). Spends model
  tokens, so last, once every piece passes offline (feedback: token economy).
- [ ] **E14 Perception ablation at the task level (S2).** The same hero tasks with tree only / eyes only / guarded
  hybrid. Turns the element-level story into "tasks finished".

### Budget

Tier 1 ~4.2 GPU-h + CPU evenings + one labelling session. Tier 2 ~30 GPU-h (E9 retrain optional). Kaggle gives
~30 GPU-h/week, so Tier 1 + E6 fit the coming week; E7 the week after.
