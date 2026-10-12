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
