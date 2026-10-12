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
