# Phase 3 Hypothesis Document: GUI Grounding on Native Windows Apps

Date: 2026-07-12
Status: Draft — precedes dataset collection and training (Phase 3, ADR-0003)

This document states, in advance of collecting data or training anything, what we
believe is true and what would prove it false. The dataset, baseline set, training
procedure, and evaluation protocol for Phase 3 are all derived from the claims below,
not decided independently of them.

## Research question

Does fine-tuning a small open VLM on a UIA-auto-labeled dataset of native Windows
desktop screenshots improve GUI-element grounding accuracy on native Windows apps,
compared to the same base VLM used zero-shot — and if so, where specifically does
that improvement come from (memorization of trained apps vs. genuine transfer to
unseen apps; visual grounding vs. structural UIA lookup)?

## Primary hypothesis (H1)

A small VLM (Qwen2-VL-2B class), LoRA-fine-tuned on our UIA-auto-labeled dataset,
will achieve higher grounding accuracy on native Windows app screenshots than the
same base model prompted zero-shot for the same task.

**Rationale**: per the lit review (`docs/research/gui-grounding-research.md`), every
public GUI grounding dataset (SeeClick, UGround, OS-Atlas) is web- or mobile-sourced.
Native Win32/WinUI visual conventions (menu bars, native dialogs, toolbar iconography)
are plausibly out-of-distribution for a base VLM's grounding pretraining, so there
should be headroom to close with in-distribution fine-tuning data.

## Secondary hypotheses

**H2 — UIA vs. vision is app-dependent, not universally one-or-the-other.**
On apps with rich, accurate UIA trees (File Explorer, Notepad, Settings), a plain
UIA structural lookup will match or beat the fine-tuned grounder, because UIA gives
exact ground-truth coordinates when the tree is accurate. On apps with weak or
custom-rendered UI (canvas-drawn controls, non-standard toolkits), the fine-tuned
grounder will beat UIA-only, because there's little or no usable tree to query.
**A hybrid strategy (UIA when available, fall back to the fine-tuned grounder
otherwise) will beat both pure strategies on average across the full app spread.**
This is the empirical justification for the hybrid perception architecture already
committed to in ADR-0001 and ADR-0003 — currently that design is argued from
first principles, not measured.

**H3 — generalization, not memorization.**
Grounding accuracy on apps *excluded from training* (held-out apps) will be lower
than accuracy on training-distribution apps (held-out *screenshots* of trained apps),
but still measurably higher than the zero-shot baseline on those same held-out apps.
This is the test for whether fine-tuning taught transferable native-Windows visual
conventions, versus just memorizing the specific layouts of the apps it saw.

## Variables

**Independent variable**: which grounding strategy produces the coordinate
prediction — this is the ablation arm (see below).

**Dependent variable (primary metric)**: click accuracy — the fraction of examples
where the predicted point falls inside the ground-truth element's bounding box.
This is the standard metric used in SeeClick/UGround, chosen for direct
comparability to the literature.

**Dependent variable (secondary metric)**: center-distance in pixels between
predicted point and bounding-box center, reported as a continuous signal for cases
that narrowly miss (useful for error analysis even where the binary metric says
"fail").

## Predicted outcome, ranked

If H1–H3 hold, the ordinal ranking of mean click accuracy across arms should be:

1. Hybrid (best overall, across the full app spread)
2. UIA-only, *on rich-tree apps specifically* — highest of all arms on that subset
3. Fine-tuned grounder — second-best on rich-tree apps, best of the non-hybrid
   arms on weak-tree apps and on held-out apps
4. Zero-shot VLM — worst on native Windows apps generally
5. UIA-only, *on weak-tree apps specifically* — near-zero, since there's little to
   query

No absolute accuracy numbers are fixed in advance — at solo-project/$0-budget scale
(per ADR-0003) we don't have a basis to predict a specific percentage. The claim
being staked here is the **ordering and the gap direction**, which is falsifiable
regardless of absolute scale.

## Falsification conditions

This hypothesis set is considered **not supported** if any of the following hold on
the held-out evaluation set:

- The fine-tuned grounder performs at or below the zero-shot baseline on held-out
  apps (H1 and H3 both fail — fine-tuning added nothing, or worse, overfit to the
  training apps without transferring).
- The hybrid strategy does not beat both pure UIA-only and pure fine-tuned-grounder
  averaged across the full app spread (H2 fails — the hybrid design isn't earning
  its complexity).
- Held-out-app accuracy is statistically indistinguishable from held-out-screenshot
  (same-app) accuracy *and* both are low — would suggest the model isn't learning
  useful grounding at all, just noise.

Per the roadmap's rigor bar, any of these outcomes gets reported as-is, not
suppressed or reframed after the fact.

## How this determines the dataset

- Must include a **deliberate spread of UIA-tree richness**: some apps with strong,
  accurate structural trees (File Explorer, Notepad, Settings) and some with
  weak/custom-rendered UI — without this spread, H2 can't be tested at all, only
  asserted.
- Must have a **hard app-level held-out split**, decided before training starts:
  a set of apps entirely excluded from the training data, reserved only for the
  final H3 evaluation. Splitting by *screenshot* instead of by *app* would leak
  layout familiarity into the "held-out" set and silently invalidate H3.
- Filtering must exclude UIA elements that exist in the tree but aren't actually
  visible on-screen (the documented noisy-label failure mode from the lit review) —
  a labeling problem here would corrupt every downstream metric, not just accuracy.

## How this determines the baseline set (ablation arms)

Four arms, each existing to isolate one hypothesis, not for generic completeness:

| Arm | Isolates |
|---|---|
| UIA-only | Ceiling for structural access; needed to test H2's "wins on rich-tree apps" claim |
| Zero-shot VLM (no fine-tuning) | The free baseline everyone already gets; needed for H1 |
| Fine-tuned grounder | The trained contribution itself |
| Hybrid (UIA when available, else fine-tuned grounder) | H2's actual claim — this arm is the point, the other three exist partly to justify it |

## How this determines the training procedure

- Requires a **train / dev / held-out-test** split, not just train/test — the dev
  split absorbs any hyperparameter or early-stopping decisions so the held-out-test
  split is genuinely untouched until the one final evaluation run. Touching the
  held-out set more than once turns it into a dev set and invalidates H3.
- LoRA configuration and stopping criterion get fixed *before* looking at held-out
  numbers, for the same reason.

## How this determines the evaluation protocol

Accuracy must be reported **sliced three ways**, not as one aggregate number,
because each hypothesis needs its own slice:

1. By ablation arm (H1, H2)
2. By app UIA-richness (rich-tree vs. weak-tree subset) (H2)
3. By held-out-screenshot vs. held-out-app (H3)

An aggregate-only number would average away exactly the distinctions this
hypothesis set is designed to detect.

## Threats to validity

- **Dataset size at solo scale**: a few thousand examples (per ADR-0003) may be too
  small to reach statistical confidence on the finer slices (e.g. weak-tree held-out
  apps specifically may have very few examples). Flagged now so it can be reported
  as a limitation rather than discovered as a surprise.
- **App-selection bias**: "rich-tree" vs. "weak-tree" is a judgment call made when
  selecting apps, not a measured property — worth double-checking empirically
  (actual UIA tree completeness per app) once the dataset tool exists, rather than
  assumed from memory of the app.
- **Base model choice**: results are specific to the chosen small VLM (Qwen2-VL-2B
  class); a different base model could shift absolute numbers even if the ordinal
  claims hold.

## Success criteria

This phase counts as a genuine research contribution — win or lose on H1–H3 — as
long as: the dataset was built with the held-out app split intact, all four arms
were run on identical held-out data, all three slices were reported, and the result
(whichever it is) matches one of the outcomes anticipated above rather than
requiring post-hoc reinterpretation to explain.
