# ADR-0003: Train and Benchmark a Native-Windows GUI Grounding Model

Date: 2026-07-11
Status: Accepted

## Context

Comparing computer-use against harder Eklavya/SRA projects (Waddle: trains and empirically benchmarks PPO vs SAC locomotion policies; AstraRTOS/TinyGPU: build real systems from scratch) surfaced an honest gap: as originally scoped, computer-use only ever *calls* a pretrained model (Claude, or a local VLM) — it never trains or rigorously evaluates anything of its own. That's a real difference in rigor, independent of subject matter, and it's fixable without abandoning the project or pivoting into an unrelated discipline (RL, embedded systems) where there's no existing skill or coherence with the rest of the portfolio.

Literature review (`docs/research/gui-grounding-research.md`) found that "auto-label GUI grounding data from structural ground truth, fine-tune a VLM" is an established, active research direction (SeeClick, UGround, OS-Atlas, Microsoft's GUI-Actor) — and that every existing dataset in that space is web (HTML/DOM) or mobile (Android) sourced. None targets native Windows desktop applications via UI Automation. Our Phase 1 `perception/uia.py` module already produces exactly the (screenshot, element, bounding box) triples this literature auto-labels from HTML — for the platform the literature doesn't cover.

## Decision

Add a genuine trained-and-benchmarked contribution to the project: a **GUI grounding model for native Windows desktop applications, fine-tuned on a dataset auto-labeled from UI Automation** (no manual annotation), following the SeeClick/UGround recipe (LoRA-fine-tune a small open VLM) adapted to this new data source, evaluated with a real ablation study (our fine-tuned grounder vs. zero-shot VLM prompting vs. UIA-only vs. the hybrid) on apps held out from training.

This becomes a real phase in the roadmap (see `docs/roadmap.md`), not a side experiment: literature review (done), extend the UIA module into a labeled-dataset collection tool, curate a dataset across a deliberate spread of common Windows apps, fine-tune, evaluate, write up results.

Scope is deliberately small relative to the cited papers (thousands of examples, not millions; a small model, not a frontier one) — an honest, correctly-methodologied contribution at solo-project/$0-budget scale (consistent with ADR-0002), not a claim to out-scale published research. Training compute: free-tier cloud GPU quota (e.g. Kaggle's free weekly T4/P100 access), keeping the $0-budget principle intact.

## Consequences

- The roadmap gets meaningfully longer and the "real brain" phase gets split: a general-purpose Brain (Claude / local VLM, per ADR-0002) still exists for actual task execution, but our own trained grounding model becomes a component it can call for improved click accuracy on native Windows apps specifically — this doesn't replace ADR-0002's pluggable Brain design, it adds a specialized perception component underneath it.
- Real ML training/evaluation infrastructure is now required: dataset versioning, a training script, an evaluation harness with a held-out app split, and honest reporting of results (including if the fine-tuned model *doesn't* clearly beat the baselines — that's still a valid, honest result worth reporting, not something to hide).
- This is the single largest addition to project scope since Day 0 and will take real time — accepted deliberately, given the explicit goal of a months-long project with genuine research rigor rather than "call some APIs and finish in a week."
