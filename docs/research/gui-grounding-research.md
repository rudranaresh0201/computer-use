# Research: GUI Grounding Models — Prior Work and Our Gap

Date: 2026-07-11

**Why this note exists**: after comparing computer-use against harder peer student projects seen elsewhere (a bipedal-locomotion RL benchmarking project, from-scratch systems/hardware projects — see `docs/journal.md`), the gap identified wasn't subject matter, it was rigor: those projects *train and empirically benchmark something*; computer-use as originally scoped only ever calls a pretrained model. This note is the literature review for closing that gap for real, not just adding a "training phase" without grounding it in what's already known.

## What "GUI grounding" means, precisely

Given a screenshot and an instruction ("click the Save icon"), predict the region of the image that matches — i.e., turn a fuzzy natural-language target into exact coordinates. This is the exact hard problem OSWorld's benchmark (see `docs/research/prior-art.md`) identified as the dominant failure mode for computer-use agents generally: not planning, grounding.

## Prior work

**SeeClick** (arxiv.org/abs/2401.10935, 2024). Trains GUI grounding into a VLM (continual pre-training on Qwen-VL, LoRA-fine-tuned) using ~1M examples assembled from three sources: ~300K web pages crawled from Common Crawl, with elements auto-labeled from their **HTML** (visible text content + `title` attribute hover text — i.e., the DOM *is* the free label source, no manual annotation); mobile UI data reorganized from public Android datasets; general vision-language instruction data from LLaVA to preserve general capability. Trained ~10k steps (~1 epoch).

**UGround** (osu-nlp-group.github.io/UGround, ICLR'25 Oral). Scales the same idea further: 10M GUI elements with referring expressions over 1.3M screenshots, ~95% web-sourced. Confirms the "web-scale synthetic data + adapt a LLaVA-style architecture" recipe is "surprisingly effective."

**OS-Atlas**. Scales to 13M+ elements, explicitly cross-platform (web + mobile + desktop), the largest open-source GUI grounding corpus at time of writing.

**Microsoft GUI-Actor**. A more recent architectural change: coordinate-free grounding — instead of predicting raw (x, y), predict which of a set of candidate regions/attention targets is correct. Relevant to us later if raw-coordinate regression proves noisy.

**Known data-quality problem, explicitly documented in this literature**: "Grounding datasets are typically collected through automated pipelines that extract data from noisy UI accessibility trees or raw HTML, often without thorough verification, introducing significant noise — including bounding boxes corresponding to elements that appear in the DOM but are not visually rendered in the UI." This is directly relevant to us: a UIA tree can contain elements that exist structurally but aren't visible on screen (scrolled off, hidden, zero-size) — our auto-labeling pipeline must filter for actual visible, on-screen rendering, not just tree presence. `perception/uia.py` already filters for `rect.width() > 0 and rect.height() > 0` (see the code), which is a start, but real visibility (not occluded, not off-screen, not behind another window) needs more care than that.

## The gap: nobody targets native Windows desktop apps via UI Automation

Every dataset above is web-sourced (HTML/DOM) or mobile (Android accessibility). **None uses Windows UI Automation as the labeling source, and none focuses on native Win32/WinUI desktop applications** (as opposed to a browser rendering a webpage). That's not a minor detail — native desktop apps have their own visual conventions (menu bars, toolbars, native dialogs, WinUI/Win32 controls) that look meaningfully different from web pages, and a grounding model trained purely on web/mobile data is exactly the kind of out-of-distribution case where general VLM grounding is known to be weak.

Our `perception/uia.py` module, built in Phase 1, already produces exactly the (screenshot, labeled element, bounding box) triples this literature auto-labels from HTML — for native Windows apps specifically, which is the gap. This is the concrete, citable, defensible novel angle: **a UIA-sourced GUI grounding dataset and fine-tuned model for native Windows desktop applications**, following the SeeClick/UGround recipe (auto-label from structural ground truth, fine-tune a small open VLM via LoRA) but applied to a platform/data-source the existing literature doesn't cover.

## Realistic scope for a solo, $0-budget project (honest constraint-setting)

SeeClick trained on ~1M examples; UGround on 1.3M screenshots. We are not attempting that scale. A defensible, honest scope: a few thousand auto-labeled (screenshot, element, bounding box, instruction) examples across a deliberately chosen spread of common native Windows apps (File Explorer, Notepad, Settings, Office apps, a few common third-party apps), LoRA-fine-tune a *small* open VLM (candidates: Qwen2-VL-2B or similar small-parameter model — small enough to fine-tune on free-tier cloud GPU quota, e.g. Kaggle's free ~30hrs/week T4/P100 access, consistent with the $0-budget principle from ADR-0002), and run a real ablation: our fine-tuned grounder vs. zero-shot prompting a general VLM vs. UIA-only vs. the hybrid — on a held-out set of apps not seen during training, to test actual generalization, not memorization.

This is small by industry-paper standards and that's fine and worth saying plainly: the goal is a genuine, honest, correctly-methodologied contribution at solo-project scale, not a claim to beat published research at its own scale.

## A comparable peer project

Among a broader survey of peer student projects seen elsewhere, the closest structural cousin to this phase is one that fine-tunes vision-language models using reinforcement learning from scratch — covering VLM architecture, reward design, GRPO training loops, and evaluation on standard benchmarks. Same general activity as this phase (fine-tuning a VLM, real training loop, real eval), different method (RL/GRPO vs. our supervised LoRA fine-tuning) and different data source (a general architecture/benchmark exercise vs. our UIA-auto-labeled native-Windows dataset).

Notably, GRPO-based RL fine-tuning for GUI grounding specifically is real, current literature too (e.g. "GRPO for GUI Grounding Done Right," Hugging Face blog/OpenReview's SE-GUI). That means there's an honest, citable stretch goal sitting on top of this phase's base plan: get supervised LoRA fine-tuning working first end-to-end (the committed plan), and *if* time allows afterward, a GRPO-based RL fine-tuning pass on the same UIA-sourced dataset would be the more advanced version — directly comparable to that peer project's approach, applied to the novel data source instead of a generic benchmark. Not committed scope yet, worth revisiting once the base version works.

## Sources
- SeeClick: https://arxiv.org/abs/2401.10935
- UGround: https://osu-nlp-group.github.io/UGround/
- GUI-Agents paper list (grounding): https://github.com/OSU-NLP-Group/GUI-Agents-Paper-List/blob/main/paper_by_key/paper_grounding.md
- GUI-Actor (Microsoft): https://microsoft.github.io/GUI-Actor/
