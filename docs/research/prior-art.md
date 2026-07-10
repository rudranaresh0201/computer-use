# Research: Prior Art in Computer-Use Agents

Date: 2026-07-10

Four projects studied to inform our architecture (see [[0001-architecture]]). Read alongside [[anthropic-computer-use]].

## Microsoft UFO / UFO2 ("Desktop AgentOS")
Sources: https://arxiv.org/abs/2504.14603, https://github.com/microsoft/UFO, https://aclanthology.org/2025.naacl-long.26.pdf

- **Architecture**: a centralized `HostAgent` decomposes a task and coordinates specialized `AppAgent`s, one per application, each with app-specific knowledge and a unified "GUI-API action layer."
- **Perception is hybrid, not pure vision**: a control-detection pipeline fuses **Windows UI Automation (UIA)** with vision-based parsing to handle apps that don't expose good UIA trees. This is the single most important finding for us — Anthropic's own reference implementation is pure-vision (Linux/X11, no OS-level accessibility API used), but UFO shows that on Windows specifically, mixing in UIA gives far more reliable element targeting than coordinates-from-pixels alone.
- **Deep OS integration**: also uses raw Win32 APIs for window/process management and WinCOM for Office apps specifically.
- **Takeaway for us**: this validates the plan (already in our roadmap) to build a Windows UI Automation layer (via `pywinauto`) alongside screenshot-based vision, rather than going pure-vision like the Anthropic Linux demo. It's more implementation work, but it's the legitimate reason our project isn't just a reskin of the reference demo.

## Microsoft OmniParser
Sources: https://github.com/microsoft/OmniParser, https://microsoft.github.io/OmniParser/

- **What it is**: a screen-parsing tool that turns a raw screenshot into a structured list of labeled, interactable UI elements (icons + text), using a detection model (finds clickable regions) plus a captioning model (describes what each region does).
- **Why it exists**: raw vision-language models are often bad at *precisely localizing* small UI elements (icons, tab titles, small buttons) even when they understand the screen semantically. OmniParser turns "where exactly is the save icon" into a solved pre-processing step, so the downstream model just picks from a labeled list instead of guessing pixel coordinates freehand.
- **OmniTool**: a companion project that plugs OmniParser into a Windows 11 VM alongside a pluggable choice of vision model backend — GPT-4o, Gemini, Qwen-VL, or Anthropic's own computer-use tool. This is a direct existence proof of the "pluggable Brain behind one perception layer" pattern we're planning in ADR-0002.
- **Takeaway for us**: OmniParser (or an open, self-hosted equivalent) is a strong candidate for our **free/local perception layer during $0-budget development** — it runs locally, doesn't require paid API calls, and can make even a weak local VLM click accurately by giving it a structured element list instead of raw pixels. Worth prototyping in Phase 3 alongside our UIA layer; the two are complementary (UIA gives ground-truth structure when available, OmniParser-style vision parsing covers apps/canvases where UIA trees are sparse, e.g. games, custom-rendered UI, browsers with canvas content).

## self-operating-computer (OthersideAI)
Sources: https://github.com/OthersideAI/self-operating-computer

- **What it is**: one of the earliest (Nov 2023) open-source "give a multimodal model control of your screen" frameworks. Model views a screenshot, decides mouse/keyboard actions, repeats — architecturally the simplest possible version of the loop we're building.
- **Model-agnostic by design**: already supports GPT-4o, Gemini, Claude, Qwen-VL, LLaVA — another real-world validation of a pluggable-Brain design.
- **Notable feature**: supports **Set-of-Mark (SoM) prompting** — overlaying numbered markers on detected UI elements in the screenshot so the model can respond with "click marker 7" instead of raw coordinates. This is a lighter-weight alternative/complement to full OmniParser-style parsing and worth keeping in our toolbox for the Phase 3 brain-integration work, especially with weaker/free local models that are bad at raw coordinate grounding.
- Also has an OCR-assisted mode (`gpt-4-with-ocr`) for text-heavy targeting.
- **Takeaway for us**: it's evidence that even a fairly minimal loop (no UIA, no fine-tuned parser) can work for basic tasks — useful as a sanity-check baseline for our own Phase 2 mocked-brain loop before we invest in the fancier hybrid perception from UFO/OmniParser.

## OSWorld (benchmark)
Sources: https://arxiv.org/abs/2404.07972, https://os-world.github.io/

- **What it is**: the standard academic benchmark for computer-use agents — 369 real tasks across real Ubuntu/Windows/macOS environments and real apps (not toy sandboxes), with **execution-based scoring**: each task ships a custom script that checks final system state (files written, settings changed, etc.), not just "did the trajectory look plausible."
- **Headline number worth remembering**: humans complete ~72% of tasks; even the best models at publication managed only ~12%, failing mostly on GUI grounding (clicking the right thing) and operational/domain knowledge (knowing *how* to do the task in that app) — not on reasoning about the goal itself. This tells us where the hard part of this project actually is: perception/grounding, not planning.
- **Takeaway for us**: our own Phase 6 eval suite should copy this methodology — small number of realistic, scripted, outcome-checked tasks rather than vibes-based "looks like it worked." Even 10-15 well-chosen tasks scored this way would make our project's claims credible instead of anecdotal, which matters a lot for the "better than a typical showcase project" bar we're aiming for.

## Summary: how this shapes our design
1. **Perception should be hybrid**: Windows UI Automation (structured, precise, free) + vision/screenshot (general fallback) + optionally an OmniParser/SoM-style labeling step to help weaker/free models ground clicks accurately. → ADR-0001.
2. **The Brain must be swappable**: every serious prior-art project already treats the model backend as pluggable. Confirms our $0-budget dev strategy (local model now, Claude later, same interface) isn't a compromise — it's the norm. → ADR-0002.
3. **Grounding, not planning, is the hard problem** (per OSWorld). This should bias our early engineering effort: get click accuracy and UIA integration right before spending time on elaborate planning/reasoning prompts.
4. **Evaluation needs to be execution-based, not vibes-based**, from the point we start claiming the thing "works."
