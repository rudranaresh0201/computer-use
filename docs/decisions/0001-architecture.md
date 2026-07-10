# ADR-0001: Hybrid Perception Architecture (UI Automation + Vision)

Date: 2026-07-10
Status: Accepted

## Context

Anthropic's own reference implementation ([[anthropic-computer-use]]) is pure-vision: it never looks at anything but screenshots, and it runs on Linux/X11 where there's no equivalent used at all. We are targeting Windows specifically, where a real, mature accessibility API — **UI Automation (UIA)** — exists and exposes the structural tree of most native applications (window, button, text field, their exact bounding boxes and names) without needing to visually parse anything.

Research into prior art ([[prior-art]]) shows this matters in practice: Microsoft's UFO/UFO2 project found that fusing UIA with vision-based parsing produces far more reliable element targeting on Windows than vision alone, and OSWorld's benchmark results show that **grounding (clicking the right thing) is the dominant failure mode** for computer-use agents generally — not high-level planning.

## Decision

Perception in this project is **hybrid, with three layers, queried in this priority order per step**:

1. **UI Automation tree** (via `pywinauto`) — when the focused/target application exposes a usable UIA tree, prefer it. It gives exact element bounds, names, and roles with no ambiguity and no vision-model cost.
2. **Vision + structured labeling** — for apps with sparse/missing UIA trees (browsers rendering canvas content, games, custom-drawn UIs), fall back to a screenshot passed through a labeling step (OmniParser-style detection, or lighter-weight Set-of-Mark numbering as used by self-operating-computer) so the Brain reasons over a list of labeled candidate elements instead of guessing raw pixel coordinates.
3. **Raw screenshot** — always captured and always sent, as ground truth and for final human-style visual verification (matches Anthropic's own recommended "screenshot + evaluate outcome" prompting pattern).

The `Brain` interface (see [[0002-brain-abstraction-and-cost-strategy]]) receives all three when available: `(goal, history, screenshot, uia_tree, labeled_elements)`. A given Brain implementation is free to ignore what it doesn't need (e.g., a pure-vision Claude computer-use call ignores the UIA tree; a cheaper local-model setup might lean on UIA/labels heavily to compensate for weaker vision grounding).

The `Action` schema mirrors Anthropic's own action vocabulary (click, type, key, scroll, drag, wait, screenshot, done) — see [[anthropic-computer-use]] — so swapping the Brain later requires no change to the execution layer.

## Consequences

- More upfront implementation work than a pure-vision clone (Phase 1 needs a UIA integration, not just mss + pyautogui).
- This is the concrete technical differentiator between this project and a weekend reskin of Anthropic's demo — it's the reason to expect meaningfully better reliability on native Windows apps.
- Adds a dependency on `pywinauto` and (later) a labeling model/service; both are free/open-source, consistent with the $0-budget constraint.
- Some apps (browser canvas content, games) will still degrade to vision-only — that's expected and acceptable, not a bug to "fix" in v1.
- We must design the `Action`/perception data structures generically enough in Phase 1 that adding the vision-labeling layer in Phase 3 doesn't require reworking the harness.
