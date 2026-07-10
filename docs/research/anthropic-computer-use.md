# Research: Anthropic's Computer Use Architecture

Date: 2026-07-10
Sources:
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool (official docs)
- https://github.com/anthropics/claude-quickstarts/tree/main/computer-use-demo (reference implementation)
- https://github.com/anthropics/claude-quickstarts/blob/main/computer-use-demo/computer_use_demo/loop.py (agent loop source)

## The core loop

This is the shape everything else in this project has to fit around:

1. Send Claude a system prompt + tools (`computer`, optionally `text_editor`, `bash`) + conversation history.
2. Claude returns a response. If it contains a `tool_use` block, `stop_reason == "tool_use"`.
3. Your application extracts the tool name + input, actually performs it against a real environment (click the mouse, take a screenshot, run a shell command, etc.), and returns a `tool_result` content block in a new `user` message — screenshots go back as base64 image blocks *inside* that tool_result.
4. Repeat from step 2. Loop ends when Claude responds with no tool_use blocks (task done), or a max-iteration safety cap is hit.

There is no magic "autonomy" beyond this — the model doesn't touch the OS directly. **Our application is 100% responsible for**: capturing screenshots, translating `tool_use` blocks into real mouse/keyboard/shell actions, and reporting results/errors back faithfully. This is exactly the "hands and eyes" harness described in our roadmap Phase 1.

## Computer tool schema (schema-less — built into the model)

Tool definition passed in the `tools` array:
```json
{
  "type": "computer_20251124",
  "name": "computer",
  "display_width_px": 1024,
  "display_height_px": 768,
  "display_number": 1,
  "enable_zoom": true
}
```
Requires beta header `anthropic-beta: computer-use-2025-11-24` (older models use `computer-use-2025-01-24`).

### Actions supported
- **Basic (all versions)**: `screenshot`, `left_click [x,y]`, `type`, `key` (e.g. "ctrl+s"), `mouse_move`
- **Enhanced (`computer_20250124`+)**: `scroll` (direction + amount), `left_click_drag`, `right_click`, `middle_click`, `double_click`, `triple_click`, `left_mouse_down`/`left_mouse_up`, `hold_key`, `wait`
- **Enhanced (`computer_20251124`, latest models)**: `zoom` — inspect a screen region `[x1,y1,x2,y2]` at full resolution when small text/UI is illegible. Requires `enable_zoom: true`.
- Modifier keys (shift/ctrl/alt/super) are passed via the `text` param on click/scroll actions, not `hold_key` (that's for holding a key with no other action).

### Coordinates & image sizing — the single trickiest implementation detail
- The API **downscales** oversized images before Claude sees them, but Claude's returned click coordinates are in the space of the image *it saw*, not your real screen. If you rely on server-side downscaling you lose the scale factor needed to map coordinates back.
- **Correct approach**: resize the screenshot yourself before sending, set `display_width_px`/`display_height_px` to the *resized* dimensions, and scale returned coordinates back to real screen space yourself.
- Size limits vary by model: newest models (Sonnet 5, Opus 4.8/4.7) accept up to 2576px on the long edge; older models cap at 1568px / ~1.15MP. Images over ~8000px on a side get hard-rejected, not downscaled.
- **Implication for us**: our screenshot module needs an explicit resize+scale-factor step from day one — this is not optional plumbing, it's the difference between clicks landing on the right button or not.

### Error handling pattern
Failures go back as a normal `tool_result` with `is_error: true` and a descriptive message (e.g. "Coordinates (1200,900) are outside display bounds (1024x768)"). Claude sees the error and can retry/adapt. We should follow this pattern exactly in our own tool layer rather than throwing/crashing — the agent's recovery ability depends on getting a legible error, not a stack trace.

## System prompt behavior
When any computer-use-schema tool is present, Claude auto-prepends a system prompt fragment ("You have access to a set of functions... including a sandboxed computing environment..."). Our own `system` prompt is appended after that, not replacing it. Anthropic's own reference demo injects the current date and OS/environment description into this suffix — we should do the same (inject OS = Windows, available apps, etc.) since it measurably helps grounding.

## Prompting best practices (directly from the docs, worth encoding into our default system prompt)
1. Keep tasks simple/well-defined with explicit steps where possible.
2. Force self-verification: prompt Claude to screenshot + explicitly evaluate outcome after each step before moving on ("I have evaluated step X..."). Without this it tends to assume actions succeeded.
3. Dropdowns/scrollbars are hard to manipulate via mouse — nudge toward keyboard shortcuts for these.
4. For repeated/known tasks, few-shot with example screenshots + tool calls in the prompt.
5. Put instruction text **before** the screenshot image in the content array — ordering measurably improves click accuracy.
6. Extended thinking `effort`: `medium` is the best accuracy/cost tradeoff for most models on UI tasks; `max` wastes tokens without helping. Relevant later for cost control (Phase 3).

## Agent loop reference implementation details (from `loop.py`)
- Loop runs `while True`, no hardcoded turn cap in the reference (their web UI provides the practical stop). **We will cap iterations explicitly** (per our ADR — safety first).
- **Prompt caching**: cache breakpoints set on the 3 most recent turns + one for the static system/tools block. When caching is on, image pruning (`only_n_most_recent_images`) is disabled because it would break the cache — caching wins even over image-count savings.
- **Image pruning**: without caching, only the N most recent screenshots are kept in context (older ones stripped) to bound context growth over long runs — pruning happens in threshold-sized chunks, not one at a time.
- Supports multiple backends (direct Anthropic API, Vertex, Bedrock) — not relevant to us yet, but confirms the loop itself is provider-agnostic in shape, which validates our own "pluggable Brain" ADR.

## Security guidance (directly informs our Phase 1 safety layer + later ADRs)
- Run in a dedicated VM/container with minimal privileges — never the daily-driver machine, given the agent can take real destructive actions.
- Don't give it access to real credentials/sensitive data.
- Restrict internet access to an allowlist where feasible.
- **Require human confirmation for consequential actions** (purchases, ToS agreement, deletes) — this maps directly to our planned "risky action confirmation" feature in Phase 4.
- Prompt injection from on-screen content (malicious text in a webpage/image) is a real, acknowledged risk class; Anthropic runs a classifier that can force a confirmation step when it detects this in a screenshot. We have no such classifier — worth flagging as a real gap in our own security posture until we build something equivalent.

## Related quickstart worth reading next (not yet fetched)
`anthropics/claude-quickstarts` also has a **"Computer Use Best Practices"** quickstart that runs natively on macOS (no Docker) and demonstrates: explicit tool definitions, image sizing/pruning, prompt caching, server-side compaction, batched tool calls, a sandboxed shell, and **trajectory recording**. Trajectory recording in particular is directly relevant to our Phase 4 "structured run logs + replay tool" — worth a follow-up research note once we're closer to that phase.

## Takeaways for our architecture
- Our `Action` schema (Phase 1) should mirror Anthropic's action vocabulary closely (click/type/key/scroll/drag/wait/screenshot) even while our own Brain is a stub — it's the vocabulary we'll need the moment we plug in Claude.
- Screenshot resize + coordinate scale-back is a first-class module, not an afterthought.
- Error results must be structured, descriptive, and non-fatal to the loop — this is how the agent "self-heals."
- The self-verification prompting pattern (screenshot + explicit evaluation before proceeding) should be baked into our default system prompt once we have a real Brain.
- Max-iteration cap and human-confirmation-for-risky-actions are both things Anthropic explicitly flags as necessary and that we already planned independently in the roadmap — good validation of Phase 1/4 scope.
