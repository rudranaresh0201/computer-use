# ADR-0002: Pluggable Brain Abstraction + $0-Budget Development Strategy

Date: 2026-07-10
Status: Accepted

## Context

There is currently no budget for paid Claude API usage. The plan is to prove the harness works, then make a funding ask backed by real data (cost-per-task, eval results) rather than asking upfront. Meanwhile every serious prior-art project studied ([[prior-art]]: UFO2, OmniParser/OmniTool, self-operating-computer) already treats the model that does the "deciding" as swappable — none of them hard-code a single vendor. This is not a workaround unique to our budget constraint; it's the normal shape of these systems.

## Decision

Define a single `Brain` interface early (Phase 2) that all decision-making backends implement:

```
Brain.decide(goal, history, screenshot, uia_tree, labeled_elements) -> Action | Done
```

Backends, in the order we'll build/use them:
1. **Scripted/fake Brain** (Phase 2) — hardcoded or rule-based responses, used only to integration-test the orchestrator loop end-to-end for free. Never intended to "work" on real tasks.
2. **Local free VLM via Ollama** (Phase 3, e.g. a Qwen2-VL or LLaVA-class model) — unlimited free iteration for prompt/loop development. Expected to be noticeably worse at grounding than Claude; that's fine, its job is to shake out harness bugs, not to be good.
3. **Claude with the computer-use tool** (Phase 3+, `computer_20251124` per [[anthropic-computer-use]]) — the real backend, used deliberately and sparingly: small controlled test batches, cost logged in `docs/journal.md` every time, not left running in a loop unsupervised until we trust it.

All backends consume/produce the same `Action` schema from [[0001-architecture]], so switching backends is a config change, not a rewrite.

## Consequences

- Phase 2's orchestrator loop, safety layer, and logging all get built and debugged against a free backend before a single paid API call happens — bugs are cheap to find.
- We accept that early "does it actually complete tasks" evaluation is not meaningful until Phase 3's local-VLM stage at the earliest, and not representative of true capability until real Claude test batches happen. This is fine — Phase 0-2 is about the harness, not intelligence.
- Every paid API call, once we start making them, gets logged with cost in the dev journal — this is what eventually supports the funding ask (real numbers, not guesses), matching the roadmap's Phase 6 goal.
- Prompt caching ([[anthropic-computer-use]]) becomes relevant the moment real Claude testing starts, since it's the single biggest lever on cost for a screenshot-heavy loop — should be on by default in the Claude backend, not an optimization we bolt on later.
