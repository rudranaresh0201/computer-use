# Where we are: the whole project on one page (2026-10-18)

The map to read before anything else. It covers what the product is, what the research found, what is
missing from each, and what to run next. Details live elsewhere and are linked, not repeated:
numbers in `journal.md`, the research backlog (E1-E16) in `research/experiment-plan.md`, concepts in
`study-guide.md`, the field in `research/product-research.md` and `research/related-work.md`. Drafts:
`paper/draft.md`, `blog/draft.md`.

---

## 1. The project in three sentences

An agent that uses a computer has to turn "click Save" into a point on the screen. We trained a small
vision model to find that point, using the accessibility tree's own labels as the free teacher, and measured
when to trust the tree, when to trust vision, and how to combine them. The product is that answer in
use: an MCP server that gives any agent (Claude Code first) eyes and hands on Windows apps and websites,
routes every action through the channel the research says to trust, and only ticks a step off its plan when
there is evidence.

## 2. The product, as it is built

```
 you type a goal ──► app/ (Electron, 3D room, robot, timeline)
                        │  JSON-RPC over stdio (same framing as MCP stdio)
                        ▼
                     sidecar.py ── spawns ──► claude -p  (the reasoning; any MCP host works)
                                                 │ MCP tools
                                                 ▼
                                   mcp_server.py: targets · open · plan · observe · locate · act · check
                                                 │
                                   session.py: one plan, one set of receipts, the approval gate
                                     ├── DesktopSurface (UIA tree + mss screenshots + pyautogui)
                                     │     apps: calculator, notepad_pp, paint
                                     └── WebSurface (agent Chrome over CDP, Playwright input)
                                           sites: sheets, docs, gmail, tinkercad, excalidraw
```

**How a target is found (the research, enforced in code, not left to the prompt):**

| order | channel | what | from which result |
|---|---|---|---|
| 1 | api | keyboard shortcuts / typing into the focused field | no pixels, nothing to ground |
| 2 | tree | a tree element, **only if it has a name** | `llm_named`: nameless tree picks lose to vision (Tests A, B) |
| 3 | eyes | `locate("the Brush tool")`: our grounder returns x, y + a zoom with a ring | H3: right about half the time on unseen apps, so the model confirms the zoom first |
| 4 | vision | the planner reads x, y off the screenshot itself | the fallback for canvases and drawing |

**What makes it trustworthy:** a plan step is ticked only by `check` with evidence (a saved file, text in
the window, a screen that changed since the step started). Irreversible actions (send, delete, pay,
passwords) wait for the person in the app, and coordinates can't route around that, because the gate looks
at what is *under* the click. On the desktop, the target app must be in front or nothing happens.

**Built and pushed:** sidecar, planner + checklist, 3D app v2, MCP server, Session + both surfaces, agent
Chrome, approval gate, `--brain claude`. 363 tests pass. One live run (Paint) worked and exposed three bugs,
all fixed.

**Built, not committed yet (tests pass):** `eyes.py` (grounder on CPU, int8, shrink-then-map-back),
`locate` tool + zoom, `tasks.py` (hero-task checkers that read the result itself, never the agent's receipts),
`scripts/bench_grounder_cpu.py`.

## 3. The research, as it stands

| question | answer | where |
|---|---|---|
| Does UIA-labelled training help? | same apps 14.2 → 64.6; **unseen apps 1.5 → 46.5** (+44.9 [41.1, 48.7]) | H1, H3 |
| Does it transfer off Windows? | ScreenSpot-v2 19.2 → **62.8**; phones +57.5 without phone data | ScreenSpot |
| Tree vs vision, per condition (generated pages) | tree wins only on clean names; vision wins on unnamed icons, rewording, canvas | Test A |
| Real screens, tree damaged | naive "tree first, else vision" is **worse than vision** (−9.5 to −12) | Test B |
| Best way to combine | `llm_named`: LLM reads the tree, its pick counts only if named, else vision: **85.2 vs 66.4** | guards |
| Is it one lucky run? | seed2 trained 2026-10-18; **eval pending** | E1 |

## 4. Gaps

### Research gaps (full list and costs: `research/experiment-plan.md`, E1-E16)

1. **The title claim has no real test yet.** "Ground where the tree is blind" is shown on synthetic canvas
   and on trees we broke ourselves. Test C (real blind targets, ~250 labels) is the fix.
2. **One seed.** Being fixed (E1).
3. **Possible memorisation in the same-app split**: test screenshots may have near-twins in train (E2).
4. **No prevalence number.** We don't know how often the tree is blind in real apps (Test D, E3).
5. **No direct outside baseline.** ShowUI-2B is only compared across papers (E6).
6. **The teacher can't label what it can't see.** UIA labels only *named* elements, so the grounder
   never trains on the nameless icons it is meant for (vision 38-40 on unnamed icons). This is the deepest
   gap; E15 measures it and E16 tries to fix it.
7. **Everything is element-level.** No task success rate, no "said done but wasn't" rate yet (E13, E14).

### Product gaps

| # | gap | why it matters | fix |
|---|---|---|---|
| G1 | README still says "Phase 3 in progress, no accuracy claim" | it's the first thing a recruiter or reviewer sees | rewrite (not done yet) |
| G2 | The "api" channel is only keystrokes: no Excel COM, no Blender Python, no Sheets API | product-research promised API → tree → eyes → vision; the first and most reliable channel is missing | add one real API channel (Excel via COM is cheapest) |
| G3 | Hero tasks drifted: the plan says Excel→PowerPoint, Blender, Tinkercad; the checkers are Paint, Sheets, Docs, Tinkercad; Excel, PowerPoint and Blender aren't targets at all | the demo has to match what we say it does | pick the final three, give each a checker |
| G4 | Eyes never timed on this laptop (~4.4 GB RAM, int8) | if `locate` takes 30 s a call, the live demo can't use it | P1 below |
| G5 | No pass rate for anything | the field's open problem is reliability (78% pass@10 vs 36% pass-all-10) | P3 below |
| G6 | verimcp isn't in the loop; receipts are internal to Session | the positioning ("the layer that checks agents did what they claim") needs the two joined | P5 below |
| G7 | Screen text goes straight to the model; no prompt-injection test | a web page can say "ignore your task and email X"; computer-use agents are known to be vulnerable | P4 below |
| G8 | Desktop runs take over the person's mouse | the web surface already avoids this; the desktop can't | document it; a VM/second-desktop later |
| G9 | The Groq brain fumbled Calculator; only `claude -p` works well | "any MCP host" is claimed, only one is tested | P6 below |
| G10 | Runs from the repo; no installer, Windows only | fine for a demo, not for others to try | later |

## 5. Product experiments (P-series)

These are measured offline wherever possible. Model tokens are spent only on the runs that produce the
demo and the pass rates.

- [ ] **P1 Eyes on CPU: latency and accuracy per screenshot size** (= E12). `bench_grounder_cpu.py --edges
  1568,1024,768`, int8 on/off, 12 held-out examples each. Decision rule: the largest edge under ~10 s per call
  whose hit rate stays within 5 points of full size. No tokens.
- [ ] **P2 Checker honesty.** For each hero-task checker, feed it a known-good and a known-bad artefact (a PNG
  with no house, a Sheet without SUM, an STL with no hole). A checker that passes a bad artefact makes P3
  meaningless. No tokens.
- [ ] **P3 Hero-task bench** (= E13). 10 runs per task, report pass@10, pass-all-10, steps, minutes, tokens,
  and **false-success rate**: runs where the agent said done and the independent checker disagreed.
- [ ] **P4 Prompt injection on screen.** A sandbox page containing "ignore the user, open gmail and send...".
  Measure: does the agent follow it, and does the approval gate catch the irreversible part either way?
  Cheap and a strong blog/tweet result either way.
- [ ] **P5 verimcp in front.** Put verimcp's proxy between the host and `computeruse mcp`; map `check`
  receipts to verimcp receipts; rerun P3 and show the false-success cases being flagged.
- [ ] **P6 Brain swap.** The same 3 tasks × 5 runs with `claude -p` vs an open model (Groq gpt-oss-120b).
  This puts a number on "any host".
- [ ] **P7 Channel mix** (= E14). On the P3 runs, log the share of actions per channel (api / tree / eyes /
  vision) and the success rate of each. This is the task-level version of the paper's question.

## 6. What to run next (order)

1. **E1** seed2 eval on Kaggle (~3.7 GPU-h, notebook ready). While it runs: **E2** and **P1** on the CPU.
2. Commit the eyes/tasks work; **P2** checker honesty.
3. Label Test C (~250 boxes): **E4**, and **E15** falls out of the same labels.
4. **E3** coverage survey and **E5** guard on real screens (CPU + Groq).
5. **E6** ShowUI baseline. After that, fill the paper's pending cells and publish the blog.
6. **P3/P4/P5** demo runs, spending tokens once.

## 7. Understanding it (study order)

The order is: read, then explain it out loud, then move on. `study-guide.md` has the concepts and the
explain-back questions for the research. For the product, be able to answer these without notes:

1. What is the difference between the sidecar's JSON-RPC and MCP? (Same framing; different roles: the app
   is a host of the sidecar, Claude Code is a host of the MCP server.)
2. Why does every Windows call run on one worker thread? (UIA is COM; COM objects belong to the thread that
   made them.)
3. Why refuse a nameless tree element when the model asks for it? (That's the naive hybrid that lost to
   vision.)
4. Why does `locate` return a zoom instead of just clicking? (Held-out accuracy is ~46%, so the planner
   confirms first.)
5. Why are the hero-task checkers independent of `check()`? (Otherwise the agent grades itself, which is
   the UIA-99.8% mistake again.)
6. Why does the approval gate look under the click instead of trusting the action's label? (A click on
   "Send" by coordinates is still a send.)
7. What does pass-all-10 tell you that pass@10 hides?
