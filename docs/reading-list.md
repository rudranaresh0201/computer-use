# Reading list: the computer-use field, explained

What people working on computer-use agents assume you know, in the order to learn it. Each entry: **what it is**,
**the key idea**, **how it touches your work**, **remember**. arXiv ids are given from memory -- if one doesn't open,
search the title. Don't quote other papers' numbers from this file; read them in the paper first.

How to read a paper fast: abstract → the main figure → the results table → the limitations. Only then the method.
Read 2 a day, not 10. After each one, write three lines: *what they did, what they found, how it relates to mine.*

---

## Week 1 — the words everyone uses

| term | meaning |
|---|---|
| **computer-use agent (CUA)** | a model that operates a computer through the screen: look, decide, click/type, repeat |
| **GUI grounding** | instruction + screenshot → where to click. Your model does exactly this |
| **planning vs grounding** | *what* to do next ("open the File menu") vs *where* that is on screen. Many systems split them: a big LLM plans, a small model grounds. Your desktop agent is a planner over the tree; your VLM is a grounder |
| **accessibility tree (a11y tree)** | the list of UI elements an app reports to the OS for screen readers. Windows: UIA. Mac: AX. Linux: AT-SPI. Web: the browser's accessibility tree (built from HTML + ARIA) |
| **action space** | the set of actions the agent may emit: click(x,y), type, key, scroll, drag... Anthropic's computer-use tool defines a common one; your `action/schema.py` mirrors it |
| **observation** | what the agent sees each step: screenshot, tree, or both. "Observation ablation" = comparing these — your research question |
| **Set-of-Mark (SoM)** | draw numbered boxes on the screenshot so the model can answer "click 12" instead of coordinates |
| **trajectory / episode** | one full attempt at a task: every observation and action |
| **task success rate** | % of tasks finished correctly, judged by a checker. The headline metric of agent benchmarks |
| **native agent** | one model that sees pixels and outputs actions end to end, no tree, no separate planner (UI-TARS) |
| **pre-registration, dev/test, bootstrap CI, paired difference** | your own evaluation vocabulary — `docs/study-guide.md` section 1 |

---

## Week 1 — foundations you must be able to explain

**1. Qwen2-VL** (Wang et al., 2024, arXiv 2409.12191) — *your base model.*
- What: Alibaba's open vision-language model family (2B/7B/72B).
- Key idea: images at their **native resolution** cut into patches (not squashed to a fixed size), and a position
  encoding (M-RoPE) that knows each patch's 2-D place — why it can point at small UI elements at all.
- Your work: you fine-tuned the 2B. Know how an image becomes tokens and why big screenshots cost more.

**2. LoRA** (Hu et al., 2021, arXiv 2106.09685) — *how you trained.*
- Key idea: freeze the big weight matrices, learn two thin matrices whose product is the update. ~1% of the
  parameters train, so a 2B model fits a free T4.
- Remember: rank, which layers get adapters, why the learning rate is higher than full fine-tuning.

**3. Accessibility APIs** — *your "tree" side.*
- Read: Microsoft's "UI Automation Overview" (learn.microsoft.com) — control types, properties, patterns
  (InvokePattern = press a button without the mouse).
- Chrome's accessibility tree: open DevTools → Elements → Accessibility pane on any site.
- Remember: the tree is written by the app's developers; that's why it can be missing, nameless, or wrong. Electron
  and Chrome often build the full tree only when they detect a screen reader.

---

## Week 2 — the benchmarks (people quote these constantly)

**4. MiniWoB++** (2017-18) — tiny synthetic web tasks ("click the button labeled OK"). Historical; solved now. Your
Test A pages are in the same spirit, but built to isolate conditions.

**5. Mind2Web** (Deng et al., 2023, arXiv 2306.06070) — real websites, offline: given the page and the task, predict
the next action. Introduced the idea of generalist web agents across many sites.

**6. WebArena** (Zhou et al., 2023, arXiv 2307.13854) — self-hosted realistic websites (shop, forum, GitLab, maps)
with **automatic checkers**. The model for your planned "sandbox with checkers" (experiment-plan S2).
VisualWebArena (2024) adds tasks that need the image.

**7. OSWorld** (Xie et al., 2024, arXiv 2404.07972) — **the** desktop benchmark: real Ubuntu/Windows VMs, ~370 tasks
across real apps, scripted checkers. Compared observations: screenshot, a11y tree, both, Set-of-Mark — the
task-level version of your question. Read its observation ablation carefully: it is your closest prior work.

**8. Windows Agent Arena** (Microsoft, 2024, arXiv 2409.08264) — OSWorld-style tasks on Windows; introduced the
Navi agent and uses UIA + OmniParser-style parsing. Your Windows setting.

**9. ScreenSpot** (inside SeeClick, 2024) / **ScreenSpot-v2** (fixed labels, from OS-Atlas) /
**ScreenSpot-Pro** (2025, arXiv 2504.07981) — grounding benchmarks. You ran v1 and v2. **Pro** uses high-resolution
screenshots of professional software (CAD, IDEs, creative tools) with tiny targets; models score far lower there.
It is where small-target weakness (your 15%-on-smallest-quartile finding) shows up. A natural next eval.

**10. AndroidWorld / AndroidControl** (2024) — the phone equivalents. Know they exist.

---

## Week 2 — the grounding models (your direct neighbours)

**11. SeeClick** (Cheng et al., 2024, arXiv 2401.10935) — first well-known "grounding pre-training" from web data
(HTML gives free labels). **The same trick as yours** (free labels from structure → train vision), on web.

**12. CogAgent** (Hong et al., 2023, arXiv 2312.08914) — an early large GUI VLM with a high-resolution branch.

**13. Ferret-UI** (Apple, 2024) — grounding and referring on mobile screens; handles small elements with sub-image
"any resolution" crops.

**14. OS-Atlas** (Wu et al., 2024, arXiv 2410.23218) — large cross-platform grounding corpus, including desktop data
synthesised with accessibility trees. **Did your training idea at scale — cite it, don't claim it.** Also made
ScreenSpot-v2.

**15. UGround** (Gou et al., 2024, arXiv 2410.05243) — argues agents should "navigate like humans": pure vision,
trained on large synthetic web data; strong grounding with a planner LLM on top.

**16. ShowUI** (Lin et al., 2024, arXiv 2411.17465) — 2B on Qwen2-VL like you; drops redundant screen patches to save
compute. Your nearest same-size comparison.

**17. Aria-UI** (2024, arXiv 2412.16256) — pure-vision grounding, uses context of past actions.

**18. UI-TARS** (ByteDance, 2025, arXiv 2501.12326), then **UI-TARS-1.5** and **UI-TARS-2** — the strongest open
**native agent**: screenshot in, reasoning + action out, trained on huge data incl. agent trajectories and
reflection; open weights and a desktop app (UI-TARS-desktop). Ignores the tree. In your project it is a better
"vision" slot, not a competitor to the trust-rule question.

**19. GroundCUA / GroundNext** (2025, arXiv 2511.07332) — 56K human-annotated desktop screenshots, 87 apps; human
labels instead of tree labels. The quality end of desktop grounding data.

**20. The 2025 "R1" wave** — GRPO-style reinforcement learning on grounding (e.g. UI-R1, GUI-R1, InfiGUI-R1): reward
= did the click land in the box. **Directly connects to your GRPO-JAX project**: your hit rule is already a reward.

---

## Week 3 — systems and agents (what products are built from)

**21. Set-of-Mark prompting** (Yang et al., 2023, arXiv 2310.11441) — numbered marks on images let GPT-4V point.
Every "click [12]" agent descends from this.

**22. SeeAct** (Zheng et al., 2024, arXiv 2401.01614) — "GPT-4V is a generalist web agent, *if grounded*": showed the
big model can plan but fails at grounding — the reason grounders like yours exist.

**23. OmniParser** (Microsoft, 2024, arXiv 2408.00203; v2 in 2025) — turns a screenshot into a list of elements
(detector + captioner) — i.e. builds a fake accessibility tree **from pixels**. Important alternative to yours: they
recreate the tree with vision; you combine the real tree with vision.

**24. UFO / UFO2** (Microsoft, arXiv 2402.07939 / 2504.14603) — Windows agents. UFO2 merges UIA + OmniParser
detections into one hybrid element list. **Closest system to your hybrid** — read how they merge and whether they
measure when the merge is wrong (we found no such measurement; confirm).

**25. Agent S / Agent S2** (Simular, arXiv 2410.08164 / 2025) — planner + specialised grounders ("mixture of
grounding"), strong on OSWorld. The planner-grounder split done well.

**26. Products** — Anthropic Computer Use (Oct 2024, the API tool and its reference loop your Phase 1 studied),
OpenAI Operator / CUA (2025; ChatGPT agent mode), Google Project Mariner (Chrome agent), browser-use (open-source
Python, a11y-tree + screenshot), Microsoft Playwright MCP (browser control as an MCP server, tree-based — the
model for your flagship).

---

## Week 3 — safety and evaluation of agents (your verimcp angle)

**27. Prompt injection on agents** — a web page saying "ignore your instructions and ...". Read OS-Harm
(arXiv 2506.14866) for a desktop safety benchmark; search "EIA environmental injection attack" for web.
**28. "Agent says done but isn't"** — the false-success problem (your experiment-plan S3). OSWorld-style checkers
exist because self-reports can't be trusted. This is verimcp's pitch: verify claims against the world.

---

## Suggested 3-week plan (1-1.5 h a day)

| days | read | output |
|---|---|---|
| 1-2 | glossary; Qwen2-VL (sections 1-3); LoRA | explain both out loud in 2 minutes |
| 3-4 | UIA overview; inspect Chrome's a11y pane on 3 sites | list 5 ways a tree can be wrong |
| 5-6 | OSWorld (esp. observation ablation); Windows Agent Arena | how is their question different from yours? |
| 7-8 | SeeClick; OS-Atlas | the 3-line "what's ours vs theirs" |
| 9-10 | ShowUI; UI-TARS | why UI-TARS is a better vision slot, not a rival |
| 11-12 | Set-of-Mark; SeeAct; OmniParser | planner vs grounder, tree vs pixel-parsed tree |
| 13-14 | UFO2; Agent S2 | where your trust rule would plug into each |
| 15 | ScreenSpot-Pro; one R1-style grounding paper | link to your GRPO project |
| 16 | OS-Harm + your verimcp README | the "verified computer use" pitch in 5 sentences |

After this you can hold a conversation with anyone in the field — and you'll see exactly where your work sits:
*others built better eyes or bigger agents; you measured when to trust the tree and when to trust the eyes.*
