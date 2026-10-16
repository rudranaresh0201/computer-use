# Product research: from a Calculator demo to a reason → ground → act → verify system

Written 2026-10-16. Sources are listed at the end; numbers marked *vendor* or *unverified* are not ours to quote
as fact. Companion to `experiment-plan.md` (the research) and `related-work.md` (what's already claimed).

## Where the field is

- **Short desktop tasks are close to solved for frontier models.** OSWorld-Verified: top agents report ~84-90%
  (mostly *vendor*-reported) vs a ~72% human baseline. The harder **OSWorld 2.0**: best full completion ~44-49%.
- **Reliability is the open problem, not peak skill.** Agent S3 + GPT-5: ~78% of tasks pass at least once in 10
  runs, only ~36% pass all 10 (arXiv 2604.17849). 95% per step over 20 steps leaves ~36% end to end.
  → our demo reports **N/10 runs, every checkpoint verified**, not one lucky take.
- **Pro software is the new frontier:** CADWorld (2609.16251, CAD, spatial precision), ProSoftArena (AutoCAD,
  SolidWorks), DeskCraft (design/video/audio/3D), ScreenSpot-Pro (grounding in pro apps).
- **Hybrid API + GUI is where systems are heading:** UFO2's Puppeteer prefers app APIs (Office COM) and falls back
  to the GUI; MCPWorld (2506.07672) benchmarks API vs GUI vs hybrid agents (201 tasks, white-box checkers);
  WeaveBench (2606.09426) requires GUI + CLI/code in one trajectory, best pass rate 41.2%.
- **Open-source agents split planner and grounder:** Agent S3 (reasoning model + UI-TARS-1.5-7B grounder, 72.6%
  OSWorld per its README), UI-TARS-desktop (native agent), Cua (VM sandboxes), Bytebot (desktop in a container).
- **Planning is the latency bottleneck** (>50% of task time in two open agents, OSWorld-Human 2506.16042).

## What that means for ours

The research result ("trust the tree only when it is sure, else the eyes") is exactly the routing question these
systems face. The product makes it visible: every step shows *which channel* it used and *the proof* it worked.

1. **Reason:** Claude Code (`claude -p`, via MCP) plans a checklist and thinks before each step. Pluggable: any MCP
   host, or Groq/OpenAI-compatible standalone.
2. **Ground:** channel router, in order: app API → accessibility tree (named elements only, the `llm_named`
   rule) → our grounder (eyes) → the planner's own vision.
3. **Act:** mouse/keyboard (`action/controller.py`), UIA patterns, Playwright, API calls.
4. **Verify:** per-checkpoint checker (API readback, tree state, file contents, screenshot diff), retry, receipts
   (verimcp).

## Apps and their automation surfaces

| app | where | channels | tree | role in the demo |
|---|---|---|---|---|
| Excel / PowerPoint | desktop | COM (win32com) + UIA | good | useful cross-app report; most reliable hero |
| Blender | desktop | Python API (cf. blender-mcp) + eyes on the viewport | blind viewport | 3D scene + render |
| Tinkercad | web | shortcuts (official sheet) + eyes | WebGL canvas: blind | keychain → STL; the eyes are essential |
| Google Docs / Sheets | web | a11y tree is thin: content is canvas-rendered | mostly blind without screen-reader mode | everyday apps where the eyes matter |
| Gmail | web | a11y tree | good | send = irreversible → approval gate |
| MS Paint | desktop | UIA (173 elements, 124 named) + canvas drags | toolbar yes, canvas no | drawing |
| AutoCAD | desktop/web | command line + AutoLISP/.NET; free education licence | partial | pro CAD |
| Onshape | web | REST API + FeatureScript (cf. onshape-mcp) | partial | pro CAD |
| Photopea | web | `app` script API (Photoshop-like) | partial | image editing |
| GIMP | desktop | Script-Fu + eyes | **7 UIA elements** (our capture) | research showcase |
| Excalidraw / tldraw | web | editor API (`excalidrawAPI.updateScene`, tldraw Editor) + shortcuts | 44 / 20 named; drawings invisible | diagrams |
| FreeCAD / KiCad | desktop | Python console / pcbnew + IPC | partial | stretch |
| OBS | desktop | obs-websocket | n/a | the agent records its own demo |

Element counts are from `runs/realtest` (2026-10-13). "Blind" = what the tree cannot see, which is where the
grounder we trained earns its place.

## Hero tasks (chosen 2026-10-16)
1. **Excel → PowerPoint report**: CSV → sheet with formulas + chart → 3-slide deck. COM + UIA, cross-app.
2. **Blender**: low-poly island with a palm tree, rendered. API builds, the eyes verify the viewport.
3. **Tinkercad**: keychain with a name, exported as STL. Shortcuts + eyes on a WebGL canvas.
Each one gets a checker and 10 runs.

## Risks
- Grounder speed on this laptop's CPU (no NVIDIA GPU) is unmeasured: measure before relying on it; crop-to-region
  cuts image tokens and helps small targets (our 15% smallest-quartile weakness).
- Google Docs/Sheets/Tinkercad UIs change; tasks are pinned to what works on the recording date.
- `claude -p` runs under the user's subscription: fine for personal demos, not something to host for others.

## Sources
- OSWorld leaderboards: presenc.ai/research/osworld-computer-use-leaderboard-2026, leaderboard.steel.dev, pointer.ai/blog/sota
- Reliability: arxiv.org/abs/2604.17849 · efficiency: arxiv.org/abs/2506.16042 · verifiers: arxiv.org/abs/2604.06240
- UFO2: arxiv.org/abs/2504.14603 · MCPWorld: arxiv.org/abs/2506.07672 · WeaveBench: arxiv.org/abs/2606.09426
- CADWorld: arxiv.org/abs/2609.16251 · ProSoftArena: arxiv.org/abs/2601.02399 · DeskCraft: arxiv.org/abs/2606.03103
- Excalidraw API: docs.excalidraw.com (excalidrawAPI) · tldraw: tldraw.dev/features/programmatic-control
- Onshape: onshape-public.github.io/docs · github.com/hedless/onshape-mcp · Photopea: photopea.com/learn/scripts
- Blender MCP: pypi.org/project/blender-mcp · Tinkercad shortcuts: Tinkercad's official shortcut PDF
