# What the accessibility tree can't see

*Blog draft v0 (2026-10-18). Plain-language version of `docs/paper/draft.md`. Before publishing: fill the
[PENDING] lines, add the figures, and check that you can explain every number out loud (the explain-back list
in `docs/study-guide.md`).*

---

When an AI agent uses your computer, every action starts with one small question: **where on the screen
is the thing I want to click?**

There are two ways to answer it.

**Ask the app.** Windows apps publish an *accessibility tree*: a list of every button, menu and text box, each
with a name and a rectangle. It exists for screen readers. If the tree says there's a button called "Save" at
(812, 44), you click (812, 44) and you're done: exact, instant, free.

**Look at the screen.** A vision model reads the screenshot the way you do and points at "Save". It works on
anything visible, but it can miss.

So the obvious design is: use the tree, and fall back to vision when the tree doesn't have it. That is roughly
what agent systems do today. I tested that design, and it's worse than vision alone.

## The tree doesn't know what it doesn't know

Trees have holes. An icon button with no name. A toolbar the app draws itself. Everything inside a canvas: a
Google Sheets grid, a Tinkercad workplane, a Paint drawing. GIMP's main window exposes **7** elements to
the tree.

The trouble isn't that the tree is missing things. It's that **it never tells you**. If you ask a name-matcher
for "Save" and the real Save button has no name, it doesn't answer "not here". It finds the closest name it
has ("Save As", "Settings") and confidently hands you the wrong rectangle.

So "tree first, else vision" doesn't fall back when it should. It overrides vision with a confident mistake.

## The experiment

First I needed a vision model that could click at all. Training data cost nothing, because the tree
provides it: every *named* element is a free example (screenshot, "click Save", box). I collected 1,572 of
these from 10 Windows apps and fine-tuned a small model (Qwen2-VL, 2B parameters, LoRA) on one free Kaggle GPU.

- On **4 apps it never saw**: 1.5% → **46.5%** of clicks land on the right element.
- On **ScreenSpot-v2**, the public benchmark (phones, Macs, websites): 19.2 → **62.8**. Phones improved most
  (+57 points), and it was never trained on a phone.

Then the comparison. To be fair to both sides, ground truth can't come from either one, so I generated web
pages where I know exactly where everything is, in five situations:

| situation | tree | vision | tree first, else vision |
|---|---|---|---|
| button with a clean name | **100** | 94 | 100 |
| icon with no name | 0 | **40** | 38 |
| instruction reworded ("store the file" for Save) | 0 | **31** | 30 |
| drawn on a canvas | 0 | **99** | 76 |
| two buttons with the same name | 26 | **65** | 26 |
| **average** | 25 | **66** | 54 |

The tree wins one row. The naive combination loses **12 points** to vision alone. I repeated it on real
Windows screens with the tree deliberately damaged, and got the same failure: **−9.5 points** whenever the
tree was incomplete.

## What works: let the tree say "I don't know"

The fix isn't a cleverer merge. It's giving the tree a way to abstain.

An LLM reads the tree as text and either names one element or says **NONE**. Its answer counts only if the
element it picked has a name. Otherwise, vision decides.

| | named | no name | reworded | canvas | same name | average |
|---|---|---|---|---|---|---|
| vision alone | 98 | 38 | 28 | 100 | 68 | 66.4 |
| LLM reading the tree | 88 | 2 | 88 | 0 | 100 | 55.6 |
| **LLM, named picks only, else vision** | 94 | 38 | 94 | 100 | 100 | **85.2** |

It takes the tree's strengths (rewording, duplicate names) and vision's (icons, canvas), and it's never
worse than vision on any row. The rule was chosen on separate dev data and scored on the test set exactly
once.

[PENDING E4: the same rule on real apps where the tree is genuinely blind.]
[PENDING E3: how often real apps are blind, per app.]

## Honest limits

- One training run so far. A second seed is being scored now. [PENDING E1]
- Only 4 unseen apps; the generated pages are synthetic; the real-blind test is still being labelled.
- The model learns only from elements the tree *names*, but its whole job is the elements it doesn't. That's
  the next thing I'm measuring.
- These are single clicks, not whole tasks. Task success rates come next.

## Where this goes

I'm building this into a product: an MCP server that gives any agent (Claude Code first) eyes and hands on
Windows apps and websites, with this rule enforced in code rather than left to the prompt. A tree element is
used only if it has a name. Unnamed things go to the trained grounder, which shows the agent a zoom to
confirm before it clicks. A step only counts as done when a checker sees the evidence.

*Code, notebooks and every frozen rule's commit: [repo link]. Adapter: [HF link].*

---

### Tweet thread draft

1. When an AI agent clicks a button, does it ask the app (the accessibility tree) or look at the screen? I
   measured both. The obvious combo, "tree first, else vision", is worse than vision alone. 🧵
2. Trees have holes (unnamed icons, canvases), and they don't say so. A name-matcher returns the closest
   wrong button with full confidence, and that overrides a correct vision answer.
3. I trained a 2B vision grounder on 1.5K examples the tree labelled for free, on one free GPU: unseen apps
   1.5 → 46.5%, ScreenSpot-v2 19 → 63 (+57 on phones, never trained on one).
4. The fix: let an LLM read the tree and say NONE, and trust its pick only if the element has a name. 85.2 vs
   66.4 for vision alone, never worse on any condition. Chosen on dev, tested once.
5. Building it into an MCP server so any agent gets this rule in code. Blog + numbers: [link]
