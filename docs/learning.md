# Learning Track (zero math/ML background)

Goal: understand *why* the system we're building works, not become an ML researcher. No math background assumed. This track is sequenced to unlock exactly when you need it in `docs/roadmap.md` — don't front-load all of it, you'll forget it before it's relevant. Watch/read the entry for a phase around the time we start that phase.

## Before Phase 0 is "done" (this week) — the absolute minimum
You need one mental model: **a model like Claude takes in text + images and predicts what comes next, one piece at a time; "tool use" is just the model predicting a structured piece of text that your code interprets as an instruction ("click here") instead of showing it to a human.** That's it — that's the whole trick behind computer-use. Everything else is engineering around that one fact (how do you get a screenshot to it, how do you turn its answer into a real mouse click, how do you keep it from going in circles).

- **Andrej Karpathy — "Deep Dive into LLMs like ChatGPT"** (YouTube, ~3.5 hrs, free). Covers the full stack of how these models work and are used, no math background required. This single video is enough context to understand every design decision in `docs/decisions/`.
- **3Blue1Brown — "But what is a Neural Network?"** (YouTube, ~20 min). Pure visual intuition, zero equations required to follow it. Answers "what is actually happening inside the box."

You do not need to understand backpropagation, gradient descent, or any training math to build this project — we are only ever *calling* pretrained models via an API or a local runtime, never training one. Skip anything that starts talking about "loss functions" or "optimizers" for now; it's not blocking us.

## Before Phase 2 (agent loop) — what "the loop" actually is
- **3Blue1Brown — visual intro to Transformers/attention** (YouTube). Gives you enough to understand *why* the model can "look at" a screenshot and a block of instruction text together and reason about both — this is what makes multimodal tool-use possible at all.
- Re-read `docs/research/anthropic-computer-use.md` "The core loop" section after watching these two — it should read as obvious mechanics rather than magic.

## Before Phase 3 (real brain integration) — vision models & grounding
This is where "why is clicking the right pixel hard" becomes concrete.
- **StatQuest (Josh Starmer) — any of the "Neural Networks" or "Image Recognition" playlist videos.** Best plain-English explanations on YouTube of how a model turns pixels into "this is a button" style understanding.
- Skim `docs/research/prior-art.md` again, specifically the OmniParser and OSWorld sections — the "grounding is the hard part, not planning" finding will make much more sense once you've seen how vision models actually process an image (as a grid of small learned features, not as a picture the way you see it).
- Optional deeper dive if curious: **Karpathy — "Neural Networks: Zero to Hero"** series (karpathy.ai/zero-to-hero.html) actually builds things by hand in code, including calculus-lite backprop. Good if you want to eventually understand training, not required to keep building this project.

## Ongoing / reference, dip in as needed
- **mlabonne/llm-course** (GitHub) — structured, free, roadmap-style course covering LLM fundamentals through fine-tuning/deployment. Good as a table of contents to search when a specific term in a paper confuses you (e.g. "what's a token", "what's an embedding").
- **DeepLearning.AI short courses** — free, ~1 hour each, very applied (prompting, agents, evals). Good for Phase 4/6 when we build the eval suite and want to know what "a good agent eval" looks like elsewhere.

## Explicit non-goals for this learning track
- No linear algebra / calculus needed to *use* these models via API — that math lives inside the model, which is already trained. We'd only need it if we were training our own model from scratch, which is out of scope for this project.
- No need to learn a deep-learning framework (PyTorch/TensorFlow) unless a later phase specifically calls for running a local model in a way that needs custom code (unlikely — Ollama abstracts this away for Phase 3).

Sources: [Karpathy Zero-to-Hero](https://karpathy.ai/zero-to-hero.html), 3Blue1Brown & StatQuest (YouTube, search channel name), [mlabonne/llm-course](https://github.com/mlabonne/llm-course).
