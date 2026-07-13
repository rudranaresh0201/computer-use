# Learning Track

Rewritten 2026-07-11 — the original version of this file assumed zero ML background, which was wrong (see `docs/journal.md` and project memory). This version is matched to actual production-level fluency: RAG, LangGraph/CrewAI multi-agent systems, PyTorch (LSTM and transformer models already trained and shipped), MCP. No 101-level material below — if something here feels like it needs a beginner explanation first, that's a signal to say so, not a gap this file should pre-fill.

## What's actually new, and when it's needed

Everything through Phase 2 (LangGraph orchestrator) is inside existing expertise — no assigned reading, build first, read only if something specific comes up mid-build.

**Before Phase 3 (GUI grounding model) — this is the genuinely new territory:**

Fine-tuning a *pretrained multimodal model* is a different exercise from training the LSTM/transformer-WAF models already shipped — those were trained from scratch (or near it) on a defined task; this is adapting an existing large VLM's behavior efficiently, without retraining its whole weight set.

- **What a VLM actually is, architecturally** (added 2026-07-13, once the base model was concretely picked and code was written against its real tokenizer): a vision encoder (ViT) chops an image into patches and turns each into a token-space embedding; a projector maps those into the same embedding space the language model's text tokens live in; the same decoder-only transformer then just sees a mixed `[visual tokens, text tokens]` sequence instead of only text. This is why the base model choice is a VLM (Qwen2-VL-2B) and not T5 — our input genuinely is (image, instruction), a VLM's native shape, and T5 has no image encoder at all.
- **Qwen2-VL Technical Report** (Wang, Bai et al., arXiv:2409.12191) — read this first, since it's literally our base model, not a generic example. Focus on *Naive Dynamic Resolution* (how an image becomes a variable number of visual tokens — explains the `<|image_pad|>` tokens the real tokenizer renders) and *M-RoPE* (multimodal rotary position embeddings — the actual mechanism that lets the model localize spatially at all, which is what our whole grounding task depends on).
- **LoRA (Low-Rank Adaptation) — the original paper** (Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685). The technique SeeClick/UGround/OS-Atlas all use to fine-tune their base VLM. Short, mechanism is simple once seen: freeze the pretrained weights, inject small trainable low-rank matrices alongside them, train only those. Worth reading the actual mechanism, not just "LoRA = cheap fine-tuning," since Phase 3's training script will directly implement this.
- **HuggingFace PEFT docs** (huggingface.co/docs/peft) — the practical knobs (`r`, `alpha`, `target_modules`) the LoRA paper doesn't spell out operationally. This is the bridge from "understand LoRA" to "know what to type in the config" for the actual training script.
- **Re-read `docs/research/gui-grounding-research.md`** (already written) — specifically the documented data-quality problem (accessibility-tree elements that exist structurally but aren't visually rendered), and re-read the SeeClick/UGround summaries now that the VLM architecture itself is understood, not just the training-data recipe.

**Optional, if curious about where GUI-Actor's coordinate-free approach came from:** the attention/transformer mechanics under "region proposal" style grounding vs. raw coordinate regression — not required to build Phase 3's first version, only relevant if raw-coordinate fine-tuning proves noisy and a fallback approach is needed.

## Ongoing reference, not sequenced

- **mlabonne/llm-course** (GitHub) — useful as a lookup table for any term that comes up mid-build without a clear definition, not a start-to-finish read.
- **Anthropic's engineering blog**, **simonwillison.net** — worth checking periodically as the agent/GUI-automation space moves; not project-blocking reading.

## Explicit non-goals

- No neural-network-basics material — already known.
- No RL — not part of this project (ADR-0003 explicitly notes the Brain is a foundation-model reasoner, not a trained policy; that's a deliberate contrast with RL-based control approaches seen elsewhere, not a gap to fill).
- No deep-math derivation of attention/backprop — useful only if it starts blocking a specific decision, not proactively.
