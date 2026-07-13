"""
The LoRA adapter configuration for fine-tuning Qwen2-VL-2B-Instruct.

target_modules is not guessed -- it's the real Linear layer names read
directly from transformers' installed Qwen2-VL modeling source (standard
LLaMA-family decoder naming: q_proj/k_proj/v_proj/o_proj for attention,
gate_proj/up_proj/down_proj for MLP), then re-confirmed by instantiating
the actual model architecture on the `meta` device (no weight download,
just the module graph) and attaching this exact config via peft.

Deliberately targets only the language-model decoder's projections, not
the vision tower (qkv/proj/fc1/fc2, a different naming family entirely)
or lm_head -- matches the SeeClick/UGround/GUI-Actor recipe of adapting
the language backbone's reasoning over already-encoded visual tokens,
not re-training vision perception itself.

r=16/alpha=32/dropout=0.1 starts from Mehul's Re:Fine Phase 1 (CodeT5 +
LoRA) config as a real working reference point rather than a guess --
see docs/journal.md for that comparison. Attaching this config to the
real Qwen2-VL-2B architecture (2,227,450,368 params) trains 18,464,768 of
them: 0.829% -- the same order of magnitude as Mehul's 0.787% on a ~10x
smaller model, a sanity check that this is a reasonable starting config,
not a wild guess.
"""

from __future__ import annotations

from peft import LoraConfig

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(TARGET_MODULES),
        task_type="CAUSAL_LM",
    )
