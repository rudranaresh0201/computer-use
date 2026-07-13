"""
Tests for training/lora_config.py. Deliberately checks only the config
values here (no network, no model download) -- the real verification,
that this config actually attaches to Qwen2-VL-2B's real architecture and
produces a sane trainable-parameter percentage, was done as a one-off
check against the real model (instantiated on the `meta` device, no
weight download) rather than a committed test, to keep the pytest suite
network-free like the rest of this project's tests.
"""

from computeruse.training.lora_config import TARGET_MODULES, build_lora_config


def test_target_modules_are_the_verified_decoder_projection_names():
    assert set(TARGET_MODULES) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


def test_target_modules_excludes_vision_tower_and_lm_head():
    # the vision tower uses a different naming family (qkv/proj/fc1/fc2)
    # and lm_head is the final output projection -- neither should be
    # targeted by this config.
    excluded = {"qkv", "proj", "fc1", "fc2", "lm_head"}
    assert excluded.isdisjoint(TARGET_MODULES)


def test_build_lora_config_defaults_match_documented_values():
    config = build_lora_config()
    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.1
    assert set(config.target_modules) == set(TARGET_MODULES)
    assert config.task_type == "CAUSAL_LM"


def test_build_lora_config_accepts_overrides():
    config = build_lora_config(r=8, lora_alpha=16, lora_dropout=0.05)
    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
