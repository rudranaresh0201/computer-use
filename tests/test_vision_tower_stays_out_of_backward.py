"""
The 2026-10-11 Kaggle OOM, reproduced small.

transformers 5.0.0's `enable_input_require_grads()` hooks every sub-model's
input embeddings, including Qwen2-VL's vision PatchEmbed -- which drags the
frozen 32-layer vision tower into every backward pass. On the T4 that
backward asked for one vision layer's full attention matrix (3.06 GiB) and
died on step 1.

A tiny Qwen2-VL (real modeling code, toy sizes) runs one real
forward+backward with the same LoRA config and reentrant gradient
checkpointing the training run uses, and counts backward calls into the
vision blocks.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
# Pinned in pyproject's training extra, and the only version the Kaggle run
# installs. Later releases changed Qwen2-VL's forward (mm_token_type_ids is
# now required), which is the reason for the pin in the first place.
pytestmark = pytest.mark.skipif(
    transformers.__version__ != "5.0.0",
    reason=f"tests the pinned transformers==5.0.0 behavior; installed {transformers.__version__}",
)
from peft import get_peft_model  # noqa: E402
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration  # noqa: E402

from computeruse.training.lora_config import build_lora_config  # noqa: E402
from computeruse.training.train_lora import keep_vision_tower_out_of_backward  # noqa: E402

IMAGE_TOKEN = 151655


def _tiny_model():
    torch.manual_seed(0)
    config = Qwen2VLConfig(
        vision_config={
            "depth": 2, "embed_dim": 32, "num_heads": 2, "hidden_size": 64, "mlp_ratio": 2,
            "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3,
        },
        text_config={
            "vocab_size": 151936, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
        },
        image_token_id=IMAGE_TOKEN,
    )
    model = Qwen2VLForConditionalGeneration(config)
    model.config.use_cache = False
    model = get_peft_model(model, build_lora_config())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    model.train()
    return model


def _batch():
    grid = torch.tensor([[1, 4, 4]])  # 16 patches -> 4 image tokens after 2x2 merge
    pixel_values = torch.randn(16, 3 * 2 * 14 * 14)
    ids = torch.tensor([[1, 2] + [IMAGE_TOKEN] * 4 + [3, 4, 5, 6]])
    labels = ids.clone()
    labels[:, :6] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": pixel_values,
            "image_grid_thw": grid, "labels": labels}


def _run(model) -> tuple[bool, int, int, int]:
    """One real forward+backward. Returns (vision output in the graph?,
    backward calls into vision blocks, lora_B params with a gradient, total lora_B)."""
    visual = model.base_model.model.model.visual
    in_graph = {}
    calls = []
    def record(module, inputs, output):  # must return None: a returned value REPLACES the output
        # The tower's OUTER output. A block's own output reads False under
        # reentrant checkpointing even when the block is in the graph,
        # because checkpoint runs the block body under no_grad.
        tensor = output if torch.is_tensor(output) else getattr(output, "pooler_output", None)
        if tensor is None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output.last_hidden_state
        in_graph.setdefault("v", tensor.requires_grad)

    visual.register_forward_hook(record)
    for block in visual.blocks:
        block.register_full_backward_hook(lambda m, gi, go: calls.append(1))
    model(**_batch()).loss.backward()
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    with_grad = sum(1 for p in lora_b if p.grad is not None and p.grad.abs().sum() > 0)
    return in_graph["v"], len(calls), with_grad, len(lora_b)


def test_gradient_checkpointing_alone_pulls_the_frozen_vision_tower_into_backward():
    # _tiny_model() calls gradient_checkpointing_enable(), exactly as the
    # Trainer does -- that call alone hooks the vision PatchEmbed.
    in_graph, calls, _, _ = _run(_tiny_model())
    assert in_graph and calls > 0  # the bug; if this fails, transformers changed


def test_vision_tower_runs_without_autograd_and_every_lora_layer_still_learns():
    model = _tiny_model()
    model.enable_input_require_grads()  # what build_trainer still does, on purpose
    keep_vision_tower_out_of_backward(model)
    in_graph, calls, with_grad, total = _run(model)
    assert not in_graph and calls == 0
    # lora_B starts at zero, so on step 1 it is lora_B that gets the gradient.
    assert with_grad == total > 0
