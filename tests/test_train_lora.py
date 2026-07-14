"""
Tests for training/train_lora.py. build_training_args is pure config, no
model/network needed. build_trainer/main download the real ~4GB base
model and need a GPU to run in reasonable time -- not exercised here,
consistent with this being a Kaggle/Colab entrypoint, not something this
machine runs.
"""

from pathlib import Path

from computeruse.training.train_lora import build_training_args


def test_training_args_match_documented_starting_hyperparameters():
    args = build_training_args(Path("runs/test"))
    assert args.learning_rate == 5e-4
    assert args.num_train_epochs == 1


def test_training_args_use_gradient_accumulation_to_fit_a_free_tier_gpu():
    # per_device_train_batch_size=4 OOM'd on a 15GB Kaggle T4 (fp16 weights
    # + activations at batch 4 didn't fit). batch_size=1 with
    # gradient_accumulation_steps=4 keeps the same effective batch size
    # with a much smaller peak memory footprint; gradient_checkpointing
    # trades compute for memory on top of that.
    args = build_training_args(Path("runs/test"))
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert args.gradient_checkpointing is True


def test_training_args_evaluates_and_saves_every_epoch():
    args = build_training_args(Path("runs/test"))
    assert args.eval_strategy == "epoch"
    assert args.save_strategy == "epoch"


def test_training_args_keeps_unused_columns_for_custom_collator():
    # our collate_fn expects pixel_values/image_grid_thw, which the
    # default HF column-pruning would otherwise silently drop.
    args = build_training_args(Path("runs/test"))
    assert args.remove_unused_columns is False
