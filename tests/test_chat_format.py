"""
Tests for training/chat_format.py's conversation structure. Deliberately
does not load the real tokenizer here (that requires a network download
and is slow/flaky for a unit test suite) -- the structure this builds is
tested directly, and the real chat-template round-trip was verified
separately as a one-off check against the live Qwen2-VL-2B-Instruct
tokenizer.
"""

from computeruse.training.chat_format import SYSTEM_PROMPT, to_conversation
from computeruse.training.prepare_dataset import TrainingExample


def _example() -> TrainingExample:
    return TrainingExample(
        id="notepad_0000_e0",
        image_path="images/notepad/notepad_0000.png",
        prompt="Click the Save button",
        target="(500,500)",
        app="notepad",
        split="train",
    )


def test_conversation_has_system_user_assistant_turns():
    conversation = to_conversation(_example())
    roles = [turn["role"] for turn in conversation]
    assert roles == ["system", "user", "assistant"]


def test_system_turn_is_the_shared_grounding_instruction():
    conversation = to_conversation(_example())
    assert conversation[0]["content"] == SYSTEM_PROMPT


def test_user_turn_carries_image_then_text_content_blocks():
    conversation = to_conversation(_example())
    user_content = conversation[1]["content"]
    assert user_content[0] == {"type": "image", "image": "images/notepad/notepad_0000.png"}
    assert user_content[1] == {"type": "text", "text": "Click the Save button"}


def test_assistant_turn_is_the_raw_normalized_coordinate_target():
    conversation = to_conversation(_example())
    assert conversation[2] == {"role": "assistant", "content": "(500,500)"}
