"""
Wraps (image, prompt, target) triples from training/prepare_dataset.py in
Qwen2-VL's real chat template -- verified against the actual tokenizer
config shipped with Qwen/Qwen2-VL-2B-Instruct on Hugging Face, not guessed
or copied from a different model family's convention.

This is the piece prepare_dataset.py deliberately deferred. T5-family
models (see the peer-project comparison in docs/journal.md) expect a
task-prefix string like "summarize: ..." baked into the input, because
that's how T5 was pretrained multi-task. Qwen2-VL is decoder-only and
chat/instruction-tuned instead, so its expected input shape is a list of
{role, content} turns rendered through <|im_start|>/<|im_end|> tokens, not
a prefix string -- using the wrong convention wouldn't error, it would
just train the model against a shape it was never taught to expect.

Only needs the tokenizer (a few MB of config/vocab files), not the ~4GB
model weights -- this step verifies prompt structure, not inference.
"""

from __future__ import annotations

from transformers import AutoTokenizer

from .prepare_dataset import TrainingExample

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

SYSTEM_PROMPT = (
    "You are a GUI grounding assistant. Given a screenshot and an "
    "instruction, respond with only the target point as (x,y), normalized "
    "to a 0-1000 range on both axes."
)


def to_conversation(example: TrainingExample) -> list[dict]:
    """Build the Qwen2-VL multimodal conversation structure for one
    training example. The image content block references the file by
    path -- the Kaggle notebook that actually loads pixels resolves this
    against the dataset root; this module only cares about text/turn
    structure, not image decoding."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example.image_path},
                {"type": "text", "text": example.prompt},
            ],
        },
        {"role": "assistant", "content": example.target},
    ]


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def render(example: TrainingExample, tokenizer) -> str:
    """Render one example through the tokenizer's real chat template,
    returning the exact string the model will be trained on."""
    return tokenizer.apply_chat_template(to_conversation(example), tokenize=False)
