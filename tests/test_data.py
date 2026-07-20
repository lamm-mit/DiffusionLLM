"""Dataset preprocessing tests. Run with ``PYTHONPATH=src pytest tests/test_data.py``."""

from __future__ import annotations

from datasets import Dataset

from diffusion_llm.data import prepare_sft


class FakeEncoding:
    """Minimal stand-in for the fast tokenizers.Encoding return type."""

    def __init__(self, ids: list[int]):
        self.ids = ids


class EncodingChatTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        if add_generation_prompt:
            return FakeEncoding([10, 11])
        return FakeEncoding([10, 11, 12, 13])


def test_sft_normalizes_chat_template_encoding_objects() -> None:
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ]
    )

    train, evaluation = prepare_sft(
        dataset,
        None,
        tokenizer=EncodingChatTokenizer(),
        max_length=32,
        mask_prompt_loss=True,
        num_proc=1,
        max_train_samples=None,
        max_eval_samples=None,
    )

    assert evaluation is None
    assert train[0]["input_ids"] == [10, 11, 12, 13]
    assert train[0]["labels"] == [-100, -100, 12, 13]
