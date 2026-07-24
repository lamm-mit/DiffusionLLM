"""Tests for reproducible held-out generation evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from datasets import Dataset

from diffusion_llm import evaluation
from diffusion_llm.evaluation import (
    GenerationEvalConfig,
    evaluate_checkpoint,
    lexical_token_f1,
    select_generation_examples,
)
from diffusion_llm.sampling import SamplerOutput


class ToyTokenizer:
    mask_token_id = 9
    pad_token_id = 0
    eos_token_id = 1
    eot_token_id = None
    chat_template = "present"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        assert add_generation_prompt
        if any(message["content"] == "long" for message in messages):
            return [2] * 10
        return [2, 3]

    def encode(self, text, add_special_tokens=True):
        return [4] * len(text.split())

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join("answer" for token_id in token_ids if token_id == 4)


class ToySampler:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer

    def sample(self, prompts, *, max_new_tokens, **kwargs):
        sequences = torch.tensor([prompt + [4] * max_new_tokens for prompt in prompts])
        return SamplerOutput(
            sequences=sequences,
            prompt_lengths=[len(prompt) for prompt in prompts],
        )


def test_lexical_token_f1() -> None:
    assert lexical_token_f1("alpha beta", "alpha gamma") == pytest.approx(0.5)
    assert lexical_token_f1("", "alpha") == 0.0


def test_select_generation_examples_preserves_chat_and_reports_skips() -> None:
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "system", "content": "Be concise"},
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Reference"},
                ],
                "source": "chat",
                "prompt": None,
                "response": None,
            },
            {
                "messages": [
                    {"role": "user", "content": "long"},
                    {"role": "assistant", "content": "Reference"},
                ],
                "source": "long",
                "prompt": None,
                "response": None,
            },
            {
                "messages": [{"role": "user", "content": "No answer"}],
                "source": "invalid",
                "prompt": None,
                "response": None,
            },
            {
                "messages": None,
                "prompt": "Fallback question",
                "response": "Fallback answer",
                "source": "prompt-response",
            },
        ]
    )

    examples, counts = select_generation_examples(
        dataset,
        ToyTokenizer(),
        num_samples=2,
        max_new_tokens=2,
        max_total_tokens=8,
        source_field="source",
    )

    assert [example.source for example in examples] == ["chat", "prompt-response"]
    assert examples[0].prompt_messages[0] == {
        "role": "system",
        "content": "Be concise",
    }
    assert counts == {
        "rows_scanned": 4,
        "skipped_unsupported": 0,
        "skipped_without_final_assistant": 1,
        "skipped_too_long": 1,
    }


def test_evaluate_checkpoint_writes_records_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "assistant", "content": "answer answer"},
                ],
                "source": "toy",
            }
            for index in range(3)
        ]
    )
    monkeypatch.setattr(evaluation, "load_dataset_source", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(evaluation, "load_tokenizer", lambda model: ToyTokenizer())
    monkeypatch.setattr(evaluation, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation, "MaskedDiffusionSampler", ToySampler)
    output = tmp_path / "heldout.jsonl"

    records_path, summary_path, summary = evaluate_checkpoint(
        GenerationEvalConfig(
            model="checkpoint",
            dataset="dataset",
            output=str(output),
            num_samples=2,
            batch_size=2,
            max_new_tokens=2,
            steps=2,
            block_size=2,
            progress=False,
        )
    )

    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert all(record["generation"] == "answer answer" for record in records)
    assert summary["metrics"]["exact_match_rate"] == 1.0
    assert saved_summary["metrics"]["mean_lexical_token_f1"] == 1.0
    assert saved_summary["by_source"]["toy"]["samples"] == 2


def test_evaluate_checkpoint_does_not_overwrite_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "heldout.jsonl"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        evaluate_checkpoint(
            GenerationEvalConfig(
                model="checkpoint",
                dataset="dataset",
                output=str(output),
            )
        )
