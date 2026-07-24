"""Dataset loading and tokenization for diffusion pretraining and SFT.

The training CLI accepts Hugging Face dataset IDs, JSON/JSONL/CSV/text files,
and directories previously written by ``datasets.save_to_disk``.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from diffusion_llm.tokenization import token_id_list


def load_dataset_source(
    source: str,
    *,
    dataset_config: str | None,
    split: str | None,
) -> Dataset | DatasetDict:
    """Load a Hub or local dataset source using the project's supported formats."""
    path = Path(source).expanduser()
    if path.exists():
        if path.is_dir() and (
            (path / "dataset_dict.json").exists() or (path / "state.json").exists()
        ):
            loaded = load_from_disk(str(path))
            if split and isinstance(loaded, DatasetDict):
                return loaded[split]
            return loaded
        suffix = path.suffix.lower()
        builders = {
            ".json": "json",
            ".jsonl": "json",
            ".csv": "csv",
            ".txt": "text",
        }
        if suffix not in builders:
            raise ValueError(f"Unsupported local dataset extension: {suffix}")
        return load_dataset(builders[suffix], data_files=str(path), split=split or "train")
    return load_dataset(source, dataset_config, split=split)


def load_splits(
    source: str,
    *,
    dataset_config: str | None = None,
    train_split: str = "train",
    eval_split: str | None = None,
    validation_fraction: float = 0.02,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None]:
    """Load train/evaluation splits, creating an evaluation split if requested."""
    train = load_dataset_source(source, dataset_config=dataset_config, split=train_split)
    if isinstance(train, DatasetDict):
        train = train[train_split]

    evaluation = None
    if eval_split:
        evaluation = load_dataset_source(
            source,
            dataset_config=dataset_config,
            split=eval_split,
        )
        if isinstance(evaluation, DatasetDict):
            evaluation = evaluation[eval_split]
    elif validation_fraction > 0 and len(train) > 1:
        split_data = train.train_test_split(test_size=validation_fraction, seed=seed)
        train, evaluation = split_data["train"], split_data["test"]
    return train, evaluation


def _limit(dataset: Dataset | None, maximum: int | None) -> Dataset | None:
    if dataset is None or maximum is None:
        return dataset
    return dataset.select(range(min(maximum, len(dataset))))


def prepare_pretraining(
    train: Dataset,
    evaluation: Dataset | None,
    *,
    tokenizer: Any,
    text_field: str,
    max_length: int,
    append_eos: bool,
    num_proc: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> tuple[Dataset, Dataset | None]:
    """Tokenize raw text and concatenate it into fixed-size training chunks."""
    workers = num_proc if num_proc > 1 else None

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        if text_field not in batch:
            raise KeyError(f"Dataset has no {text_field!r} column. Available: {sorted(batch)}")
        tokenized = tokenizer(
            batch[text_field],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if append_eos:
            eos_id = tokenizer.eos_token_id
            tokenized = [
                ids + ([] if ids and ids[-1] == eos_id else [eos_id]) for ids in tokenized
            ]
        flattened = list(chain.from_iterable(tokenized))
        usable = (len(flattened) // max_length) * max_length
        chunks = [flattened[i : i + max_length] for i in range(0, usable, max_length)]
        return {"input_ids": chunks, "labels": [chunk.copy() for chunk in chunks]}

    def transform(dataset: Dataset | None) -> Dataset | None:
        if dataset is None:
            return None
        return dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=workers,
            desc="Tokenizing pretraining text",
        )

    train = _limit(transform(train), max_train_samples)
    evaluation = _limit(transform(evaluation), max_eval_samples)
    if train is None or len(train) == 0:
        raise ValueError(
            "Tokenization produced no full sequences. Reduce --max-length or use more text."
        )
    return train, evaluation


def messages_from_row(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """Normalize a supported SFT row into chat messages when possible."""
    messages = row.get("messages")
    if messages:
        return [{"role": str(item["role"]), "content": str(item["content"])} for item in messages]

    instruction = row.get("instruction")
    output = row.get("output")
    if instruction is not None and output is not None:
        user = str(instruction)
        input_text = row.get("input")
        if input_text:
            user += f"\n\nInput:\n{input_text}"
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(output)},
        ]

    prompt = row.get("prompt")
    response = row.get("response", row.get("completion"))
    if prompt is not None and response is not None:
        return [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(response)},
        ]
    return None


def _manual_chat(messages: list[dict[str, str]], generation_prompt: bool) -> str:
    rendered = "".join(
        f"<{message['role']}>\n{message['content'].strip()}\n" for message in messages
    )
    if generation_prompt:
        rendered += "<assistant>\n"
    return rendered


def encode_chat_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    generation_prompt: bool,
) -> list[int]:
    """Encode chat messages exactly as the SFT preprocessing path does."""
    if getattr(tokenizer, "chat_template", None):
        return token_id_list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=generation_prompt,
            )
        )
    return token_id_list(
        tokenizer.encode(
            _manual_chat(messages, generation_prompt),
            add_special_tokens=True,
        )
    )


def _common_prefix_length(first: list[int], second: list[int]) -> int:
    length = 0
    for left, right in zip(first, second, strict=False):
        if left != right:
            break
        length += 1
    return length


def prepare_sft(
    train: Dataset,
    evaluation: Dataset | None,
    *,
    tokenizer: Any,
    max_length: int,
    mask_prompt_loss: bool,
    num_proc: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> tuple[Dataset, Dataset | None]:
    """Normalize common instruction formats and tokenize assistant responses."""
    workers = num_proc if num_proc > 1 else None

    def tokenize_row(row: dict[str, Any]) -> dict[str, list[int]]:
        messages = messages_from_row(row)
        if messages is None:
            text = row.get("text")
            if text is None:
                raise KeyError(
                    "SFT rows need messages, instruction/output, prompt/response, or text."
                )
            ids = token_id_list(
                tokenizer.encode(str(text), add_special_tokens=True)
            )[:max_length]
            return {"input_ids": ids, "labels": ids.copy()}

        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Each SFT conversation must end with an assistant message.")
        full_ids = encode_chat_messages(tokenizer, messages, generation_prompt=False)
        labels = full_ids.copy()
        if mask_prompt_loss:
            prompt_ids = encode_chat_messages(
                tokenizer,
                messages[:-1],
                generation_prompt=True,
            )
            prompt_length = _common_prefix_length(prompt_ids, full_ids)
            labels[:prompt_length] = [-100] * prompt_length
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
        return {"input_ids": full_ids, "labels": labels}

    def transform(dataset: Dataset | None) -> Dataset | None:
        if dataset is None:
            return None
        mapped = dataset.map(
            tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=workers,
            desc="Tokenizing supervised conversations",
        )
        return mapped.filter(
            lambda row: any(label != -100 for label in row["labels"]),
            num_proc=workers,
            desc="Removing examples without target tokens",
        )

    train = _limit(transform(train), max_train_samples)
    evaluation = _limit(transform(evaluation), max_eval_samples)
    if train is None or len(train) == 0:
        raise ValueError("SFT preprocessing produced no usable examples.")
    return train, evaluation
