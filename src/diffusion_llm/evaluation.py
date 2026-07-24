"""Held-out generation evaluation for masked diffusion checkpoints."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from datasets import Dataset, DatasetDict

from diffusion_llm.data import (
    encode_chat_messages,
    load_dataset_source,
    messages_from_row,
)
from diffusion_llm.loading import load_model, load_tokenizer
from diffusion_llm.sampling import MaskedDiffusionSampler, decode_generations


@dataclass
class GenerationEvalConfig:
    """Serializable configuration for held-out generation evaluation."""

    model: str
    dataset: str
    output: str
    dataset_config: str | None = None
    split: str = "validation"
    num_samples: int = 32
    batch_size: int = 1
    max_total_tokens: int | None = None
    max_new_tokens: int = 64
    steps: int = 64
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 42
    progress: bool = True
    source_field: str = "source"
    summary_output: str | None = None
    overwrite: bool = False


@dataclass
class GenerationExample:
    """One normalized held-out prompt and its reference answer."""

    selection_index: int
    source: str | None
    prompt_messages: list[dict[str, str]]
    prompt_ids: list[int]
    reference: str


def _word_counts(text: str) -> Counter[str]:
    return Counter(re.findall(r"\w+", text.lower()))


def lexical_token_f1(prediction: str, reference: str) -> float:
    """Compute bag-of-words token F1 as a lightweight lexical diagnostic."""
    predicted = _word_counts(prediction)
    expected = _word_counts(reference)
    overlap = sum((predicted & expected).values())
    if not predicted or not expected or overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def select_generation_examples(
    dataset: Dataset,
    tokenizer: Any,
    *,
    num_samples: int,
    max_new_tokens: int,
    max_total_tokens: int | None,
    source_field: str,
) -> tuple[list[GenerationExample], dict[str, int]]:
    """Select usable SFT rows while retaining their full prompt conversations."""
    examples: list[GenerationExample] = []
    counts = {
        "rows_scanned": 0,
        "skipped_unsupported": 0,
        "skipped_without_final_assistant": 0,
        "skipped_too_long": 0,
    }
    for selection_index, row in enumerate(dataset):
        counts["rows_scanned"] += 1
        normalized = messages_from_row(dict(row))
        if normalized is None:
            counts["skipped_unsupported"] += 1
            continue
        if not normalized or normalized[-1]["role"] != "assistant":
            counts["skipped_without_final_assistant"] += 1
            continue

        prompt_messages = normalized[:-1]
        prompt_ids = encode_chat_messages(
            tokenizer,
            prompt_messages,
            generation_prompt=True,
        )
        if (
            max_total_tokens is not None
            and len(prompt_ids) + max_new_tokens > max_total_tokens
        ):
            counts["skipped_too_long"] += 1
            continue

        raw_source = row.get(source_field)
        examples.append(
            GenerationExample(
                selection_index=selection_index,
                source=None if raw_source is None else str(raw_source),
                prompt_messages=prompt_messages,
                prompt_ids=prompt_ids,
                reference=normalized[-1]["content"],
            )
        )
        if len(examples) == num_samples:
            break
    return examples, counts


def _validate_config(config: GenerationEvalConfig) -> tuple[Path, Path]:
    if config.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if config.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if config.max_total_tokens is not None and config.max_total_tokens <= config.max_new_tokens:
        raise ValueError("--max-total-tokens must be greater than --max-new-tokens.")

    output = Path(config.output).expanduser().resolve()
    summary = (
        Path(config.summary_output).expanduser().resolve()
        if config.summary_output
        else output.with_suffix(".summary.json")
    )
    if output == summary:
        raise ValueError("Generation and summary output paths must differ.")
    for path in (output, summary):
        if path.exists() and not config.overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")
    return output, summary


def _source_summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["source"] or "unknown"].append(record)
    return {
        source: {
            "samples": len(items),
            "nonempty_rate": mean(not item["empty"] for item in items),
            "exact_match_rate": mean(item["exact_match"] for item in items),
            "mean_lexical_token_f1": mean(item["lexical_token_f1"] for item in items),
        }
        for source, items in sorted(grouped.items())
    }


def evaluate_checkpoint(config: GenerationEvalConfig) -> tuple[Path, Path, dict[str, Any]]:
    """Generate from held-out prompts and write detailed and aggregate results."""
    output_path, summary_path = _validate_config(config)
    loaded = load_dataset_source(
        config.dataset,
        dataset_config=config.dataset_config,
        split=config.split,
    )
    if isinstance(loaded, DatasetDict):
        if config.split not in loaded:
            raise KeyError(
                f"Dataset has no {config.split!r} split. Available: {sorted(loaded)}"
            )
        dataset = loaded[config.split]
    else:
        dataset = loaded
    dataset = dataset.shuffle(seed=config.seed)

    tokenizer = load_tokenizer(config.model)
    examples, selection_counts = select_generation_examples(
        dataset,
        tokenizer,
        num_samples=config.num_samples,
        max_new_tokens=config.max_new_tokens,
        max_total_tokens=config.max_total_tokens,
        source_field=config.source_field,
    )
    if len(examples) < config.num_samples:
        raise ValueError(
            f"Only {len(examples)} usable examples were found after scanning "
            f"{selection_counts['rows_scanned']} rows; requested {config.num_samples}. "
            f"Skip counts: {selection_counts}."
        )

    model = load_model(
        config.model,
        dtype=config.dtype,
        device=config.device,
    )
    sampler = MaskedDiffusionSampler(model, tokenizer)
    torch.manual_seed(config.seed)
    records: list[dict[str, Any]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(examples), config.batch_size):
            batch = examples[start : start + config.batch_size]
            sampled = sampler.sample(
                [example.prompt_ids for example in batch],
                max_new_tokens=config.max_new_tokens,
                steps=config.steps,
                block_size=config.block_size,
                temperature=config.temperature,
                remasking=config.remasking,
                return_history=False,
                show_progress=config.progress,
            )
            generations = decode_generations(tokenizer, sampled)
            for example, generation in zip(batch, generations, strict=True):
                normalized_generation = generation.strip()
                normalized_reference = example.reference.strip()
                record = {
                    "selection_index": example.selection_index,
                    "source": example.source,
                    "prompt_messages": example.prompt_messages,
                    "reference": example.reference,
                    "generation": generation,
                    "prompt_tokens": len(example.prompt_ids),
                    "generated_tokens": len(
                        tokenizer.encode(generation, add_special_tokens=False)
                    ),
                    "empty": not normalized_generation,
                    "exact_match": normalized_generation == normalized_reference,
                    "lexical_token_f1": lexical_token_f1(
                        generation,
                        example.reference,
                    ),
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    summary: dict[str, Any] = {
        "model": config.model,
        "dataset": config.dataset,
        "dataset_config": config.dataset_config,
        "split": config.split,
        "samples": len(records),
        "seed": config.seed,
        "generation": {
            "max_new_tokens": config.max_new_tokens,
            "steps": config.steps,
            "block_size": config.block_size,
            "temperature": config.temperature,
            "remasking": config.remasking,
            "max_total_tokens": config.max_total_tokens,
        },
        "selection": selection_counts,
        "metrics": {
            "nonempty_rate": mean(not record["empty"] for record in records),
            "exact_match_rate": mean(record["exact_match"] for record in records),
            "mean_lexical_token_f1": mean(
                record["lexical_token_f1"] for record in records
            ),
            "mean_generated_tokens": mean(record["generated_tokens"] for record in records),
        },
        "by_source": _source_summaries(records),
        "metric_note": (
            "Lexical token F1 measures surface overlap only. For open-ended chat, "
            "inspect generations or use a blinded human/LLM judge."
        ),
        "config": asdict(config),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path, summary_path, summary
