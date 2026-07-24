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
from diffusion_llm.tokenization import token_id_list


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
    max_nfe: int | None = None
    block_size: int = 32
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    sampling_method: str = "multinomial"
    sampling_precision: str = "float64"
    sampling_chunk_size: int = 64
    commit_policy: str = "max-prob"
    commit_schedule: str = "fixed"
    confidence_threshold: float = 0.9
    min_commit: int = 1
    max_commit: int | None = None
    uncode_base_policy: str = "max-prob"
    uncode_position_lambda: float = 1.0
    uncode_information_alpha: float = 10.0
    uncode_trivial_penalty: float = 0.35
    token_frequency_file: str | None = None
    cfg_scale: float = 0.0
    cfg_unconditional: str = "mask"
    negative_prompt: str | None = None
    remask_policy: str = "none"
    remask_rate: float = 0.05
    remask_start_fraction: float = 0.5
    max_remasks_per_step: int = 4
    max_revisions_per_token: int = 2
    remask_window: str = "current"
    remask_cooldown: int = 1
    remask_candidate_pool: int = 16
    remask_accept: str = "improve"
    remask_eos: bool = False
    min_new_tokens: int = 0
    eos_stability_steps: int = 1
    stop_on_eos: bool = True
    remasking: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 42
    progress: bool = True
    sampling_stats: bool = True
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
    sampler_totals = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(examples), config.batch_size):
            batch = examples[start : start + config.batch_size]
            negative_prompts = None
            if config.negative_prompt is not None:
                negative_ids = token_id_list(
                    tokenizer.encode(
                        config.negative_prompt,
                        add_special_tokens=True,
                    )
                )
                negative_prompts = [negative_ids.copy() for _ in batch]
            sampled = sampler.sample(
                [example.prompt_ids for example in batch],
                max_new_tokens=config.max_new_tokens,
                steps=config.steps,
                max_nfe=config.max_nfe,
                block_size=config.block_size,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                sampling_method=config.sampling_method,
                sampling_precision=config.sampling_precision,
                sampling_chunk_size=config.sampling_chunk_size,
                commit_policy=config.commit_policy,
                commit_schedule=config.commit_schedule,
                confidence_threshold=config.confidence_threshold,
                min_commit=config.min_commit,
                max_commit=config.max_commit,
                uncode_base_policy=config.uncode_base_policy,
                uncode_position_lambda=config.uncode_position_lambda,
                uncode_information_alpha=config.uncode_information_alpha,
                uncode_trivial_penalty=config.uncode_trivial_penalty,
                token_frequency_file=config.token_frequency_file,
                cfg_scale=config.cfg_scale,
                cfg_unconditional=config.cfg_unconditional,
                negative_prompts=negative_prompts,
                remask_policy=config.remask_policy,
                remask_rate=config.remask_rate,
                remask_start_fraction=config.remask_start_fraction,
                max_remasks_per_step=config.max_remasks_per_step,
                max_revisions_per_token=config.max_revisions_per_token,
                remask_window=config.remask_window,
                remask_cooldown=config.remask_cooldown,
                remask_candidate_pool=config.remask_candidate_pool,
                remask_accept=config.remask_accept,
                remask_eos=config.remask_eos,
                min_new_tokens=config.min_new_tokens,
                eos_stability_steps=config.eos_stability_steps,
                stop_on_eos=config.stop_on_eos,
                remasking=config.remasking,
                return_history=False,
                show_progress=config.progress,
            )
            if sampled.stats:
                for key, value in sampled.stats.to_dict().items():
                    if key != "committed_tokens_per_forward":
                        sampler_totals[key] += value
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
                    "batch_sampler": sampled.stats.to_dict() if sampled.stats else None,
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
            "max_nfe": config.max_nfe,
            "block_size": config.block_size,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "sampling_method": config.sampling_method,
            "sampling_precision": config.sampling_precision,
            "commit_policy": config.commit_policy,
            "commit_schedule": config.commit_schedule,
            "remask_policy": config.remask_policy,
            "cfg_scale": config.cfg_scale,
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
        "sampler_totals": dict(sampler_totals),
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
