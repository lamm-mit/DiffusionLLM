"""Deterministic fixed-corruption evaluation for diffusion checkpoints."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

from diffusion_llm.collator import DiffusionDataCollator
from diffusion_llm.data import (
    load_dataset_source,
    prepare_pretraining,
    prepare_sft,
)
from diffusion_llm.loading import load_model, load_tokenizer


@dataclass
class DenoisingEvalConfig:
    """Serializable configuration for fixed-corruption denoising metrics."""

    model: str
    dataset: str
    output: str
    dataset_config: str | None = None
    split: str = "validation"
    mode: str = "sft"
    text_field: str = "text"
    max_length: int = 1024
    append_eos: bool = True
    mask_prompt_loss: bool = True
    num_samples: int = 512
    batch_size: int = 4
    max_batches: int | None = None
    mask_probabilities: str = "0.15,0.30,0.50,0.70,0.90"
    block_size: int = 32
    calibration_bins: int = 10
    num_proc: int = 1
    device: str = "auto"
    dtype: str = "auto"
    seed: int = 1729
    overwrite: bool = False


def parse_mask_probabilities(value: str | Iterable[float]) -> tuple[float, ...]:
    """Normalize and validate a fixed corruption grid."""
    if isinstance(value, str):
        try:
            probabilities = tuple(
                float(piece.strip())
                for piece in value.split(",")
                if piece.strip()
            )
        except ValueError as exc:
            raise ValueError("mask-probabilities must contain numbers.") from exc
    else:
        probabilities = tuple(float(item) for item in value)
    if not probabilities or any(not 0 < item <= 1 for item in probabilities):
        raise ValueError("mask-probabilities must lie in (0, 1].")
    if len(set(probabilities)) != len(probabilities):
        raise ValueError("mask-probabilities must be unique.")
    return probabilities


def fixed_probability_corruption(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    mask_token_id: int,
    probability: float,
    seed: int,
    prediction_parameterization: str,
    block_size: int | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Mask an exact deterministic count per row at one corruption level."""
    if not 0 < probability <= 1:
        raise ValueError("probability must lie in (0, 1].")
    target_mask = labels.ne(-100)
    if prediction_parameterization == "shifted":
        target_mask = target_mask.clone()
        target_mask[:, 0] = False
    elif prediction_parameterization != "same-position":
        raise ValueError("Unknown prediction parameterization.")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    noised_ids = input_ids.clone()
    loss_mask = torch.zeros_like(target_mask)
    diffusion_time = torch.zeros(
        input_ids.shape[0],
        dtype=torch.float32,
        device=input_ids.device,
    )
    block_starts = None
    block_ends = None
    if block_size is not None:
        if block_size < 1:
            raise ValueError("block-size must be positive.")
        block_starts = torch.zeros(
            input_ids.shape[0],
            dtype=torch.long,
            device=input_ids.device,
        )
        block_ends = torch.zeros_like(block_starts)

    for row in range(input_ids.shape[0]):
        positions = target_mask[row].nonzero(as_tuple=False).flatten()
        if not positions.numel():
            continue
        active = positions
        if block_size is not None:
            number_of_blocks = math.ceil(positions.numel() / block_size)
            block_index = int(
                torch.randint(
                    0,
                    number_of_blocks,
                    (1,),
                    generator=generator,
                ).item()
            )
            active = positions[
                block_index * block_size : (block_index + 1) * block_size
            ]
            assert block_starts is not None and block_ends is not None
            block_starts[row] = active[0]
            block_ends[row] = active[-1] + 1
            future = positions[positions >= block_ends[row]]
            noised_ids[row, future] = mask_token_id

        count = max(1, min(active.numel(), round(probability * active.numel())))
        ordering = torch.randperm(active.numel(), generator=generator)
        selected = active[ordering[:count].to(active.device)]
        loss_mask[row, selected] = True
        noised_ids[row, selected] = mask_token_id
        diffusion_time[row] = count / active.numel()
    return noised_ids, loss_mask, diffusion_time, block_starts, block_ends


def _aligned_logits(model: torch.nn.Module, logits: torch.Tensor) -> torch.Tensor:
    parameterization = getattr(
        model.config,
        "diffusion_prediction_parameterization",
        "same-position",
    )
    if parameterization == "same-position":
        return logits
    if parameterization == "shifted":
        return torch.nn.functional.pad(logits[:, :-1], (0, 0, 1, 0))
    raise ValueError(f"Unknown prediction parameterization: {parameterization}.")


@torch.inference_mode()
def evaluate_denoising_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    mask_token_id: int,
    mask_probabilities: str | Iterable[float],
    block_size: int,
    calibration_bins: int,
    max_batches: int | None,
    seed: int,
) -> dict[str, Any]:
    """Evaluate NLL, top-1 accuracy, confidence, and ECE at fixed mask levels."""
    probabilities = parse_mask_probabilities(mask_probabilities)
    if calibration_bins < 2:
        raise ValueError("calibration-bins must be at least 2.")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max-batches must be positive.")

    device = next(model.parameters()).device
    parameterization = getattr(
        model.config,
        "diffusion_prediction_parameterization",
        "same-position",
    )
    attention_pattern = getattr(
        model.config,
        "diffusion_attention_pattern",
        "full-bidirectional",
    )
    use_blocks = attention_pattern == "block-causal"
    was_training = model.training
    model.eval()
    levels: dict[str, dict[str, float | int]] = {}
    aggregate = {
        "nll_sum": 0.0,
        "correct": 0,
        "confidence_sum": 0.0,
        "tokens": 0,
    }

    for level_index, probability in enumerate(probabilities):
        nll_sum = 0.0
        correct = 0
        confidence_sum = 0.0
        token_count = 0
        bin_counts = torch.zeros(calibration_bins, dtype=torch.long)
        bin_confidence = torch.zeros(calibration_bins, dtype=torch.float64)
        bin_correct = torch.zeros(calibration_bins, dtype=torch.float64)
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            corruption = fixed_probability_corruption(
                input_ids,
                labels,
                mask_token_id=mask_token_id,
                probability=probability,
                seed=seed + level_index * 1_000_003 + batch_index,
                prediction_parameterization=parameterization,
                block_size=block_size if use_blocks else None,
            )
            noised_ids, loss_mask, diffusion_time, starts, ends = corruption
            kwargs: dict[str, Any] = {
                "input_ids": noised_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
            }
            if (
                getattr(
                    model.config,
                    "diffusion_time_conditioning",
                    "none",
                )
                == "additive"
            ):
                kwargs["diffusion_time"] = diffusion_time
            if use_blocks:
                kwargs["diffusion_block_starts"] = starts
                kwargs["diffusion_block_ends"] = ends
            logits = _aligned_logits(model, model(**kwargs).logits).float()
            active_logits = logits[loss_mask]
            active_targets = input_ids[loss_mask]
            if not active_targets.numel():
                continue
            log_probabilities = active_logits.log_softmax(dim=-1)
            losses = -log_probabilities.gather(
                -1,
                active_targets[:, None],
            ).squeeze(-1)
            confidences, predictions = log_probabilities.exp().max(dim=-1)
            correctness = predictions.eq(active_targets)
            nll_sum += losses.sum().item()
            correct += int(correctness.sum().item())
            confidence_sum += confidences.sum().item()
            token_count += active_targets.numel()

            indices = torch.clamp(
                (confidences * calibration_bins).to(torch.long),
                max=calibration_bins - 1,
            ).cpu()
            for bin_index in range(calibration_bins):
                selected = indices.eq(bin_index)
                if selected.any():
                    bin_counts[bin_index] += selected.sum()
                    bin_confidence[bin_index] += confidences[selected].sum().cpu()
                    bin_correct[bin_index] += correctness[selected].sum().cpu()

        if not token_count:
            raise ValueError("Denoising evaluation found no trainable target tokens.")
        ece = 0.0
        for bin_index in range(calibration_bins):
            count = int(bin_counts[bin_index].item())
            if count:
                mean_confidence = bin_confidence[bin_index].item() / count
                mean_accuracy = bin_correct[bin_index].item() / count
                ece += count / token_count * abs(mean_confidence - mean_accuracy)
        key = f"{probability:.2f}"
        mean_nll = nll_sum / token_count
        levels[key] = {
            "mask_probability": probability,
            "tokens": token_count,
            "nll": mean_nll,
            "perplexity": math.exp(min(mean_nll, 20)),
            "accuracy": correct / token_count,
            "mean_confidence": confidence_sum / token_count,
            "ece": ece,
        }
        aggregate["nll_sum"] += nll_sum
        aggregate["correct"] += correct
        aggregate["confidence_sum"] += confidence_sum
        aggregate["tokens"] += token_count

    if was_training:
        model.train()
    total_tokens = int(aggregate["tokens"])
    aggregate_nll = float(aggregate["nll_sum"]) / total_tokens
    return {
        "mask_probabilities": list(probabilities),
        "levels": levels,
        "aggregate": {
            "tokens": total_tokens,
            "nll": aggregate_nll,
            "perplexity": math.exp(min(aggregate_nll, 20)),
            "accuracy": int(aggregate["correct"]) / total_tokens,
            "mean_confidence": (
                float(aggregate["confidence_sum"]) / total_tokens
            ),
        },
        "prediction_parameterization": parameterization,
        "attention_pattern": attention_pattern,
        "block_size": block_size if use_blocks else None,
        "seed": seed,
    }


def evaluate_denoising_checkpoint(
    config: DenoisingEvalConfig,
) -> tuple[Path, dict[str, Any]]:
    """Load a dataset/checkpoint and write deterministic denoising metrics."""
    if config.mode not in {"pretrain", "sft"}:
        raise ValueError("mode must be 'pretrain' or 'sft'.")
    if config.num_samples < 1 or config.batch_size < 1:
        raise ValueError("num-samples and batch-size must be positive.")
    output = Path(config.output).expanduser().resolve()
    if output.exists() and not config.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it.")

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
        dataset: Dataset = loaded[config.split]
    else:
        dataset = loaded
    tokenizer = load_tokenizer(config.model)
    preparation = {
        "tokenizer": tokenizer,
        "max_length": config.max_length,
        "num_proc": config.num_proc,
        "max_train_samples": config.num_samples,
        "max_eval_samples": None,
    }
    if config.mode == "sft":
        dataset, _ = prepare_sft(
            dataset,
            None,
            mask_prompt_loss=config.mask_prompt_loss,
            **preparation,
        )
    else:
        dataset, _ = prepare_pretraining(
            dataset,
            None,
            text_field=config.text_field,
            append_eos=config.append_eos,
            **preparation,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=DiffusionDataCollator(tokenizer),
    )
    model = load_model(
        config.model,
        dtype=config.dtype,
        device=config.device,
    )
    metrics = evaluate_denoising_model(
        model,
        dataloader,
        mask_token_id=tokenizer.mask_token_id,
        mask_probabilities=config.mask_probabilities,
        block_size=config.block_size,
        calibration_bins=config.calibration_bins,
        max_batches=config.max_batches,
        seed=config.seed,
    )
    payload = {
        "model": config.model,
        "dataset": config.dataset,
        "dataset_config": config.dataset_config,
        "split": config.split,
        "metrics": metrics,
        "config": asdict(config),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output, payload
