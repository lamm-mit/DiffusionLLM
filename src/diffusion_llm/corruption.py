"""Forward corruption and loss reduction for diffusion-language-model training.

The legacy trainer keeps its original inline implementation.  This module is
the explicit, testable foundation for opt-in v2 objectives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffusion_llm.schedule import loss_weight, mask_probability


@dataclass
class CorruptionBatch:
    """A clean batch paired with one sampled masked-diffusion state."""

    clean_ids: torch.Tensor
    noised_ids: torch.Tensor
    target_mask: torch.Tensor
    loss_mask: torch.Tensor
    token_weights: torch.Tensor
    diffusion_time: torch.Tensor
    mask_probability: torch.Tensor
    target_counts: torch.Tensor
    masked_counts: torch.Tensor
    original_target_mask: torch.Tensor | None = None
    block_starts: torch.Tensor | None = None
    block_ends: torch.Tensor | None = None


def _random(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator)


def sample_diffusion_times(
    batch_size: int,
    *,
    device: torch.device,
    epsilon: float,
    method: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample uniform diffusion times, optionally stratified across the batch."""
    if batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if not 0 < epsilon < 1:
        raise ValueError("time-epsilon must lie in (0, 1).")
    if method == "uniform":
        unit_time = _random((batch_size,), device=device, generator=generator)
    elif method == "stratified":
        offsets = _random((batch_size,), device=device, generator=generator)
        unit_time = (torch.arange(batch_size, device=device) + offsets) / batch_size
        permutation = torch.randperm(batch_size, device=device, generator=generator)
        unit_time = unit_time[permutation]
    else:
        raise ValueError("time-sampling must be 'uniform' or 'stratified'.")
    return epsilon + (1.0 - epsilon) * unit_time


def _bernoulli_masks(
    target_mask: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    sampled = _random(
        target_mask.shape,
        device=target_mask.device,
        generator=generator,
    )
    masked = sampled < probabilities[:, None]
    masked &= target_mask

    # Preserve the legacy guarantee that every non-empty row contributes a
    # gradient, while isolating it behind the explicit Bernoulli sampler.
    missing = target_mask.any(dim=1) & ~masked.any(dim=1)
    if missing.any():
        scores = _random(
            target_mask.shape,
            device=target_mask.device,
            generator=generator,
        ).masked_fill(~target_mask, -1.0)
        fallback = scores.argmax(dim=1)
        masked[missing, fallback[missing]] = True
    return masked


def _uniform_count_masks(
    target_mask: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a nonzero mask count uniformly for every trainable row."""
    batch_size, _ = target_mask.shape
    counts = target_mask.sum(dim=1)
    masked = torch.zeros_like(target_mask)
    sampled_counts = torch.zeros(
        (batch_size,),
        dtype=torch.long,
        device=target_mask.device,
    )
    random_scores = _random(
        target_mask.shape,
        device=target_mask.device,
        generator=generator,
    ).masked_fill(~target_mask, torch.inf)
    for row in range(batch_size):
        target_count = int(counts[row].item())
        if target_count == 0:
            continue
        sampled_count = int(
            torch.randint(
                1,
                target_count + 1,
                (1,),
                device=target_mask.device,
                generator=generator,
            ).item()
        )
        positions = random_scores[row].argsort()[:sampled_count]
        masked[row, positions] = True
        sampled_counts[row] = sampled_count
    return masked, sampled_counts


def build_full_corruption(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    mask_token_id: int,
    time_epsilon: float,
    time_sampling: str,
    mask_sampling: str,
    loss_weighting: str,
    generator: torch.Generator | None = None,
) -> CorruptionBatch:
    """Construct a full-span MDLM corruption state."""
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape.")
    target_mask = labels.ne(-100)
    if not target_mask.any():
        raise ValueError("Batch contains no trainable target tokens.")
    if not torch.equal(input_ids[target_mask], labels[target_mask]):
        raise ValueError("Target labels must equal clean input IDs at trainable positions.")
    if loss_weighting not in {"schedule", "uniform"}:
        raise ValueError("loss-weighting must be 'schedule' or 'uniform'.")

    batch_size = input_ids.shape[0]
    target_counts = target_mask.sum(dim=1)
    sampled_time = sample_diffusion_times(
        batch_size,
        device=input_ids.device,
        epsilon=time_epsilon,
        method=time_sampling,
        generator=generator,
    )

    if mask_sampling == "bernoulli":
        probabilities = mask_probability(sampled_time)
        loss_mask = _bernoulli_masks(
            target_mask,
            probabilities,
            generator=generator,
        )
        if loss_weighting == "schedule":
            row_weights = loss_weight(sampled_time)
        else:
            row_weights = torch.ones_like(sampled_time)
    elif mask_sampling == "uniform-count":
        loss_mask, sampled_counts = _uniform_count_masks(
            target_mask,
            generator=generator,
        )
        probabilities = sampled_counts / target_counts.clamp_min(1)
        sampled_time = probabilities.clamp_min(time_epsilon)
        if loss_weighting == "schedule":
            # Rao-Blackwellized form of the absorbing-mask estimator:
            # sum(masked CE * L/K) / L == mean(masked CE).
            row_weights = target_counts / sampled_counts.clamp_min(1)
        else:
            row_weights = torch.ones_like(probabilities)
    else:
        raise ValueError("mask-sampling must be 'bernoulli' or 'uniform-count'.")

    noised_ids = input_ids.masked_fill(loss_mask, mask_token_id)
    token_weights = row_weights[:, None].expand_as(input_ids)
    return CorruptionBatch(
        clean_ids=input_ids,
        noised_ids=noised_ids,
        target_mask=target_mask,
        loss_mask=loss_mask,
        token_weights=token_weights,
        diffusion_time=sampled_time,
        mask_probability=probabilities,
        target_counts=target_counts,
        masked_counts=loss_mask.sum(dim=1),
        original_target_mask=target_mask,
    )


def parse_block_sizes(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize a comma-separated or materialized block-size collection."""
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        try:
            sizes = tuple(int(piece) for piece in pieces)
        except ValueError as exc:
            raise ValueError("train-block-sizes must contain integers.") from exc
    else:
        sizes = tuple(int(size) for size in value)
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("train-block-sizes must contain positive integers.")
    return sizes


def build_block_corruption(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    mask_token_id: int,
    block_sizes: tuple[int, ...],
    time_epsilon: float,
    time_sampling: str,
    mask_sampling: str,
    loss_weighting: str,
    generator: torch.Generator | None = None,
) -> CorruptionBatch:
    """Corrupt one randomly selected target block and mask every later target."""
    original_target = labels.ne(-100)
    if not original_target.any():
        raise ValueError("Batch contains no trainable target tokens.")
    batch_size, sequence_length = input_ids.shape
    block_starts = torch.zeros(
        (batch_size,),
        dtype=torch.long,
        device=input_ids.device,
    )
    block_ends = torch.zeros_like(block_starts)
    active_target = torch.zeros_like(original_target)

    size_choices = torch.randint(
        0,
        len(block_sizes),
        (batch_size,),
        device=input_ids.device,
        generator=generator,
    )
    for row in range(batch_size):
        positions = original_target[row].nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        block_size = block_sizes[int(size_choices[row].item())]
        number_of_blocks = (positions.numel() + block_size - 1) // block_size
        block_index = int(
            torch.randint(
                0,
                number_of_blocks,
                (1,),
                device=input_ids.device,
                generator=generator,
            ).item()
        )
        first = block_index * block_size
        selected = positions[first : first + block_size]
        start = int(selected[0].item())
        end = int(selected[-1].item()) + 1
        block_starts[row] = start
        block_ends[row] = end
        active_target[row, selected] = True

    active_labels = labels.masked_fill(~active_target, -100)
    corruption = build_full_corruption(
        input_ids,
        active_labels,
        mask_token_id=mask_token_id,
        time_epsilon=time_epsilon,
        time_sampling=time_sampling,
        mask_sampling=mask_sampling,
        loss_weighting=loss_weighting,
        generator=generator,
    )
    future_target = original_target & (
        torch.arange(sequence_length, device=input_ids.device)[None, :]
        >= block_ends[:, None]
    )
    corruption.noised_ids = corruption.noised_ids.masked_fill(
        future_target,
        mask_token_id,
    )
    corruption.original_target_mask = original_target
    corruption.block_starts = block_starts
    corruption.block_ends = block_ends
    return corruption


def same_position_token_loss(
    logits: torch.Tensor,
    clean_ids: torch.Tensor,
) -> torch.Tensor:
    """Return same-position vocabulary cross entropy."""
    return torch.nn.functional.cross_entropy(
        logits.transpose(1, 2),
        clean_ids,
        reduction="none",
    )


def shifted_token_loss(
    logits: torch.Tensor,
    clean_ids: torch.Tensor,
) -> torch.Tensor:
    """Align token ``i`` with the pretrained AR logit at position ``i - 1``."""
    shifted = torch.nn.functional.cross_entropy(
        logits[:, :-1].transpose(1, 2),
        clean_ids[:, 1:],
        reduction="none",
    )
    return torch.nn.functional.pad(shifted, (1, 0))


def reduce_diffusion_loss(
    token_loss: torch.Tensor,
    corruption: CorruptionBatch,
    *,
    normalization: str,
) -> torch.Tensor:
    """Reduce weighted token losses by token or by sequence."""
    weighted = (
        token_loss
        * corruption.token_weights.to(token_loss.dtype)
        * corruption.loss_mask.to(token_loss.dtype)
    )
    if normalization == "token":
        return weighted.sum() / corruption.target_counts.sum().clamp_min(1)
    if normalization == "sequence":
        valid = corruption.target_counts.gt(0)
        row_loss = weighted.sum(dim=1) / corruption.target_counts.clamp_min(1)
        return row_loss[valid].mean()
    raise ValueError("loss-normalization must be 'token' or 'sequence'.")
