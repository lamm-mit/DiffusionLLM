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
    )


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
