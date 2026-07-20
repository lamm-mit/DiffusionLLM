"""Noise and reveal schedules for masked diffusion.

Run ``python -m diffusion_llm generate --help`` to configure the reverse
denoising process.
"""

from __future__ import annotations

import torch


def alpha(t: torch.Tensor) -> torch.Tensor:
    """Probability that a token remains clean at diffusion time ``t``."""
    if torch.any((t < 0) | (t > 1)):
        raise ValueError("Diffusion times must lie in [0, 1].")
    return 1.0 - t


def mask_probability(t: torch.Tensor) -> torch.Tensor:
    """Probability that a token is replaced by the mask token at time ``t``."""
    return 1.0 - alpha(t)


def loss_weight(t: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Continuous-time MDLM weight ``-alpha'(t) / (1 - alpha(t))``."""
    return 1.0 / t.clamp_min(epsilon)


def reveal_counts(mask_counts: torch.Tensor, steps: int) -> torch.Tensor:
    """Allocate deterministic token reveals so every mask is filled.

    Args:
        mask_counts: Number of masks in each batch row, shape ``[batch]``.
        steps: Maximum denoising steps.

    Returns:
        Integer tensor of shape ``[batch, effective_steps]``. Each row sums to
        its original mask count.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    remaining = mask_counts.to(dtype=torch.long).clone()
    effective_steps = min(steps, int(remaining.max().item())) if remaining.numel() else 0
    if effective_steps == 0:
        return torch.zeros((remaining.numel(), 0), dtype=torch.long, device=remaining.device)

    counts = torch.zeros(
        (remaining.numel(), effective_steps),
        dtype=torch.long,
        device=remaining.device,
    )
    for step in range(effective_steps):
        steps_left = effective_steps - step
        count = torch.div(remaining + steps_left - 1, steps_left, rounding_mode="floor")
        counts[:, step] = count
        remaining -= count
    return counts
