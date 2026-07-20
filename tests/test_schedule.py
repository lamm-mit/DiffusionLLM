"""Schedule tests. Run with ``PYTHONPATH=src pytest tests/test_schedule.py``."""

from __future__ import annotations

import torch

from diffusion_llm.schedule import alpha, loss_weight, reveal_counts


def test_linear_schedule_endpoints() -> None:
    times = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(alpha(times), torch.tensor([1.0, 0.5, 0.0]))
    assert torch.allclose(loss_weight(times[1:]), torch.tensor([2.0, 1.0]))


def test_reveal_counts_resolve_every_mask() -> None:
    masks = torch.tensor([0, 3, 8])
    plan = reveal_counts(masks, steps=4)
    assert torch.equal(plan.sum(dim=1), masks)
    assert torch.all(plan >= 0)
