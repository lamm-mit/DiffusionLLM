"""Tests for deterministic fixed-corruption evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from diffusion_llm.denoising_evaluation import (
    evaluate_denoising_model,
    fixed_probability_corruption,
    parse_mask_probabilities,
)


class ConstantModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(
            diffusion_prediction_parameterization="same-position",
            diffusion_attention_pattern="full-bidirectional",
            diffusion_time_conditioning="none",
        )

    def forward(self, input_ids, **kwargs):
        logits = torch.zeros((*input_ids.shape, 10), device=input_ids.device)
        logits[..., 4] = 5.0 + self.anchor
        return SimpleNamespace(logits=logits)


def test_fixed_probability_corruption_is_exact_and_reproducible() -> None:
    input_ids = torch.tensor([[2, 4, 4, 4, 4]])
    labels = torch.tensor([[-100, 4, 4, 4, 4]])
    first = fixed_probability_corruption(
        input_ids,
        labels,
        mask_token_id=9,
        probability=0.5,
        seed=7,
        prediction_parameterization="same-position",
        block_size=None,
    )
    second = fixed_probability_corruption(
        input_ids,
        labels,
        mask_token_id=9,
        probability=0.5,
        seed=7,
        prediction_parameterization="same-position",
        block_size=None,
    )

    assert torch.equal(first[0], second[0])
    assert first[1].sum().item() == 2
    assert first[2].item() == pytest.approx(0.5)
    assert first[0][first[1]].eq(9).all()


def test_denoising_metrics_are_deterministic_and_calibrated() -> None:
    batch = {
        "input_ids": torch.tensor([[2, 4, 4, 4], [2, 4, 4, 4]]),
        "labels": torch.tensor([[-100, 4, 4, 4], [-100, 4, 4, 4]]),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
    }
    dataloader = DataLoader([batch], batch_size=None)
    model = ConstantModel()
    kwargs = {
        "mask_token_id": 9,
        "mask_probabilities": "0.25,0.75",
        "block_size": 2,
        "calibration_bins": 5,
        "max_batches": None,
        "seed": 11,
    }

    first = evaluate_denoising_model(model, dataloader, **kwargs)
    second = evaluate_denoising_model(model, dataloader, **kwargs)

    assert first == second
    assert first["aggregate"]["accuracy"] == 1.0
    assert first["levels"]["0.25"]["tokens"] == 2
    assert first["levels"]["0.75"]["tokens"] == 4
    assert first["levels"]["0.25"]["ece"] > 0


def test_mask_probability_grid_validation() -> None:
    assert parse_mask_probabilities("0.1, 0.5,1") == (0.1, 0.5, 1.0)
    with pytest.raises(ValueError, match="unique"):
        parse_mask_probabilities("0.5,0.5")
