"""Attention tests. Run with ``PYTHONPATH=src pytest tests/test_modeling.py``."""

from __future__ import annotations

import pytest
import torch

from diffusion_llm.modeling import (
    DiffusionLlamaConfig,
    DiffusionLlamaForMaskedLM,
    DiffusionQwen2Config,
    DiffusionQwen2ForMaskedLM,
    DiffusionQwen3Config,
    DiffusionQwen3ForMaskedLM,
)


def tiny_model() -> DiffusionQwen2ForMaskedLM:
    config = DiffusionQwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        attn_implementation="sdpa",
    )
    return DiffusionQwen2ForMaskedLM(config).eval()


@pytest.mark.parametrize(
    ("config_class", "model_class"),
    [
        (DiffusionLlamaConfig, DiffusionLlamaForMaskedLM),
        (DiffusionQwen2Config, DiffusionQwen2ForMaskedLM),
        (DiffusionQwen3Config, DiffusionQwen3ForMaskedLM),
    ],
)
def test_future_token_affects_past_logits(config_class, model_class) -> None:
    torch.manual_seed(0)
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        attn_implementation="sdpa",
    )
    model = model_class(config).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 3, 5]])
    with torch.no_grad():
        first_logits = model(first).logits
        second_logits = model(second).logits
    difference = (first_logits[:, 1] - second_logits[:, 1]).abs().max()
    assert difference > 1e-5


def test_right_padding_does_not_change_real_token_logits() -> None:
    torch.manual_seed(0)
    model = tiny_model()
    with torch.no_grad():
        unpadded = model(
            torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        ).logits
        padded = model(
            torch.tensor([[1, 2, 3, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
        ).logits[:, :3]
    assert torch.allclose(unpadded, padded, atol=1e-6, rtol=1e-6)
