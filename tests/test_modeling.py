"""Attention tests. Run with ``PYTHONPATH=src pytest tests/test_modeling.py``."""

from __future__ import annotations

import pytest
import torch

from diffusion_llm.attention import create_block_causal_mask
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


def test_block_causal_mask_is_causal_in_prefix_and_bidirectional_in_block() -> None:
    padding = torch.tensor([[1, 1, 1, 1, 1, 0]])
    mask = create_block_causal_mask(
        padding,
        torch.tensor([2]),
        torch.tensor([4]),
        shifted_prediction=True,
    )[0, 0]

    assert mask[0].tolist() == [True, False, False, False, False, False]
    assert mask[1].tolist() == [True, True, True, True, False, False]
    assert mask[2].tolist() == [True, True, True, True, False, False]
    assert mask[4].tolist() == [True, True, True, True, True, False]
    assert not mask[5].any()


def test_block_logits_ignore_future_blocks_but_use_active_block() -> None:
    torch.manual_seed(0)
    model = tiny_model()
    model.config.diffusion_attention_pattern = "block-causal"
    model.config.diffusion_prediction_parameterization = "same-position"
    base = torch.tensor([[1, 2, 3, 4, 5]])
    changed_active = torch.tensor([[1, 2, 3, 8, 5]])
    changed_future = torch.tensor([[1, 2, 3, 4, 9]])
    kwargs = {
        "attention_mask": torch.ones_like(base),
        "diffusion_block_starts": torch.tensor([2]),
        "diffusion_block_ends": torch.tensor([4]),
    }
    with torch.no_grad():
        base_logits = model(base, **kwargs).logits
        active_logits = model(changed_active, **kwargs).logits
        future_logits = model(changed_future, **kwargs).logits

    assert (base_logits[:, 2] - active_logits[:, 2]).abs().max() > 1e-5
    assert torch.allclose(base_logits[:, 2], future_logits[:, 2], atol=1e-6, rtol=1e-6)


def test_additive_time_conditioning_starts_as_exact_noop() -> None:
    torch.manual_seed(0)
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
        diffusion_time_conditioning="additive",
        diffusion_time_embedding_dim=16,
    )
    model = DiffusionQwen2ForMaskedLM(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        early = model(
            tokens,
            diffusion_time=torch.tensor([0.1]),
        ).logits
        late = model(
            tokens,
            diffusion_time=torch.tensor([0.9]),
        ).logits
    assert torch.equal(early, late)

    conditioner = model.model.diffusion_time_conditioner
    assert conditioner is not None
    with torch.no_grad():
        conditioner.projection[-1].weight.normal_(std=0.01)
    with torch.no_grad():
        learned_early = model(
            tokens,
            diffusion_time=torch.tensor([0.1]),
        ).logits
        learned_late = model(
            tokens,
            diffusion_time=torch.tensor([0.9]),
        ).logits
    assert not torch.allclose(learned_early, learned_late)
