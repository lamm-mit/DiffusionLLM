"""Sampler tests. Run with ``PYTHONPATH=src pytest tests/test_sampling.py``."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from diffusion_llm.sampling import MaskedDiffusionSampler


class ToyTokenizer:
    mask_token_id = 9
    pad_token_id = 0
    eos_token_id = 1
    eot_token_id = None


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_position_embeddings=32)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 10), -20.0, device=input_ids.device)
        logits[..., 4] = 20.0
        return SimpleNamespace(logits=logits)


def test_sampler_preserves_prompt_and_resolves_masks() -> None:
    sampler = MaskedDiffusionSampler(ToyModel(), ToyTokenizer())
    output = sampler.sample(
        [[2, 3]],
        max_new_tokens=5,
        steps=3,
        block_size=5,
    )
    assert output.sequences[0, :2].tolist() == [2, 3]
    assert output.sequences[0, 2:].tolist() == [4, 4, 4, 4, 4]
    assert not output.sequences.eq(ToyTokenizer.mask_token_id).any()
