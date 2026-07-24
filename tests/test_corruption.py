"""Tests for the opt-in v2 corruption engine."""

from __future__ import annotations

import torch

from diffusion_llm.corruption import (
    CorruptionBatch,
    build_block_corruption,
    build_full_corruption,
    reduce_diffusion_loss,
    sample_diffusion_times,
)


def test_stratified_times_cover_the_unit_interval() -> None:
    generator = torch.Generator().manual_seed(7)
    epsilon = 0.01
    times = sample_diffusion_times(
        8,
        device=torch.device("cpu"),
        epsilon=epsilon,
        method="stratified",
        generator=generator,
    )
    unit_times = (times - epsilon) / (1.0 - epsilon)
    ordered = unit_times.sort().values
    lower = torch.arange(8) / 8
    upper = torch.arange(1, 9) / 8
    assert torch.all(ordered >= lower)
    assert torch.all(ordered < upper)


def test_uniform_count_masks_are_nonzero_and_confined_to_targets() -> None:
    input_ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 0]])
    labels = torch.tensor([[-100, 3, 4, 5], [-100, -100, 8, -100]])
    corruption = build_full_corruption(
        input_ids,
        labels,
        mask_token_id=9,
        time_epsilon=1e-3,
        time_sampling="stratified",
        mask_sampling="uniform-count",
        loss_weighting="schedule",
        generator=torch.Generator().manual_seed(11),
    )

    assert corruption.masked_counts.ge(1).all()
    assert not (corruption.loss_mask & ~corruption.target_mask).any()
    assert corruption.noised_ids[corruption.loss_mask].eq(9).all()
    assert torch.equal(
        corruption.noised_ids[~corruption.loss_mask],
        input_ids[~corruption.loss_mask],
    )


def test_sequence_normalization_equalizes_examples() -> None:
    target_mask = torch.tensor(
        [
            [True, False, False],
            [True, True, True],
        ]
    )
    corruption = CorruptionBatch(
        clean_ids=torch.zeros((2, 3), dtype=torch.long),
        noised_ids=torch.zeros((2, 3), dtype=torch.long),
        target_mask=target_mask,
        loss_mask=target_mask,
        token_weights=torch.ones((2, 3)),
        diffusion_time=torch.ones(2),
        mask_probability=torch.ones(2),
        target_counts=torch.tensor([1, 3]),
        masked_counts=torch.tensor([1, 3]),
    )
    token_loss = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )

    token_reduced = reduce_diffusion_loss(
        token_loss,
        corruption,
        normalization="token",
    )
    sequence_reduced = reduce_diffusion_loss(
        token_loss,
        corruption,
        normalization="sequence",
    )

    assert token_reduced.item() == 1.75
    assert sequence_reduced.item() == 2.5


def test_block_corruption_masks_future_targets_and_only_trains_active_block() -> None:
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    labels = torch.tensor([[-100, -100, 4, 5, 6, 7]])
    corruption = build_block_corruption(
        input_ids,
        labels,
        mask_token_id=9,
        block_sizes=(2,),
        time_epsilon=1e-3,
        time_sampling="uniform",
        mask_sampling="uniform-count",
        loss_weighting="schedule",
        generator=torch.Generator().manual_seed(3),
    )

    assert corruption.block_starts is not None
    assert corruption.block_ends is not None
    start = corruption.block_starts.item()
    end = corruption.block_ends.item()
    assert end - start <= 2
    assert not corruption.target_mask[:, :start].any()
    assert not corruption.target_mask[:, end:].any()
    assert corruption.noised_ids[0, end:].eq(9).all()
