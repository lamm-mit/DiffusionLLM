"""Attention layouts used by full-span and block diffusion."""

from __future__ import annotations

import torch


def create_block_causal_mask(
    attention_mask: torch.Tensor,
    block_starts: torch.Tensor,
    block_ends: torch.Tensor,
    *,
    shifted_prediction: bool,
) -> torch.Tensor:
    """Create a prefix-causal, active-block-bidirectional SDPA mask.

    Prefix representations remain causal.  Queries responsible for predicting
    the active block can attend to the complete prefix and active block.  The
    one-position extension for shifted prediction lets the final prefix state
    predict the first token in the block.
    """
    if attention_mask.ndim != 2:
        raise ValueError("Block attention requires a 2D padding mask.")
    batch_size, sequence_length = attention_mask.shape
    if block_starts.shape != (batch_size,) or block_ends.shape != (batch_size,):
        raise ValueError("Block boundaries must have shape [batch].")

    device = attention_mask.device
    query_positions = torch.arange(sequence_length, device=device)[None, :, None]
    key_positions = torch.arange(sequence_length, device=device)[None, None, :]
    starts = block_starts[:, None, None]
    ends = block_ends[:, None, None]
    active_query_start = (starts - int(shifted_prediction)).clamp_min(0)

    prefix_queries = query_positions < active_query_start
    active_queries = (query_positions >= active_query_start) & (query_positions < ends)
    suffix_queries = query_positions >= ends

    causal_prefix = key_positions <= query_positions
    active_context = key_positions < ends
    # Suffix representations are not used by the block loss, but a causal
    # fallback avoids rows with no permitted keys and keeps the tensor valid.
    suffix_context = key_positions <= query_positions
    allowed = (
        (prefix_queries & causal_prefix)
        | (active_queries & active_context)
        | (suffix_queries & suffix_context)
    )

    valid_queries = attention_mask[:, :, None].bool()
    valid_keys = attention_mask[:, None, :].bool()
    allowed &= valid_queries & valid_keys
    return allowed[:, None, :, :]
