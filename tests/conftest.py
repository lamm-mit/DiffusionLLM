"""Shared test fixtures. Run with ``PYTHONPATH=src pytest``."""

from __future__ import annotations

from pathlib import Path

import pytest
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer


@pytest.fixture
def tiny_ar_checkpoint(tmp_path: Path) -> Path:
    """Create a local random Qwen2 checkpoint without network access."""
    source = tmp_path / "ar"
    source.mkdir()
    tokens = ["<|endoftext|>", "Ġ", "h", "e", "l", "o", "w", "r", "d", "s", "m", "a", "n"]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    tokenizer = Qwen2Tokenizer(
        vocab=vocabulary,
        merges=[],
        pad_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )
    tokenizer.save_pretrained(source)

    config = Qwen2Config(
        vocab_size=len(vocabulary),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        attn_implementation="sdpa",
    )
    Qwen2ForCausalLM(config).save_pretrained(source)
    return source
