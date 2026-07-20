"""Tokenizer compatibility helpers shared by training and inference."""

from __future__ import annotations

from typing import Any


def token_id_list(encoded: Any) -> list[int]:
    """Normalize common tokenizer outputs to a flat list of token IDs."""
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    elif isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, (list, tuple))
        and len(encoded) == 1
        and isinstance(encoded[0], (list, tuple))
    ):
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)) or not all(
        isinstance(token_id, int) for token_id in encoded
    ):
        raise TypeError(
            "Tokenizer must return token IDs as a sequence, an Encoding, "
            "or an object with input_ids."
        )
    return list(encoded)
