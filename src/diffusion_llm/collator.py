"""Padding collator for diffusion pretraining and supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DiffusionDataCollator:
    """Right-pad token IDs while excluding padding from diffusion loss."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None
    pad_to_length: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty feature list.")
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_length is not None:
            if self.pad_to_length < max_length:
                raise ValueError(
                    "pad_to_length cannot be shorter than an input feature."
                )
            max_length = self.pad_to_length
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("The tokenizer must define pad_token_id.")

        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for feature in features:
            ids = list(feature["input_ids"])
            labels = list(feature.get("labels", ids))
            if len(ids) != len(labels):
                raise ValueError("input_ids and labels must have the same length.")
            padding = max_length - len(ids)
            input_rows.append(ids + [pad_id] * padding)
            label_rows.append(labels + [-100] * padding)
            attention_rows.append([1] * len(ids) + [0] * padding)

        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        }
