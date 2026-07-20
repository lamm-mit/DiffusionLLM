"""Conversion tests. Run with ``PYTHONPATH=src pytest tests/test_conversion.py``."""

from __future__ import annotations

from pathlib import Path

from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from diffusion_llm.conversion import convert_checkpoint
from diffusion_llm.modeling import DiffusionQwen2ForMaskedLM


def test_conversion_preserves_model_and_adds_mask(
    tiny_ar_checkpoint: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "diffusion"
    convert_checkpoint(str(tiny_ar_checkpoint), output, dtype="float32")

    config = AutoConfig.from_pretrained(output)
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForMaskedLM.from_pretrained(output)
    assert config.model_type == "diffusion-qwen2"
    assert isinstance(model, DiffusionQwen2ForMaskedLM)
    assert tokenizer.mask_token_id == config.mask_token_id
    assert model.get_input_embeddings().num_embeddings == len(tokenizer)
    assert (output / "diffusion_metadata.json").exists()
