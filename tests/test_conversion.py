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
    assert config.diffusion_prediction_parameterization == "same-position"
    assert config.diffusion_attention_pattern == "full-bidirectional"
    assert config.diffusion_training_objective == "legacy-mdlm"
    assert model.get_input_embeddings().num_embeddings == len(tokenizer)
    assert (output / "diffusion_metadata.json").exists()


def test_conversion_records_block_shift_architecture(
    tiny_ar_checkpoint: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "diffusion-block-shift"
    convert_checkpoint(
        str(tiny_ar_checkpoint),
        output,
        dtype="float32",
        prediction_parameterization="shifted",
        attention_pattern="block-causal",
    )

    config = AutoConfig.from_pretrained(output)
    assert config.diffusion_prediction_parameterization == "shifted"
    assert config.diffusion_attention_pattern == "block-causal"
    assert config.diffusion_training_objective == "block-hybrid"


def test_conversion_can_add_zero_initialized_time_conditioning(
    tiny_ar_checkpoint: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "diffusion-time"
    convert_checkpoint(
        str(tiny_ar_checkpoint),
        output,
        dtype="float32",
        time_conditioning="additive",
        time_embedding_dim=16,
    )

    config = AutoConfig.from_pretrained(output)
    model = AutoModelForMaskedLM.from_pretrained(output)
    assert config.diffusion_time_conditioning == "additive"
    assert config.diffusion_time_embedding_dim == 16
    assert model.model.diffusion_time_conditioner is not None
