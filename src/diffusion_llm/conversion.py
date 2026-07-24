"""Convert supported autoregressive checkpoints into diffusion checkpoints.

Run ``python -m diffusion_llm convert --help`` for the command-line interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from diffusion_llm.modeling import (
    BASE_MODEL_BY_SOURCE_TYPE,
    CONFIG_BY_SOURCE_TYPE,
    MODEL_BY_SOURCE_TYPE,
    configure_time_conditioning,
)


def parse_dtype(name: str) -> str | torch.dtype:
    """Translate a CLI dtype name into a Transformers-compatible value."""
    mapping: dict[str, str | torch.dtype] = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}. Choose from {sorted(mapping)}.") from exc


def _validate_output_directory(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty. Choose another directory or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _diffusion_config(source_config: Any, source_type: str) -> Any:
    config_dict = source_config.to_dict()
    for key in ("model_type", "architectures", "auto_map"):
        config_dict.pop(key, None)
    config_class = CONFIG_BY_SOURCE_TYPE[source_type]
    return config_class.from_dict(config_dict)


def convert_checkpoint(
    source: str,
    output: str | Path,
    *,
    mask_token: str = "<|diffusion_mask|>",
    dtype: str = "auto",
    random_init: bool = False,
    trust_remote_code: bool = False,
    overwrite: bool = False,
    prediction_parameterization: str = "same-position",
    attention_pattern: str = "full-bidirectional",
    time_conditioning: str = "none",
    time_embedding_dim: int = 256,
) -> Path:
    """Convert an AR model without duplicating the model in host memory.

    The source weights and parameter names are retained. The nested decoder and
    LM-head objects are reclassified as their bidirectional counterparts before
    saving. This is safe because each target class is a strict subclass with the
    same module layout.
    """
    output_dir = Path(output).expanduser().resolve()
    _validate_output_directory(output_dir, overwrite)
    if prediction_parameterization not in {"same-position", "shifted"}:
        raise ValueError(
            "prediction-parameterization must be 'same-position' or 'shifted'."
        )
    if attention_pattern not in {"full-bidirectional", "block-causal"}:
        raise ValueError(
            "attention-pattern must be 'full-bidirectional' or 'block-causal'."
        )
    if time_conditioning not in {"none", "additive"}:
        raise ValueError("time-conditioning must be 'none' or 'additive'.")
    if time_embedding_dim < 2:
        raise ValueError("time-embedding-dim must be at least 2.")

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    source_model = None
    if random_init:
        source_config = AutoConfig.from_pretrained(
            source,
            trust_remote_code=trust_remote_code,
        )
    else:
        source_model = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=parse_dtype(dtype),
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        source_config = source_model.config
    source_type = source_config.model_type
    if source_type not in CONFIG_BY_SOURCE_TYPE:
        supported = ", ".join(sorted(CONFIG_BY_SOURCE_TYPE))
        raise ValueError(
            f"Unsupported model_type={source_type!r}. Supported source types: {supported}."
        )

    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is None:
        raise ValueError("The source tokenizer must define at least an EOS or padding token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.pad_token

    added_tokens = tokenizer.add_special_tokens({"mask_token": mask_token})
    if source_model is not None and (
        added_tokens or len(tokenizer) != source_model.get_input_embeddings().num_embeddings
    ):
        source_model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

    target_config = _diffusion_config(source_config, source_type)
    target_config.vocab_size = len(tokenizer)
    target_config.pad_token_id = tokenizer.pad_token_id
    target_config.eos_token_id = tokenizer.eos_token_id
    target_config.bos_token_id = tokenizer.bos_token_id
    target_config.mask_token_id = tokenizer.mask_token_id
    target_config.use_cache = False
    target_config.diffusion_method = "mdlm"
    target_config.diffusion_prediction_parameterization = prediction_parameterization
    target_config.diffusion_attention_pattern = attention_pattern
    target_config.diffusion_training_objective = (
        "block-hybrid"
        if attention_pattern == "block-causal"
        else "legacy-mdlm"
        if prediction_parameterization == "same-position"
        else "mdlm-v2"
    )
    target_config.diffusion_time_conditioning = time_conditioning
    target_config.diffusion_time_embedding_dim = time_embedding_dim
    target_config.diffusion_training_version = 1
    target_config.source_model_name_or_path = source
    target_config.source_model_type = source_type
    target_config.architectures = [MODEL_BY_SOURCE_TYPE[source_type].__name__]

    target_model_class = MODEL_BY_SOURCE_TYPE[source_type]
    if random_init:
        target_model = target_model_class(target_config)
    else:
        assert source_model is not None
        source_model.__class__ = target_model_class
        source_model.model.__class__ = BASE_MODEL_BY_SOURCE_TYPE[source_type]
        source_model.config = target_config
        source_model.model.config = target_config
        for module in source_model.modules():
            if hasattr(module, "config"):
                module.config = target_config
        target_model = source_model
        configure_time_conditioning(
            target_model,
            kind=time_conditioning,
            embedding_dim=time_embedding_dim,
        )

    target_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "format": "classroom-diffusion-llm",
        "method": "masked diffusion language modeling (MDLM)",
        "source": source,
        "source_model_type": source_type,
        "mask_token": tokenizer.mask_token,
        "mask_token_id": tokenizer.mask_token_id,
        "random_init": random_init,
        "prediction_parameterization": prediction_parameterization,
        "attention_pattern": attention_pattern,
        "training_objective": target_config.diffusion_training_objective,
        "time_conditioning": time_conditioning,
        "time_embedding_dim": time_embedding_dim,
    }
    (output_dir / "diffusion_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir
