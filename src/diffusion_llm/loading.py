"""Checkpoint loading helpers used by training and inference CLIs."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from diffusion_llm.conversion import parse_dtype


def choose_device(requested: str = "auto") -> torch.device:
    """Choose CUDA, MPS, or CPU in that order unless explicitly requested."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(model_name_or_path: str):
    """Load and validate a converted checkpoint's tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
    if tokenizer.mask_token_id is None:
        raise ValueError(
            "Tokenizer has no mask token. Run `diffusion-llm convert` on the AR checkpoint first."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _adapter_base_path(model_name_or_path: str) -> str | None:
    path = Path(model_name_or_path)
    if not (path / "adapter_config.json").exists():
        return None
    from peft import PeftConfig

    return PeftConfig.from_pretrained(model_name_or_path).base_model_name_or_path


def load_model(
    model_name_or_path: str,
    *,
    dtype: str = "auto",
    device: str = "auto",
    for_training: bool = False,
):
    """Load a full diffusion checkpoint or a PEFT adapter checkpoint."""
    adapter_base = _adapter_base_path(model_name_or_path)
    checkpoint = adapter_base or model_name_or_path
    load_dtype = None if for_training and dtype == "auto" else parse_dtype(dtype)
    kwargs = {"attn_implementation": "sdpa"}
    if load_dtype is not None:
        kwargs["dtype"] = load_dtype
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, **kwargs)

    if adapter_base:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_name_or_path, is_trainable=for_training)

    if not for_training:
        target_device = choose_device(device)
        model = model.to(target_device)
        model.eval()
    model.config.use_cache = False
    return model
