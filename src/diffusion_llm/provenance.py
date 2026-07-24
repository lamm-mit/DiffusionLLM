"""Reproducible run manifests and safe checkpoint-resume validation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from diffusion_llm import __version__

TRAINING_CONFIG_NAME = "training_config.json"
RUN_MANIFEST_NAME = "run_manifest.json"

_RESUME_MUTABLE_FIELDS = {
    "allow_resume_mismatch",
    "eval_steps",
    "hub_model_id",
    "hub_private",
    "hub_strategy",
    "logging_steps",
    "output",
    "push_to_hub",
    "report_to",
    "resume_from_checkpoint",
    "run_name",
    "save_steps",
    "save_total_limit",
    "wandb_entity",
    "wandb_project",
}


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable, human-readable JSON artifact."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_training_config(output_dir: Path, config: Any) -> Path:
    """Persist the complete CLI-equivalent training configuration."""
    path = output_dir / TRAINING_CONFIG_NAME
    write_json(path, asdict(config))
    return path


def _resume_config_path(
    resume_from_checkpoint: str,
    output_dir: Path,
) -> Path | None:
    checkpoint = Path(resume_from_checkpoint).expanduser()
    candidates = [
        checkpoint / TRAINING_CONFIG_NAME,
        checkpoint.parent / TRAINING_CONFIG_NAME,
        output_dir / TRAINING_CONFIG_NAME,
    ]
    return next((path for path in candidates if path.exists()), None)


def validate_resume_configuration(
    config: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Reject material configuration drift when resuming an optimizer state."""
    if not config.resume_from_checkpoint:
        return {"requested": False, "status": "not_requested", "mismatches": {}}
    previous_path = _resume_config_path(
        config.resume_from_checkpoint,
        output_dir,
    )
    if previous_path is None:
        return {
            "requested": True,
            "status": "previous_config_unavailable",
            "mismatches": {},
        }
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    current = asdict(config)
    mismatches = {
        key: {"previous": previous[key], "current": current[key]}
        for key in sorted(previous.keys() & current.keys())
        if key not in _RESUME_MUTABLE_FIELDS and previous[key] != current[key]
    }
    if mismatches and not config.allow_resume_mismatch:
        details = ", ".join(
            f"{key}={values['previous']!r}->{values['current']!r}"
            for key, values in mismatches.items()
        )
        raise ValueError(
            "Resume configuration differs in training-critical fields: "
            f"{details}. Use --allow-resume-mismatch only if this is intentional."
        )
    return {
        "requested": True,
        "status": "mismatch_allowed" if mismatches else "validated",
        "source": str(previous_path),
        "mismatches": mismatches,
    }


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": run("branch", "--show-current"),
        "remote": run("config", "--get", "remote.origin.url"),
    }


def _config_fingerprint(config: Any) -> str:
    stable = {
        key: value
        for key, value in asdict(config).items()
        if key not in _RESUME_MUTABLE_FIELDS
    }
    encoded = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_run_manifest(
    config: Any,
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    resume_validation: dict[str, Any],
) -> dict[str, Any]:
    """Capture method, data, software, hardware, and checkpoint provenance."""
    model_config = model.config
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "status": "initialized",
        "created_at": utc_now(),
        "package": {
            "name": "classroom-diffusion-llm",
            "version": __version__,
        },
        "git": _git_state(repo_root),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
            "accelerate": _package_version("accelerate"),
            "peft": _package_version("peft"),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        },
        "method": {
            "objective": config.objective,
            "prediction_parameterization": config.prediction_parameterization,
            "attention_pattern": config.attention_pattern,
            "time_conditioning": config.time_conditioning,
            "time_sampling": config.time_sampling,
            "mask_sampling": config.mask_sampling,
            "loss_normalization": config.loss_normalization,
            "training_version": getattr(
                model_config,
                "diffusion_training_version",
                None,
            ),
        },
        "model": {
            "source": config.model,
            "model_type": model_config.model_type,
            "architectures": getattr(model_config, "architectures", None),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "vocab_size": model_config.vocab_size,
            "mask_token_id": tokenizer.mask_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "data": {
            "source": config.dataset,
            "configuration": config.dataset_config,
            "train_split": config.train_split,
            "eval_split": config.eval_split,
            "train_rows": len(train_dataset),
            "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
            "train_fingerprint": getattr(train_dataset, "_fingerprint", None),
            "eval_fingerprint": (
                getattr(eval_dataset, "_fingerprint", None)
                if eval_dataset is not None
                else None
            ),
            "max_length": config.max_length,
        },
        "optimization": {
            "per_device_batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_batch_size": (
                config.batch_size
                * config.gradient_accumulation_steps
                * int(os.environ.get("WORLD_SIZE", "1"))
            ),
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "max_steps": config.max_steps,
            "seed": config.seed,
        },
        "resume_validation": resume_validation,
        "training_config_sha256": _config_fingerprint(config),
    }


def finalize_run_manifest(
    manifest: dict[str, Any],
    *,
    status: str,
    global_step: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Return a finalized copy of a run manifest."""
    finalized = dict(manifest)
    finalized["status"] = status
    finalized["finished_at"] = utc_now()
    if global_step is not None:
        finalized["global_step"] = global_step
    if error is not None:
        finalized["error"] = error
    return finalized
