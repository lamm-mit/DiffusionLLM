"""Loss tests. Run with ``PYTHONPATH=src pytest tests/test_training.py``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from datasets import config as datasets_config
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments

from diffusion_llm.collator import DiffusionDataCollator
from diffusion_llm.conversion import convert_checkpoint
from diffusion_llm.loading import load_model, load_tokenizer
from diffusion_llm.sampling import MaskedDiffusionSampler
from diffusion_llm.training import (
    MDLMTrainer,
    TrainConfig,
    _apply_lora,
    _build_training_arguments,
    train,
)


class _ScalarLossModel(torch.nn.Module):
    """Tiny model whose deterministic loss makes accumulation scaling observable."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs):
        return None


class _DeterministicMDLMTrainer(MDLMTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = model.weight * inputs["scale"].float().mean()
        return (loss, None) if return_outputs else loss


def test_one_diffusion_loss_backward(
    tiny_ar_checkpoint: Path,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "diffusion"
    convert_checkpoint(str(tiny_ar_checkpoint), checkpoint, dtype="float32")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    collator = DiffusionDataCollator(tokenizer)
    batch = collator(
        [
            {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
            {"input_ids": [4, 8], "labels": [-100, 8]},
        ]
    )
    trainer = MDLMTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer"),
            report_to=[],
            use_cpu=True,
        ),
        mask_token_id=tokenizer.mask_token_id,
    )
    torch.manual_seed(0)
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_gradient_accumulation_normalizes_custom_loss(tmp_path: Path) -> None:
    model = _ScalarLossModel()
    trainer = _DeterministicMDLMTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer-accumulation"),
            gradient_accumulation_steps=4,
            report_to=[],
            use_cpu=True,
        ),
        mask_token_id=9,
    )
    trainer.current_gradient_accumulation_steps = 4

    returned_loss = trainer.training_step(
        model,
        {"scale": torch.tensor([2.0])},
        num_items_in_batch=torch.tensor(1),
    )

    assert trainer.model_accepts_loss_kwargs is False
    assert returned_loss.item() == pytest.approx(0.5)
    assert model.weight.grad is not None
    assert model.weight.grad.item() == pytest.approx(0.5)


def test_lora_wraps_diffusion_model(
    tiny_ar_checkpoint: Path,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "diffusion-lora"
    convert_checkpoint(str(tiny_ar_checkpoint), checkpoint, dtype="float32")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    wrapped = _apply_lora(
        model,
        TrainConfig(
            model=str(checkpoint),
            dataset="unused",
            output=str(tmp_path / "adapter"),
            lora=True,
        ),
    )
    trainable = [name for name, parameter in wrapped.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("lora_" in name for name in trainable)


def test_hub_training_arguments_are_forwarded(tmp_path: Path) -> None:
    config = TrainConfig(
        model="base",
        dataset="data",
        output=str(tmp_path),
        push_to_hub=True,
        hub_model_id="lamm-mit/classroom-diffusion",
        hub_private=True,
        hub_strategy="checkpoint",
    )

    arguments = _build_training_arguments(config, tmp_path, has_eval=False)

    assert arguments.push_to_hub
    assert arguments.hub_model_id == "lamm-mit/classroom-diffusion"
    assert arguments.hub_private_repo
    assert arguments.hub_strategy.value == "checkpoint"
    assert arguments.warmup_steps == pytest.approx(0.03)


def test_wandb_tracking_defaults_and_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    output = tmp_path / "descriptive-run"
    config = TrainConfig(
        model="base",
        dataset="data",
        output=str(output),
        report_to="wandb",
        wandb_entity="lamm-mit",
    )

    arguments = _build_training_arguments(config, output, has_eval=False)

    assert arguments.report_to == ["wandb"]
    assert arguments.run_name == "descriptive-run"
    assert os.environ["WANDB_PROJECT"] == "DiffusionLLM"
    assert os.environ["WANDB_ENTITY"] == "lamm-mit"

    monkeypatch.setenv("WANDB_PROJECT", "EnvironmentOverride")
    overridden = _build_training_arguments(config, output, has_eval=False)
    assert overridden.run_name == "descriptive-run"
    assert os.environ["WANDB_PROJECT"] == "EnvironmentOverride"


def test_push_to_hub_requires_model_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--hub-model-id"):
        train(
            TrainConfig(
                model="unused",
                dataset="unused",
                output=str(tmp_path),
                push_to_hub=True,
            )
        )


def test_end_to_end_one_step_training(
    tiny_ar_checkpoint: Path,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "diffusion-base"
    output = tmp_path / "trained"
    data_file = tmp_path / "text.jsonl"
    datasets_config.HF_DATASETS_CACHE = str(tmp_path / "datasets-cache")
    convert_checkpoint(str(tiny_ar_checkpoint), checkpoint, dtype="float32")
    Dataset.from_dict(
        {
            "text": [
                "hello world small model answer hello world small model answer",
                "small model answer hello world small model answer hello world",
            ]
        }
    ).to_json(data_file)

    train(
        TrainConfig(
            model=str(checkpoint),
            dataset=str(data_file),
            output=str(output),
            mode="pretrain",
            validation_fraction=0.0,
            max_length=4,
            max_steps=1,
            batch_size=1,
            save_steps=10,
            logging_steps=1,
            report_to="none",
        )
    )
    assert (output / "config.json").exists()
    assert (output / "training_config.json").exists()
    assert AutoTokenizer.from_pretrained(output).mask_token_id is not None

    trained_tokenizer = load_tokenizer(str(output))
    trained_model = load_model(
        str(output),
        dtype="float32",
        device="cpu",
    )
    generated = MaskedDiffusionSampler(trained_model, trained_tokenizer).sample(
        [trained_tokenizer.encode("hello")],
        max_new_tokens=2,
        steps=2,
        block_size=2,
    )
    assert not generated.sequences.eq(trained_tokenizer.mask_token_id).any()
