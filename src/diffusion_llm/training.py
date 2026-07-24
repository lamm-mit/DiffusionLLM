"""Masked diffusion training built on the Hugging Face Trainer.

Run ``python -m diffusion_llm train --help`` to launch training.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, set_seed

from diffusion_llm.collator import DiffusionDataCollator
from diffusion_llm.corruption import build_full_corruption, reduce_diffusion_loss
from diffusion_llm.data import load_splits, prepare_pretraining, prepare_sft
from diffusion_llm.loading import load_model, load_tokenizer
from diffusion_llm.schedule import loss_weight, mask_probability


@dataclass
class TrainConfig:
    """Serializable configuration consumed by :func:`train`."""

    model: str
    dataset: str
    output: str
    mode: str = "sft"
    dataset_config: str | None = None
    train_split: str = "train"
    eval_split: str | None = None
    validation_fraction: float = 0.02
    text_field: str = "text"
    max_length: int = 512
    append_eos: bool = True
    mask_prompt_loss: bool = True
    num_proc: int = 1
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    learning_rate: float = 1e-4
    epochs: float = 3.0
    max_steps: int = -1
    batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 2
    time_epsilon: float = 1e-3
    loss_weighting: str = "schedule"
    objective: str = "legacy-mdlm"
    time_sampling: str = "uniform"
    mask_sampling: str = "bernoulli"
    loss_normalization: str = "token"
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    report_to: str = "none"
    run_name: str | None = None
    wandb_project: str = "DiffusionLLM"
    wandb_entity: str | None = None
    resume_from_checkpoint: str | None = None
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_private: bool = False
    hub_strategy: str = "every_save"


class MDLMTrainer(Trainer):
    """Trainer implementing the continuous-time masked diffusion objective."""

    def __init__(
        self,
        *args: Any,
        mask_token_id: int,
        time_epsilon: float = 1e-3,
        loss_weighting: str = "schedule",
        objective: str = "legacy-mdlm",
        time_sampling: str = "uniform",
        mask_sampling: str = "bernoulli",
        loss_normalization: str = "token",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # This custom loss is already reduced over each microbatch and does not
        # consume ``num_items_in_batch``. Transformers 5 otherwise infers from
        # the converted model's **kwargs forward signature that the model will
        # normalize accumulated losses itself and skips Trainer's division by
        # gradient_accumulation_steps.
        self.model_accepts_loss_kwargs = False
        if not 0 < time_epsilon < 1:
            raise ValueError("time_epsilon must lie in (0, 1).")
        if loss_weighting not in {"schedule", "uniform"}:
            raise ValueError("loss_weighting must be 'schedule' or 'uniform'.")
        if objective not in {"legacy-mdlm", "mdlm-v2"}:
            raise ValueError("objective must be 'legacy-mdlm' or 'mdlm-v2'.")
        if time_sampling not in {"uniform", "stratified"}:
            raise ValueError("time_sampling must be 'uniform' or 'stratified'.")
        if mask_sampling not in {"bernoulli", "uniform-count"}:
            raise ValueError("mask_sampling must be 'bernoulli' or 'uniform-count'.")
        if loss_normalization not in {"token", "sequence"}:
            raise ValueError("loss_normalization must be 'token' or 'sequence'.")
        self.mask_token_id = mask_token_id
        self.time_epsilon = time_epsilon
        self.loss_weighting = loss_weighting
        self.objective = objective
        self.time_sampling = time_sampling
        self.mask_sampling = mask_sampling
        self.loss_normalization = loss_normalization

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ):
        if self.objective == "legacy-mdlm":
            return self._compute_legacy_loss(
                model,
                inputs,
                return_outputs=return_outputs,
            )

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask")
        corruption = build_full_corruption(
            input_ids,
            labels,
            mask_token_id=self.mask_token_id,
            time_epsilon=self.time_epsilon,
            time_sampling=self.time_sampling,
            mask_sampling=self.mask_sampling,
            loss_weighting=self.loss_weighting,
        )
        outputs = model(
            input_ids=corruption.noised_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        token_loss = F.cross_entropy(
            outputs.logits.transpose(1, 2),
            input_ids,
            reduction="none",
        )
        loss = reduce_diffusion_loss(
            token_loss,
            corruption,
            normalization=self.loss_normalization,
        )
        return (loss, outputs) if return_outputs else loss

    def _compute_legacy_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        *,
        return_outputs: bool,
    ):
        """Original loss path retained verbatim for checkpoint reproducibility."""
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask")
        maskable = labels.ne(-100)
        if not maskable.any():
            raise ValueError("Batch contains no trainable target tokens.")
        if not torch.equal(input_ids[maskable], labels[maskable]):
            raise ValueError("Target labels must equal clean input IDs at trainable positions.")

        batch_size, sequence_length = input_ids.shape
        time = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
            batch_size,
            device=input_ids.device,
        )
        probability = mask_probability(time).unsqueeze(1).expand(batch_size, sequence_length)
        masked = (torch.rand_like(probability) < probability) & maskable

        # Very short examples can otherwise produce a zero-loss row. Force one
        # valid mask in each such row while preserving the sampled time.
        missing = maskable.any(dim=1) & ~masked.any(dim=1)
        if missing.any():
            random_scores = torch.rand_like(probability).masked_fill(~maskable, -1.0)
            fallback_positions = random_scores.argmax(dim=1)
            masked[missing, fallback_positions[missing]] = True

        noised_ids = input_ids.masked_fill(masked, self.mask_token_id)
        outputs = model(
            input_ids=noised_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        token_loss = F.cross_entropy(
            outputs.logits.transpose(1, 2),
            input_ids,
            reduction="none",
        )
        if self.loss_weighting == "schedule":
            weights = loss_weight(time).unsqueeze(1)
        else:
            weights = torch.ones((batch_size, 1), device=input_ids.device)
        weighted = token_loss * weights * masked.to(token_loss.dtype)
        loss = weighted.sum() / maskable.sum().clamp_min(1)
        return (loss, outputs) if return_outputs else loss


def _apply_lora(model: torch.nn.Module, config: TrainConfig) -> torch.nn.Module:
    if not config.lora:
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def _build_training_arguments(
    config: TrainConfig,
    output_dir: Path,
    *,
    has_eval: bool,
) -> TrainingArguments:
    """Translate the stable project config into Transformers arguments."""
    report_targets = [] if config.report_to in {"", "none"} else config.report_to.split(",")
    run_name = config.run_name
    if "wandb" in report_targets:
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
        if config.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", config.wandb_entity)
        if run_name is None:
            run_name = output_dir.name
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=config.eval_steps if has_eval else None,
        prediction_loss_only=True,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=report_targets,
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=config.seed,
        data_seed=config.seed,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_private_repo=config.hub_private,
        hub_strategy=config.hub_strategy,
    )


def train(config: TrainConfig) -> Path:
    """Prepare data, train the converted model, and save a final checkpoint."""
    if config.mode not in {"pretrain", "sft"}:
        raise ValueError("mode must be 'pretrain' or 'sft'.")
    if config.bf16 and config.fp16:
        raise ValueError("Choose at most one of bf16 and fp16.")
    if config.push_to_hub and not config.hub_model_id:
        raise ValueError("--push-to-hub requires --hub-model-id.")
    if not config.push_to_hub and (config.hub_model_id or config.hub_private):
        raise ValueError("--hub-model-id and --hub-private require --push-to-hub.")
    set_seed(config.seed)

    output_dir = Path(config.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model, for_training=True)
    model.config.diffusion_method = "mdlm"
    model.config.mask_token_id = tokenizer.mask_token_id
    model.config.use_cache = False
    model = _apply_lora(model, config)

    train_dataset, eval_dataset = load_splits(
        config.dataset,
        dataset_config=config.dataset_config,
        train_split=config.train_split,
        eval_split=config.eval_split,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    preparation_args = {
        "tokenizer": tokenizer,
        "max_length": config.max_length,
        "num_proc": config.num_proc,
        "max_train_samples": config.max_train_samples,
        "max_eval_samples": config.max_eval_samples,
    }
    if config.mode == "pretrain":
        train_dataset, eval_dataset = prepare_pretraining(
            train_dataset,
            eval_dataset,
            text_field=config.text_field,
            append_eos=config.append_eos,
            **preparation_args,
        )
    else:
        train_dataset, eval_dataset = prepare_sft(
            train_dataset,
            eval_dataset,
            mask_prompt_loss=config.mask_prompt_loss,
            **preparation_args,
        )

    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    training_args = _build_training_arguments(config, output_dir, has_eval=has_eval)
    trainer = MDLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DiffusionDataCollator(tokenizer),
        processing_class=tokenizer,
        mask_token_id=tokenizer.mask_token_id,
        time_epsilon=config.time_epsilon,
        loss_weighting=config.loss_weighting,
        objective=config.objective,
        time_sampling=config.time_sampling,
        mask_sampling=config.mask_sampling,
        loss_normalization=config.loss_normalization,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    (output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )
    tokenizer.save_pretrained(output_dir)
    trainer.save_model(str(output_dir))
    return output_dir
