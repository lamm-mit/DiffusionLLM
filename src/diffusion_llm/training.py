"""Masked diffusion training built on the Hugging Face Trainer.

Run ``python -m diffusion_llm train --help`` to launch training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, set_seed

from diffusion_llm.collator import DiffusionDataCollator
from diffusion_llm.corruption import (
    build_block_corruption,
    build_full_corruption,
    fully_mask_targets,
    parse_block_sizes,
    progressive_corruption_from_confidence,
    reduce_diffusion_loss,
    same_position_token_loss,
    shifted_token_loss,
)
from diffusion_llm.data import load_splits, prepare_pretraining, prepare_sft
from diffusion_llm.loading import load_model, load_tokenizer
from diffusion_llm.modeling import configure_time_conditioning
from diffusion_llm.provenance import (
    RUN_MANIFEST_NAME,
    build_run_manifest,
    finalize_run_manifest,
    validate_resume_configuration,
    write_json,
    write_training_config,
)
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
    prediction_parameterization: str = "same-position"
    attention_pattern: str = "full-bidirectional"
    train_block_sizes: str = "16,32,64"
    full_mdlm_ratio: float = 0.25
    ar_loss_weight: float = 0.0
    progressive_stages: int = 8
    progressive_mask_probability: float = 1.0
    condition_dropout: float = 0.0
    condition_dropout_mode: str = "mask"
    mask_tail_augmentation: float = 0.0
    mask_tail_max_tokens: int = 64
    mask_consistency_weight: float = 0.0
    time_conditioning: str = "none"
    time_embedding_dim: int = 256
    self_conditioning_probability: float = 0.0
    draft_commit_probability: float = 0.5
    draft_loss_weight: float = 0.1
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
    allow_resume_mismatch: bool = False
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
        prediction_parameterization: str = "same-position",
        attention_pattern: str = "full-bidirectional",
        train_block_sizes: str | list[int] | tuple[int, ...] = "16,32,64",
        full_mdlm_ratio: float = 0.25,
        ar_loss_weight: float = 0.0,
        progressive_stages: int = 8,
        progressive_mask_probability: float = 1.0,
        condition_dropout: float = 0.0,
        condition_dropout_mode: str = "mask",
        pad_token_id: int | None = None,
        mask_tail_augmentation: float = 0.0,
        mask_tail_max_tokens: int = 64,
        mask_consistency_weight: float = 0.0,
        time_conditioning: str = "none",
        self_conditioning_probability: float = 0.0,
        draft_commit_probability: float = 0.5,
        draft_loss_weight: float = 0.1,
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
        if objective not in {
            "legacy-mdlm",
            "mdlm-v2",
            "block-mdlm",
            "block-hybrid",
        }:
            raise ValueError("Unknown diffusion training objective.")
        if time_sampling not in {"uniform", "stratified"}:
            raise ValueError("time_sampling must be 'uniform' or 'stratified'.")
        if mask_sampling not in {"bernoulli", "uniform-count", "progressive"}:
            raise ValueError(
                "mask_sampling must be 'bernoulli', 'uniform-count', or 'progressive'."
            )
        if loss_normalization not in {"token", "sequence"}:
            raise ValueError("loss_normalization must be 'token' or 'sequence'.")
        if prediction_parameterization not in {"same-position", "shifted"}:
            raise ValueError("Unknown prediction parameterization.")
        if attention_pattern not in {"full-bidirectional", "block-causal"}:
            raise ValueError("Unknown diffusion attention pattern.")
        if objective.startswith("block-") and attention_pattern != "block-causal":
            raise ValueError("Block objectives require attention_pattern='block-causal'.")
        if not 0 <= full_mdlm_ratio <= 1:
            raise ValueError("full_mdlm_ratio must lie in [0, 1].")
        if ar_loss_weight < 0:
            raise ValueError("ar_loss_weight must be non-negative.")
        if progressive_stages < 1:
            raise ValueError("progressive_stages must be positive.")
        for name, value in {
            "progressive_mask_probability": progressive_mask_probability,
            "condition_dropout": condition_dropout,
            "mask_tail_augmentation": mask_tail_augmentation,
            "self_conditioning_probability": self_conditioning_probability,
            "draft_commit_probability": draft_commit_probability,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1].")
        if condition_dropout_mode not in {"mask", "pad"}:
            raise ValueError("condition_dropout_mode must be 'mask' or 'pad'.")
        if condition_dropout and condition_dropout_mode == "pad" and pad_token_id is None:
            raise ValueError("pad condition dropout requires pad_token_id.")
        if mask_tail_max_tokens < 1:
            raise ValueError("mask_tail_max_tokens must be positive.")
        if mask_consistency_weight < 0:
            raise ValueError("mask_consistency_weight must be non-negative.")
        if mask_consistency_weight and not mask_tail_augmentation:
            raise ValueError(
                "mask_consistency_weight requires nonzero mask_tail_augmentation."
            )
        if time_conditioning not in {"none", "additive"}:
            raise ValueError("time_conditioning must be 'none' or 'additive'.")
        if draft_loss_weight < 0:
            raise ValueError("draft_loss_weight must be non-negative.")
        self.mask_token_id = mask_token_id
        self.time_epsilon = time_epsilon
        self.loss_weighting = loss_weighting
        self.objective = objective
        self.time_sampling = time_sampling
        self.mask_sampling = mask_sampling
        self.loss_normalization = loss_normalization
        self.prediction_parameterization = prediction_parameterization
        self.attention_pattern = attention_pattern
        self.train_block_sizes = parse_block_sizes(train_block_sizes)
        self.full_mdlm_ratio = full_mdlm_ratio
        self.ar_loss_weight = ar_loss_weight
        self.progressive_stages = progressive_stages
        self.progressive_mask_probability = progressive_mask_probability
        self.condition_dropout = condition_dropout
        self.condition_dropout_mode = condition_dropout_mode
        self.pad_token_id = pad_token_id
        self.mask_tail_augmentation = mask_tail_augmentation
        self.mask_tail_max_tokens = mask_tail_max_tokens
        self.mask_consistency_weight = mask_consistency_weight
        self.time_conditioning = time_conditioning
        self.self_conditioning_probability = self_conditioning_probability
        self.draft_commit_probability = draft_commit_probability
        self.draft_loss_weight = draft_loss_weight

    def _model_kwargs(
        self,
        corruption: Any,
        attention_mask: torch.Tensor,
        *,
        use_block: bool,
    ) -> dict[str, torch.Tensor | bool | None]:
        kwargs: dict[str, torch.Tensor | bool | None] = {
            "input_ids": corruption.noised_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if self.time_conditioning == "additive":
            kwargs["diffusion_time"] = corruption.diffusion_time
        if use_block:
            kwargs["diffusion_block_starts"] = corruption.block_starts
            kwargs["diffusion_block_ends"] = corruption.block_ends
        return kwargs

    def _align_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.prediction_parameterization == "same-position":
            return logits
        return F.pad(logits[:, :-1], (0, 0, 1, 0))

    def _drop_conditions(
        self,
        corruption: Any,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[Any, torch.Tensor]:
        if not self.condition_dropout:
            return corruption, attention_mask
        dropped_rows = (
            torch.rand(labels.shape[0], device=labels.device)
            < self.condition_dropout
        )
        condition_mask = (
            labels.eq(-100)
            & attention_mask.bool()
            & dropped_rows[:, None]
        )
        replacement = (
            self.mask_token_id
            if self.condition_dropout_mode == "mask"
            else self.pad_token_id
        )
        assert replacement is not None
        noised_ids = corruption.noised_ids.masked_fill(
            condition_mask,
            replacement,
        )
        updated_attention = attention_mask.clone()
        if self.condition_dropout_mode == "pad":
            updated_attention = updated_attention.masked_fill(condition_mask, 0)
        return replace(corruption, noised_ids=noised_ids), updated_attention

    def _augment_mask_tail(
        self,
        corruption: Any,
        attention_mask: torch.Tensor,
        original_attention_mask: torch.Tensor,
    ) -> tuple[Any, torch.Tensor, torch.Tensor]:
        tail_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        if not self.mask_tail_augmentation:
            return corruption, attention_mask, tail_mask
        for row in range(attention_mask.shape[0]):
            if torch.rand((), device=attention_mask.device) >= self.mask_tail_augmentation:
                continue
            valid_positions = original_attention_mask[row].nonzero(
                as_tuple=False
            ).flatten()
            start = (
                int(valid_positions[-1].item()) + 1
                if valid_positions.numel()
                else 0
            )
            available = attention_mask.shape[1] - start
            maximum = min(self.mask_tail_max_tokens, available)
            if maximum < 1:
                continue
            count = int(
                torch.randint(
                    1,
                    maximum + 1,
                    (1,),
                    device=attention_mask.device,
                ).item()
            )
            tail_mask[row, start : start + count] = True
        noised_ids = corruption.noised_ids.masked_fill(
            tail_mask,
            self.mask_token_id,
        )
        updated_attention = attention_mask.masked_fill(tail_mask, 1)
        return replace(corruption, noised_ids=noised_ids), updated_attention, tail_mask

    def _commit_drafts(
        self,
        corruption: Any,
        predictions: torch.Tensor,
    ) -> tuple[Any, torch.Tensor]:
        candidates = corruption.loss_mask
        draft_mask = (
            torch.rand(candidates.shape, device=candidates.device)
            < self.draft_commit_probability
        ) & candidates
        for row in range(candidates.shape[0]):
            positions = candidates[row].nonzero(as_tuple=False).flatten()
            selected = draft_mask[row].sum()
            if positions.numel() <= 1:
                draft_mask[row] = False
            elif selected == positions.numel():
                draft_mask[row, positions[-1]] = False
        noised_ids = corruption.noised_ids.clone()
        noised_ids[draft_mask] = predictions[draft_mask]
        loss_mask = corruption.loss_mask & ~draft_mask
        masked_counts = loss_mask.sum(dim=1)
        row_weights = corruption.target_counts / masked_counts.clamp_min(1)
        return (
            replace(
                corruption,
                noised_ids=noised_ids,
                loss_mask=loss_mask,
                token_weights=row_weights[:, None].expand_as(noised_ids),
                diffusion_time=(
                    masked_counts / corruption.target_counts.clamp_min(1)
                ),
                mask_probability=(
                    masked_counts / corruption.target_counts.clamp_min(1)
                ),
                masked_counts=masked_counts,
            ),
            draft_mask,
        )

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
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        original_attention_mask = attention_mask.clone()
        objective_labels = labels
        if self.prediction_parameterization == "shifted":
            objective_labels = labels.clone()
            objective_labels[:, 0] = -100

        use_block = self.objective == "block-mdlm"
        if self.objective == "block-hybrid":
            use_block = bool(
                torch.rand((), device=input_ids.device) >= self.full_mdlm_ratio
            )
        use_progressive = (
            self.mask_sampling == "progressive"
            and bool(
                torch.rand((), device=input_ids.device)
                < self.progressive_mask_probability
            )
        )
        base_mask_sampling = (
            "uniform-count"
            if self.mask_sampling == "progressive"
            else self.mask_sampling
        )
        if use_block:
            corruption = build_block_corruption(
                input_ids,
                objective_labels,
                mask_token_id=self.mask_token_id,
                block_sizes=self.train_block_sizes,
                time_epsilon=self.time_epsilon,
                time_sampling=self.time_sampling,
                mask_sampling=base_mask_sampling,
                loss_weighting=self.loss_weighting,
            )
        else:
            corruption = build_full_corruption(
                input_ids,
                objective_labels,
                mask_token_id=self.mask_token_id,
                time_epsilon=self.time_epsilon,
                time_sampling=self.time_sampling,
                mask_sampling=base_mask_sampling,
                loss_weighting=self.loss_weighting,
            )
        corruption, attention_mask = self._drop_conditions(
            corruption,
            labels,
            attention_mask,
        )
        reference_attention_mask = attention_mask.clone()
        corruption, attention_mask, tail_mask = self._augment_mask_tail(
            corruption,
            attention_mask,
            original_attention_mask,
        )

        use_self_conditioning = (
            self.self_conditioning_probability
            and bool(
                torch.rand((), device=input_ids.device)
                < self.self_conditioning_probability
            )
        )
        proposal_logits: torch.Tensor | None = None
        if use_progressive:
            proposal_state = fully_mask_targets(
                corruption,
                mask_token_id=self.mask_token_id,
            )
            with torch.no_grad():
                proposal_logits = model(
                    **self._model_kwargs(
                        proposal_state,
                        attention_mask,
                        use_block=use_block,
                    )
                ).logits
            aligned_proposal = self._align_logits(proposal_logits)
            confidence = aligned_proposal.float().log_softmax(dim=-1).gather(
                -1,
                input_ids.unsqueeze(-1),
            ).squeeze(-1).exp()
            corruption = progressive_corruption_from_confidence(
                corruption,
                confidence,
                mask_token_id=self.mask_token_id,
                stages=self.progressive_stages,
            )

        draft_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if use_self_conditioning:
            if proposal_logits is None:
                with torch.no_grad():
                    proposal_logits = model(
                        **self._model_kwargs(
                            corruption,
                            attention_mask,
                            use_block=use_block,
                        )
                    ).logits
            predictions = self._align_logits(proposal_logits).argmax(dim=-1)
            corruption, draft_mask = self._commit_drafts(
                corruption,
                predictions,
            )

        outputs = model(
            **self._model_kwargs(
                corruption,
                attention_mask,
                use_block=use_block,
            )
        )
        if self.prediction_parameterization == "shifted":
            token_loss = shifted_token_loss(outputs.logits, input_ids)
        else:
            token_loss = same_position_token_loss(outputs.logits, input_ids)
        loss = reduce_diffusion_loss(
            token_loss,
            corruption,
            normalization=self.loss_normalization,
        )
        if use_block and self.ar_loss_weight and corruption.original_target_mask is not None:
            positions = torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
            )[None, :]
            ar_mask = corruption.original_target_mask & positions.gt(0)
            assert corruption.block_starts is not None
            ar_mask &= positions < corruption.block_starts[:, None]
            if ar_mask.any():
                ar_token_loss = shifted_token_loss(outputs.logits, input_ids)
                ar_loss = (ar_token_loss * ar_mask).sum() / ar_mask.sum()
                loss = loss + self.ar_loss_weight * ar_loss
        if self.draft_loss_weight and draft_mask.any():
            draft_loss = (token_loss * draft_mask).sum() / draft_mask.sum()
            loss = loss + self.draft_loss_weight * draft_loss
        if self.mask_consistency_weight and tail_mask.any():
            reference_ids = corruption.noised_ids.clone()
            reference_ids[tail_mask] = input_ids[tail_mask]
            reference_state = replace(corruption, noised_ids=reference_ids)
            with torch.no_grad():
                reference_logits = model(
                    **self._model_kwargs(
                        reference_state,
                        reference_attention_mask,
                        use_block=use_block,
                    )
                ).logits
            primary = self._align_logits(outputs.logits).float()
            reference = self._align_logits(reference_logits).float()
            consistency_tokens = F.kl_div(
                primary.log_softmax(dim=-1),
                reference.softmax(dim=-1),
                reduction="none",
            ).sum(dim=-1)
            consistency_loss = (
                consistency_tokens * corruption.loss_mask
            ).sum() / corruption.loss_mask.sum().clamp_min(1)
            loss = loss + self.mask_consistency_weight * consistency_loss
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
    resume_validation = validate_resume_configuration(config, output_dir)
    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model, for_training=True)
    model.config.diffusion_method = "mdlm"
    model.config.mask_token_id = tokenizer.mask_token_id
    model.config.use_cache = False
    model.config.diffusion_prediction_parameterization = (
        config.prediction_parameterization
    )
    model.config.diffusion_attention_pattern = config.attention_pattern
    model.config.diffusion_time_conditioning = config.time_conditioning
    model.config.diffusion_time_embedding_dim = config.time_embedding_dim
    advanced_training = (
        config.mask_sampling == "progressive"
        or config.condition_dropout > 0
        or config.mask_tail_augmentation > 0
        or config.mask_consistency_weight > 0
        or config.time_conditioning != "none"
        or config.self_conditioning_probability > 0
    )
    model.config.diffusion_training_version = (
        1
        if config.objective == "legacy-mdlm" and not advanced_training
        else 3 if advanced_training else 2
    )
    if hasattr(model, "model"):
        model.model.config = model.config
    configure_time_conditioning(
        model,
        kind=config.time_conditioning,
        embedding_dim=config.time_embedding_dim,
    )
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
    fixed_canvas = (
        config.max_length
        if config.mask_tail_augmentation or config.mask_consistency_weight
        else None
    )
    trainer = MDLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DiffusionDataCollator(
            tokenizer,
            pad_to_length=fixed_canvas,
        ),
        processing_class=tokenizer,
        mask_token_id=tokenizer.mask_token_id,
        time_epsilon=config.time_epsilon,
        loss_weighting=config.loss_weighting,
        objective=config.objective,
        time_sampling=config.time_sampling,
        mask_sampling=config.mask_sampling,
        loss_normalization=config.loss_normalization,
        prediction_parameterization=config.prediction_parameterization,
        attention_pattern=config.attention_pattern,
        train_block_sizes=config.train_block_sizes,
        full_mdlm_ratio=config.full_mdlm_ratio,
        ar_loss_weight=config.ar_loss_weight,
        progressive_stages=config.progressive_stages,
        progressive_mask_probability=config.progressive_mask_probability,
        condition_dropout=config.condition_dropout,
        condition_dropout_mode=config.condition_dropout_mode,
        pad_token_id=tokenizer.pad_token_id,
        mask_tail_augmentation=config.mask_tail_augmentation,
        mask_tail_max_tokens=config.mask_tail_max_tokens,
        mask_consistency_weight=config.mask_consistency_weight,
        time_conditioning=config.time_conditioning,
        self_conditioning_probability=config.self_conditioning_probability,
        draft_commit_probability=config.draft_commit_probability,
        draft_loss_weight=config.draft_loss_weight,
    )
    manifest = build_run_manifest(
        config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        resume_validation=resume_validation,
    )
    if trainer.is_world_process_zero():
        write_training_config(output_dir, config)
        write_json(output_dir / RUN_MANIFEST_NAME, manifest)
    try:
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    except BaseException as error:
        if trainer.is_world_process_zero():
            write_json(
                output_dir / RUN_MANIFEST_NAME,
                finalize_run_manifest(
                    manifest,
                    status="failed",
                    global_step=trainer.state.global_step,
                    error=f"{type(error).__name__}: {error}",
                ),
            )
        raise
    tokenizer.save_pretrained(output_dir)
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        write_json(
            output_dir / RUN_MANIFEST_NAME,
            finalize_run_manifest(
                manifest,
                status="completed",
                global_step=trainer.state.global_step,
            ),
        )
    return output_dir
