"""Iterative masked-diffusion sampling and prompt formatting.

Run ``python -m diffusion_llm generate --help`` for one-shot inference or
``python -m diffusion_llm chat --help`` for an interactive session.
"""

from __future__ import annotations

import json
import math
import string
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffusion_llm.tokenization import token_id_list

_COMMIT_POLICIES = {
    "max_prob",
    "margin",
    "entropy",
    "random",
    "left_to_right",
    "uncode",
}
_REMASK_POLICIES = {"none", "confidence", "rescore", "random"}


@dataclass
class TrajectoryEvent:
    """One state-changing event in a denoising trajectory."""

    kind: str
    iteration: int
    forward_evaluations: int
    block_index: int | None
    committed: int = 0
    remasked: int = 0
    revised: int = 0
    masked_remaining: int = 0


@dataclass
class SamplerStats:
    """Machine-readable accounting for one sampler invocation."""

    requested_steps: int
    planned_iterations: int
    forward_evaluations: int = 0
    iterations: int = 0
    tokens_committed: int = 0
    tokens_remasked: int = 0
    tokens_revised: int = 0
    sequences_finished_by_eos: int = 0
    elapsed_seconds: float = 0.0

    @property
    def committed_tokens_per_forward(self) -> float:
        if self.forward_evaluations == 0:
            return 0.0
        return self.tokens_committed / self.forward_evaluations

    def to_dict(self) -> dict[str, int | float]:
        payload = asdict(self)
        payload["committed_tokens_per_forward"] = self.committed_tokens_per_forward
        return payload


@dataclass
class SamplerOutput:
    """Token sequences, denoising history, trajectory events, and sampler statistics."""

    sequences: torch.Tensor
    prompt_lengths: list[int]
    histories: list[torch.Tensor] | None = None
    events: list[TrajectoryEvent] | None = None
    stats: SamplerStats | None = None


@dataclass
class _PredictionBatch:
    """Predictions and confidence diagnostics aligned with the token canvas."""

    token_ids: torch.Tensor
    candidate_probabilities: torch.Tensor
    scores: torch.Tensor


def encode_prompt(
    tokenizer: Any,
    prompt: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    chat_template: bool = False,
) -> list[int]:
    """Encode either a raw prompt or a chat conversation."""
    if chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("This tokenizer has no chat template; use a raw prompt.")
        messages = messages or [{"role": "user", "content": prompt or ""}]
        return token_id_list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return token_id_list(tokenizer.encode(prompt or "", add_special_tokens=True))


def _normalize_choice(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Apply top-k and nucleus filtering to already temperature-scaled logits."""
    filtered = logits
    vocabulary_size = filtered.shape[-1]
    if top_k > 0 and top_k < vocabulary_size:
        cutoff = filtered.topk(top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < cutoff, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = filtered.sort(dim=-1, descending=True)
        sorted_probabilities = F.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities > top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        restored = torch.full_like(sorted_logits, -torch.inf)
        filtered = restored.scatter(-1, sorted_indices, sorted_logits)
    return filtered


def _categorical_sample(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    sampling_method: str,
    sampling_precision: str,
) -> torch.Tensor:
    """Sample categorical predictions without the truncated float32 Gumbel shortcut."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    dtype = torch.float64 if sampling_precision == "float64" else torch.float32
    scaled = logits.to(dtype=dtype) / temperature
    scaled = _filter_logits(scaled, top_k=top_k, top_p=top_p)
    if sampling_method == "multinomial":
        probabilities = F.softmax(scaled, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    uniform = torch.rand(scaled.shape, dtype=dtype, device=scaled.device)
    epsilon = torch.finfo(dtype).eps
    uniform.clamp_(min=epsilon, max=1.0 - epsilon)
    gumbel = -torch.log(-torch.log(uniform))
    return (scaled + gumbel).argmax(dim=-1)


def _raw_scores(
    logits: torch.Tensor,
    predicted: torch.Tensor,
    *,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate probabilities and a larger-is-better commitment score."""
    probabilities = F.softmax(logits.float(), dim=-1)
    candidate_probabilities = probabilities.gather(-1, predicted[:, None]).squeeze(-1)
    if policy == "max_prob":
        return candidate_probabilities, candidate_probabilities
    if policy == "margin":
        top_probabilities, top_ids = probabilities.topk(2, dim=-1)
        best_other = torch.where(
            predicted.eq(top_ids[:, 0]),
            top_probabilities[:, 1],
            top_probabilities[:, 0],
        )
        return candidate_probabilities, candidate_probabilities - best_other
    if policy == "entropy":
        log_probabilities = torch.log(probabilities.clamp_min(torch.finfo(torch.float32).tiny))
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        normalized = entropy / math.log(max(2, logits.shape[-1]))
        return candidate_probabilities, 1.0 - normalized
    if policy == "random":
        return candidate_probabilities, torch.rand_like(candidate_probabilities)
    return candidate_probabilities, candidate_probabilities


def _selected_probabilities(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=logits.device)
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    return log_probabilities.gather(-1, token_ids[:, None]).squeeze(-1).exp()


class MaskedDiffusionSampler:
    """Generate and revise token blocks with configurable masked-diffusion policies."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self._decoded_token_cache: dict[int, str] = {}
        self._frequency_cache: dict[tuple[str, int], torch.Tensor] = {}

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def prediction_parameterization(self) -> str:
        return getattr(
            self.model.config,
            "diffusion_prediction_parameterization",
            "same-position",
        )

    def _align_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.prediction_parameterization == "same-position":
            return logits
        if self.prediction_parameterization != "shifted":
            raise ValueError(
                f"Unknown prediction parameterization: "
                f"{self.prediction_parameterization}."
            )
        aligned = torch.empty_like(logits)
        aligned[:, 0] = logits[:, 0]
        aligned[:, 1:] = logits[:, :-1]
        return aligned

    def _stop_ids(self) -> set[int]:
        return {
            int(token_id)
            for token_id in (
                self.tokenizer.eos_token_id,
                getattr(self.tokenizer, "eot_token_id", None),
            )
            if token_id is not None
        }

    def _token_piece(self, token_id: int) -> str:
        if token_id not in self._decoded_token_cache:
            decoded = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if not decoded:
                converted = self.tokenizer.convert_ids_to_tokens(token_id)
                decoded = "" if converted is None else str(converted)
            self._decoded_token_cache[token_id] = decoded
        return self._decoded_token_cache[token_id]

    def _frequency_tensor(self, path: str, vocabulary_size: int) -> torch.Tensor:
        cache_key = (str(Path(path).expanduser().resolve()), vocabulary_size)
        if cache_key in self._frequency_cache:
            return self._frequency_cache[cache_key].to(self.device)
        raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if isinstance(raw, list):
            values = {index: float(value) for index, value in enumerate(raw)}
        elif isinstance(raw, dict):
            values = {int(index): float(value) for index, value in raw.items()}
        else:
            raise ValueError("Token-frequency JSON must be an object or a list.")
        if not values or any(value < 0 for value in values.values()):
            raise ValueError("Token frequencies must contain non-negative values.")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("Token frequencies must have a positive total.")
        floor = 1.0 / (total * max(2, vocabulary_size))
        frequencies = torch.full((vocabulary_size,), floor, dtype=torch.float64)
        for token_id, value in values.items():
            if 0 <= token_id < vocabulary_size:
                frequencies[token_id] = max(floor, value / total)
        frequencies /= frequencies.sum()
        self._frequency_cache[cache_key] = frequencies
        return frequencies.to(self.device)

    def _information_weights(
        self,
        predicted: torch.Tensor,
        *,
        vocabulary_size: int,
        frequency_file: str | None,
        alpha: float,
        trivial_penalty: float,
    ) -> torch.Tensor:
        if frequency_file:
            frequencies = self._frequency_tensor(frequency_file, vocabulary_size)
            minimum = torch.finfo(torch.float64).tiny
            information = -torch.log(frequencies[predicted].clamp_min(minimum))
            return information.clamp(max=alpha).div(alpha).to(dtype=torch.float32)
        weights = torch.ones(predicted.shape, dtype=torch.float32, device=predicted.device)
        punctuation = set(string.punctuation)
        for index, token_id in enumerate(predicted.detach().cpu().tolist()):
            piece = self._token_piece(int(token_id))
            stripped = piece.strip()
            if not stripped or all(character in punctuation for character in stripped):
                weights[index] = trivial_penalty
        return weights

    def _prediction_batch(
        self,
        logits: torch.Tensor,
        active: torch.Tensor,
        *,
        prompt_lengths: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        sampling_method: str,
        sampling_precision: str,
        commit_policy: str,
        uncode_base_policy: str,
        uncode_position_lambda: float,
        uncode_information_alpha: float,
        uncode_trivial_penalty: float,
        token_frequency_file: str | None,
        sampling_chunk_size: int,
    ) -> _PredictionBatch:
        token_ids = torch.zeros(active.shape, dtype=torch.long, device=logits.device)
        candidate_probabilities = torch.zeros(
            active.shape,
            dtype=torch.float32,
            device=logits.device,
        )
        scores = torch.full(
            active.shape,
            -torch.inf,
            dtype=torch.float32,
            device=logits.device,
        )
        positions = active.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return _PredictionBatch(token_ids, candidate_probabilities, scores)

        flat_logits = logits[active]
        base_policy = uncode_base_policy if commit_policy == "uncode" else commit_policy
        predicted_chunks: list[torch.Tensor] = []
        probability_chunks: list[torch.Tensor] = []
        score_chunks: list[torch.Tensor] = []
        for start in range(0, flat_logits.shape[0], sampling_chunk_size):
            chunk = flat_logits[start : start + sampling_chunk_size]
            predicted = _categorical_sample(
                chunk,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sampling_method=sampling_method,
                sampling_precision=sampling_precision,
            )
            probabilities, chunk_scores = _raw_scores(
                chunk,
                predicted,
                policy=base_policy,
            )
            predicted_chunks.append(predicted)
            probability_chunks.append(probabilities)
            score_chunks.append(chunk_scores)

        flat_predictions = torch.cat(predicted_chunks)
        flat_probabilities = torch.cat(probability_chunks)
        flat_scores = torch.cat(score_chunks)
        relative_positions = torch.tensor(
            [
                (int(position) - prompt_lengths[int(row)]) / max(1, max_new_tokens - 1)
                for row, position in positions.detach().cpu().tolist()
            ],
            dtype=torch.float32,
            device=logits.device,
        )
        if commit_policy == "left_to_right":
            flat_scores = 1.0 - relative_positions
        elif commit_policy == "uncode":
            position_weights = torch.exp(-uncode_position_lambda * relative_positions)
            information_weights = self._information_weights(
                flat_predictions,
                vocabulary_size=logits.shape[-1],
                frequency_file=token_frequency_file,
                alpha=uncode_information_alpha,
                trivial_penalty=uncode_trivial_penalty,
            )
            flat_scores = flat_scores * position_weights * information_weights

        token_ids[active] = flat_predictions
        candidate_probabilities[active] = flat_probabilities
        scores[active] = flat_scores
        return _PredictionBatch(token_ids, candidate_probabilities, scores)

    def _unconditional_inputs(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
        *,
        mask_id: int,
        pad_id: int,
        cfg_unconditional: str,
        negative_prompts: list[list[int]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unconditional = tokens.clone()
        unconditional_attention = attention_mask.clone()
        replacement = mask_id if cfg_unconditional == "mask" else pad_id
        for row, prompt_length in enumerate(prompt_lengths):
            unconditional[row, :prompt_length] = replacement
            if cfg_unconditional == "pad":
                unconditional_attention[row, :prompt_length] = 0
            if negative_prompts:
                negative = negative_prompts[row][-prompt_length:]
                if negative:
                    start = prompt_length - len(negative)
                    unconditional[row, start:prompt_length] = torch.tensor(
                        negative,
                        dtype=torch.long,
                        device=tokens.device,
                    )
                    unconditional_attention[row, start:prompt_length] = 1
        return unconditional, unconditional_attention

    def _conditional_forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        stats: SamplerStats,
        progress: Any,
        max_nfe: int | None,
        block_starts: torch.Tensor | None = None,
        block_ends: torch.Tensor | None = None,
        diffusion_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if max_nfe is not None and stats.forward_evaluations >= max_nfe:
            raise RuntimeError("The maximum forward-evaluation budget was exhausted.")
        model_kwargs: dict[str, Any] = {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if (
            getattr(
                self.model.config,
                "diffusion_time_conditioning",
                "none",
            )
            == "additive"
        ):
            if diffusion_time is None:
                active = attention_mask.bool()
                if block_starts is not None and block_ends is not None:
                    positions = torch.arange(
                        tokens.shape[1],
                        device=tokens.device,
                    )[None, :]
                    active &= (
                        positions >= block_starts[:, None]
                    ) & (
                        positions < block_ends[:, None]
                    )
                diffusion_time = (
                    tokens.eq(self.tokenizer.mask_token_id)
                    .logical_and(active)
                    .sum(dim=1)
                    / active.sum(dim=1).clamp_min(1)
                )
            model_kwargs["diffusion_time"] = diffusion_time
        if (
            getattr(
                self.model.config,
                "diffusion_attention_pattern",
                "full-bidirectional",
            )
            == "block-causal"
        ):
            model_kwargs["diffusion_block_starts"] = block_starts
            model_kwargs["diffusion_block_ends"] = block_ends
        output = self.model(**model_kwargs).logits
        stats.forward_evaluations += 1
        progress.update(1)
        return output

    def _guided_forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: list[int],
        *,
        mask_id: int,
        pad_id: int,
        cfg_scale: float,
        cfg_unconditional: str,
        negative_prompts: list[list[int]] | None,
        stats: SamplerStats,
        progress: Any,
        max_nfe: int | None,
        block_starts: torch.Tensor | None = None,
        block_ends: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diffusion_time = None
        if (
            getattr(
                self.model.config,
                "diffusion_time_conditioning",
                "none",
            )
            == "additive"
        ):
            active = attention_mask.bool()
            if block_starts is not None and block_ends is not None:
                positions = torch.arange(
                    tokens.shape[1],
                    device=tokens.device,
                )[None, :]
                active &= (
                    positions >= block_starts[:, None]
                ) & (
                    positions < block_ends[:, None]
                )
            diffusion_time = (
                tokens.eq(mask_id).logical_and(active).sum(dim=1)
                / active.sum(dim=1).clamp_min(1)
            )
        conditional = self._conditional_forward(
            tokens,
            attention_mask,
            stats=stats,
            progress=progress,
            max_nfe=max_nfe,
            block_starts=block_starts,
            block_ends=block_ends,
            diffusion_time=diffusion_time,
        )
        if cfg_scale == 0:
            return self._align_logits(conditional)
        unconditional_inputs, unconditional_attention = self._unconditional_inputs(
            tokens,
            attention_mask,
            prompt_lengths,
            mask_id=mask_id,
            pad_id=pad_id,
            cfg_unconditional=cfg_unconditional,
            negative_prompts=negative_prompts,
        )
        unconditional = self._conditional_forward(
            unconditional_inputs,
            unconditional_attention,
            stats=stats,
            progress=progress,
            max_nfe=max_nfe,
            block_starts=block_starts,
            block_ends=block_ends,
            diffusion_time=diffusion_time,
        )
        guided = conditional.float() + cfg_scale * (
            conditional.float() - unconditional.float()
        )
        return self._align_logits(guided)

    def _remask_scope(
        self,
        response_positions: torch.Tensor,
        *,
        block_region: torch.Tensor,
        block_index: int,
        block_size: int,
        prompt_lengths: list[int],
        remask_window: str,
    ) -> torch.Tensor:
        if remask_window == "current":
            return block_region.clone()
        scope = torch.zeros_like(response_positions)
        for row, prompt_length in enumerate(prompt_lengths):
            if remask_window == "previous":
                start = prompt_length + max(0, block_index - 1) * block_size
            else:
                start = prompt_length
            stop = prompt_length + (block_index + 1) * block_size
            scope[row, start:stop] = True
        return scope & response_positions

    def _select_remasks(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        scope: torch.Tensor,
        *,
        commit_confidence: torch.Tensor,
        revision_counts: torch.Tensor,
        last_commit_iteration: torch.Tensor,
        global_iteration: int,
        mask_id: int,
        pad_id: int,
        stop_ids: set[int],
        remask_policy: str,
        remask_rate: float,
        max_remasks_per_step: int,
        max_revisions_per_token: int,
        remask_cooldown: int,
        remask_candidate_pool: int,
        remask_eos: bool,
        stats: SamplerStats,
        progress: Any,
        max_nfe: int | None,
        reserved_nfe: int,
        block_starts: torch.Tensor | None,
        block_ends: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = scope & tokens.ne(mask_id)
        # Many chat tokenizers deliberately use EOS as their padding token.
        # Keep ordinary padding out of the candidate set, but permit the
        # shared pad/EOS ID when the caller explicitly enables EOS revision.
        if pad_id not in stop_ids or not remask_eos:
            candidates &= tokens.ne(pad_id)
        candidates &= revision_counts.lt(max_revisions_per_token)
        candidates &= global_iteration - last_commit_iteration >= remask_cooldown
        candidates &= torch.isfinite(commit_confidence)
        if not remask_eos:
            for stop_id in stop_ids:
                candidates &= tokens.ne(stop_id)

        selected = torch.zeros_like(candidates)
        preselected = torch.zeros_like(candidates)
        desired_counts = [0] * tokens.shape[0]
        for row in range(tokens.shape[0]):
            row_positions = candidates[row].nonzero(as_tuple=False).flatten()
            if row_positions.numel() == 0:
                continue
            count = min(
                max_remasks_per_step,
                max(1, math.ceil(row_positions.numel() * remask_rate)),
                row_positions.numel(),
            )
            desired_counts[row] = count
            if remask_policy == "random":
                ordering = torch.rand(
                    row_positions.shape,
                    device=tokens.device,
                ).argsort()
            else:
                ordering = commit_confidence[row, row_positions].argsort()
            if remask_policy == "rescore":
                pool_size = min(
                    row_positions.numel(),
                    max(count, remask_candidate_pool),
                )
                preselected[row, row_positions[ordering[:pool_size]]] = True
            else:
                selected[row, row_positions[ordering[:count]]] = True

        if remask_policy != "rescore" or not preselected.any():
            old_tokens = tokens.masked_fill(~selected, -1)
            return selected, old_tokens

        can_probe = max_nfe is None or (
            stats.forward_evaluations + reserved_nfe + 1 <= max_nfe
        )
        if not can_probe:
            for row in range(tokens.shape[0]):
                row_positions = preselected[row].nonzero(as_tuple=False).flatten()
                if row_positions.numel():
                    count = min(desired_counts[row], row_positions.numel())
                    ordering = commit_confidence[row, row_positions].argsort()
                    selected[row, row_positions[ordering[:count]]] = True
            old_tokens = tokens.masked_fill(~selected, -1)
            return selected, old_tokens

        old_probe_tokens = tokens[preselected].clone()
        probe_inputs = tokens.masked_fill(preselected, mask_id)
        probe_logits = self._conditional_forward(
            probe_inputs,
            attention_mask,
            stats=stats,
            progress=progress,
            max_nfe=max_nfe,
            block_starts=block_starts,
            block_ends=block_ends,
        )
        probe_logits = self._align_logits(probe_logits)
        old_probabilities = _selected_probabilities(
            probe_logits[preselected],
            old_probe_tokens,
        )
        rescored = torch.full_like(commit_confidence, torch.inf)
        rescored[preselected] = old_probabilities
        for row in range(tokens.shape[0]):
            row_positions = preselected[row].nonzero(as_tuple=False).flatten()
            if row_positions.numel() == 0:
                continue
            count = min(desired_counts[row], row_positions.numel())
            ordering = rescored[row, row_positions].argsort()
            selected[row, row_positions[ordering[:count]]] = True
        old_tokens = tokens.masked_fill(~selected, -1)
        return selected, old_tokens

    @staticmethod
    def _iteration_budgets(
        *,
        steps: int,
        number_of_blocks: int,
        base_forward_cost: int,
        max_nfe: int | None,
    ) -> list[int]:
        per_block = max(1, math.ceil(steps / number_of_blocks))
        budgets = [per_block] * number_of_blocks
        if max_nfe is None:
            return budgets
        available = max_nfe // base_forward_cost
        if available < number_of_blocks:
            raise ValueError(
                f"--max-nfe must allow at least one iteration for each of "
                f"{number_of_blocks} blocks; need at least "
                f"{number_of_blocks * base_forward_cost}."
            )
        total = min(sum(budgets), available)
        budgets = [1] * number_of_blocks
        for index in range(total - number_of_blocks):
            budgets[index % number_of_blocks] += 1
        return budgets

    @torch.inference_mode()
    def sample(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int = 64,
        steps: int = 64,
        max_nfe: int | None = None,
        block_size: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sampling_method: str = "multinomial",
        sampling_precision: str = "float64",
        sampling_chunk_size: int = 64,
        commit_policy: str = "max_prob",
        commit_schedule: str = "fixed",
        confidence_threshold: float = 0.9,
        min_commit: int = 1,
        max_commit: int | None = None,
        uncode_base_policy: str = "max_prob",
        uncode_position_lambda: float = 1.0,
        uncode_information_alpha: float = 10.0,
        uncode_trivial_penalty: float = 0.35,
        token_frequency_file: str | None = None,
        cfg_scale: float = 0.0,
        cfg_unconditional: str = "mask",
        negative_prompts: list[list[int]] | None = None,
        remask_policy: str = "none",
        remask_rate: float = 0.05,
        remask_start_fraction: float = 0.5,
        max_remasks_per_step: int = 4,
        max_revisions_per_token: int = 2,
        remask_window: str = "current",
        remask_cooldown: int = 1,
        remask_candidate_pool: int = 16,
        remask_accept: str = "improve",
        remask_eos: bool = False,
        min_new_tokens: int = 0,
        eos_stability_steps: int = 1,
        stop_on_eos: bool = True,
        remasking: str | None = None,
        return_history: bool = False,
        show_progress: bool = False,
    ) -> SamplerOutput:
        """Denoise a response canvas and optionally revise committed response tokens."""
        started_at = time.perf_counter()
        if not prompts:
            raise ValueError("At least one prompt is required.")
        if max_new_tokens < 1 or steps < 1 or block_size < 1:
            raise ValueError("max-new-tokens, steps, and block-size must be positive.")
        if max_nfe is not None and max_nfe < 1:
            raise ValueError("max-nfe must be positive.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("top-k must be non-negative and top-p must lie in (0, 1].")
        if sampling_method not in {"multinomial", "gumbel"}:
            raise ValueError("sampling-method must be 'multinomial' or 'gumbel'.")
        if sampling_precision not in {"float32", "float64"}:
            raise ValueError("sampling-precision must be 'float32' or 'float64'.")
        if sampling_chunk_size < 1:
            raise ValueError("sampling-chunk-size must be positive.")

        commit_policy = _normalize_choice(commit_policy)
        uncode_base_policy = _normalize_choice(uncode_base_policy)
        remask_policy = _normalize_choice(remask_policy)
        if remasking is not None:
            warnings.warn(
                "'remasking' is deprecated; use commit_policy and remask_policy.",
                DeprecationWarning,
                stacklevel=2,
            )
            legacy = _normalize_choice(remasking)
            if legacy == "low_confidence":
                commit_policy = "max_prob"
            elif legacy == "random":
                commit_policy = "random"
            else:
                raise ValueError("Legacy remasking must be 'low_confidence' or 'random'.")
        if commit_policy not in _COMMIT_POLICIES:
            raise ValueError(f"Unknown commit policy: {commit_policy}.")
        if uncode_base_policy not in {"max_prob", "margin", "entropy"}:
            raise ValueError("uncode-base-policy must be max-prob, margin, or entropy.")
        if commit_schedule not in {"fixed", "threshold"}:
            raise ValueError("commit-schedule must be 'fixed' or 'threshold'.")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence-threshold must lie in [0, 1].")
        if min_commit < 1 or (max_commit is not None and max_commit < 1):
            raise ValueError("min-commit and max-commit must be positive.")
        if max_commit is not None and max_commit < min_commit:
            raise ValueError("max-commit cannot be smaller than min-commit.")
        if uncode_information_alpha <= 0 or uncode_position_lambda < 0:
            raise ValueError("UNCODE alpha must be positive and lambda non-negative.")
        if not 0 < uncode_trivial_penalty <= 1:
            raise ValueError("uncode-trivial-penalty must lie in (0, 1].")
        if cfg_scale < 0 or cfg_unconditional not in {"mask", "pad"}:
            raise ValueError("CFG scale must be non-negative; mode must be mask or pad.")
        if negative_prompts is not None and len(negative_prompts) != len(prompts):
            raise ValueError("negative-prompts must match the prompt batch size.")
        if remask_policy not in _REMASK_POLICIES:
            raise ValueError(f"Unknown remask policy: {remask_policy}.")
        if not 0 < remask_rate <= 1 or not 0 <= remask_start_fraction <= 1:
            raise ValueError("remask-rate must lie in (0,1] and start-fraction in [0,1].")
        if max_remasks_per_step < 1 or max_revisions_per_token < 1:
            raise ValueError("Remask and revision limits must be positive.")
        if remask_window not in {"current", "previous", "global"}:
            raise ValueError("remask-window must be current, previous, or global.")
        if remask_cooldown < 0 or remask_candidate_pool < 1:
            raise ValueError("remask-cooldown must be non-negative and pool positive.")
        if remask_accept not in {"always", "improve"}:
            raise ValueError("remask-accept must be 'always' or 'improve'.")
        if min_new_tokens < 0 or min_new_tokens > max_new_tokens:
            raise ValueError("min-new-tokens must lie in [0, max-new-tokens].")
        if eos_stability_steps < 1:
            raise ValueError("eos-stability-steps must be positive.")

        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id
        if mask_id is None or pad_id is None:
            raise ValueError("Tokenizer must define mask and padding tokens.")
        stop_ids = self._stop_ids()
        prompt_tensors = [
            torch.tensor(prompt, dtype=torch.long, device=self.device) for prompt in prompts
        ]
        prompt_lengths = [len(prompt) for prompt in prompts]
        total_length = max(prompt_lengths) + max_new_tokens
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if context_limit and total_length > context_limit:
            raise ValueError(
                f"Requested sequence length {total_length} exceeds model limit {context_limit}."
            )

        batch_size = len(prompts)
        tokens = torch.full(
            (batch_size, total_length),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(tokens)
        response_positions = torch.zeros_like(tokens, dtype=torch.bool)
        for row, prompt in enumerate(prompt_tensors):
            length = prompt_lengths[row]
            tokens[row, :length] = prompt
            tokens[row, length : length + max_new_tokens] = mask_id
            attention_mask[row, : length + max_new_tokens] = 1
            response_positions[row, length : length + max_new_tokens] = True

        number_of_blocks = math.ceil(max_new_tokens / block_size)
        base_forward_cost = 2 if cfg_scale else 1
        budgets = self._iteration_budgets(
            steps=steps,
            number_of_blocks=number_of_blocks,
            base_forward_cost=base_forward_cost,
            max_nfe=max_nfe,
        )
        planned_iterations = sum(budgets)
        stats = SamplerStats(
            requested_steps=steps,
            planned_iterations=planned_iterations,
        )
        histories = [tokens.clone()] if return_history else None
        events = [
            TrajectoryEvent(
                kind="initial",
                iteration=0,
                forward_evaluations=0,
                block_index=None,
                masked_remaining=int((tokens.eq(mask_id) & response_positions).sum().item()),
            )
        ]
        commit_confidence = torch.full(
            tokens.shape,
            torch.inf,
            dtype=torch.float32,
            device=self.device,
        )
        revision_counts = torch.zeros_like(tokens)
        last_commit_iteration = torch.full_like(tokens, -10_000)
        eos_streak = torch.zeros_like(tokens)
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        global_iteration = 0

        if max_nfe is not None:
            progress_total: int | None = max_nfe
        elif remask_policy == "rescore":
            progress_total = planned_iterations * (base_forward_cost + 1)
        elif remask_policy != "none":
            progress_total = planned_iterations * base_forward_cost
        else:
            progress_total = sum(
                min(
                    budget,
                    min(block_size, max_new_tokens - index * block_size),
                )
                for index, budget in enumerate(budgets)
            ) * base_forward_cost

        def record_event(
            kind: str,
            *,
            block_index: int,
            committed: int = 0,
            remasked: int = 0,
            revised: int = 0,
        ) -> None:
            events.append(
                TrajectoryEvent(
                    kind=kind,
                    iteration=global_iteration,
                    forward_evaluations=stats.forward_evaluations,
                    block_index=block_index,
                    committed=committed,
                    remasked=remasked,
                    revised=revised,
                    masked_remaining=int(
                        (tokens.eq(mask_id) & response_positions).sum().item()
                    ),
                )
            )
            if histories is not None:
                histories.append(tokens.clone())

        with tqdm(
            total=progress_total,
            desc="Denoising",
            unit="forward",
            disable=not show_progress,
            dynamic_ncols=True,
        ) as progress:
            for block_index, iteration_budget in enumerate(budgets):
                block_region = torch.zeros_like(response_positions)
                current_block_starts = torch.zeros(
                    (batch_size,),
                    dtype=torch.long,
                    device=tokens.device,
                )
                current_block_ends = torch.zeros_like(current_block_starts)
                for row, prompt_length in enumerate(prompt_lengths):
                    start = prompt_length + block_index * block_size
                    stop = min(
                        start + block_size,
                        prompt_length + max_new_tokens,
                    )
                    current_block_starts[row] = start
                    current_block_ends[row] = stop
                    block_region[row, start:stop] = True
                block_region &= response_positions
                filled_once = finished.clone()

                for local_iteration in range(iteration_budget):
                    global_iteration += 1
                    block_clear = ~(tokens.eq(mask_id) & block_region).any(dim=1)
                    filled_once |= block_clear
                    pending_revision = torch.zeros_like(response_positions)
                    old_revision_tokens = torch.full_like(tokens, -1)
                    refinement_allowed = (
                        remask_policy != "none"
                        and local_iteration / max(1, iteration_budget)
                        >= remask_start_fraction
                    )
                    rows_for_refinement = filled_once & ~finished

                    if refinement_allowed and rows_for_refinement.any():
                        remaining_main_iterations = (
                            iteration_budget
                            - local_iteration
                            + sum(budgets[block_index + 1 :])
                        )
                        scope = self._remask_scope(
                            response_positions,
                            block_region=block_region,
                            block_index=block_index,
                            block_size=block_size,
                            prompt_lengths=prompt_lengths,
                            remask_window=remask_window,
                        )
                        scope &= rows_for_refinement[:, None]
                        pending_revision, old_revision_tokens = self._select_remasks(
                            tokens,
                            attention_mask,
                            scope,
                            commit_confidence=commit_confidence,
                            revision_counts=revision_counts,
                            last_commit_iteration=last_commit_iteration,
                            global_iteration=global_iteration,
                            mask_id=mask_id,
                            pad_id=pad_id,
                            stop_ids=stop_ids,
                            remask_policy=remask_policy,
                            remask_rate=remask_rate,
                            max_remasks_per_step=max_remasks_per_step,
                            max_revisions_per_token=max_revisions_per_token,
                            remask_cooldown=remask_cooldown,
                            remask_candidate_pool=remask_candidate_pool,
                            remask_eos=remask_eos,
                            stats=stats,
                            progress=progress,
                            max_nfe=max_nfe,
                            reserved_nfe=base_forward_cost * remaining_main_iterations,
                            block_starts=current_block_starts,
                            block_ends=current_block_ends,
                        )
                        if pending_revision.any():
                            remasked_count = int(pending_revision.sum().item())
                            tokens[pending_revision] = mask_id
                            revision_counts[pending_revision] += 1
                            stats.tokens_remasked += remasked_count
                            record_event(
                                "remask",
                                block_index=block_index,
                                remasked=remasked_count,
                            )

                    active = tokens.eq(mask_id) & (block_region | pending_revision)
                    active &= ~finished[:, None]
                    if not active.any():
                        break
                    if (
                        max_nfe is not None
                        and stats.forward_evaluations + base_forward_cost > max_nfe
                    ):
                        raise RuntimeError(
                            "The NFE budget ended before all active masks were resolved. "
                            "Increase --max-nfe or reduce the number of blocks."
                        )

                    logits = self._guided_forward(
                        tokens,
                        attention_mask,
                        prompt_lengths,
                        mask_id=mask_id,
                        pad_id=pad_id,
                        cfg_scale=cfg_scale,
                        cfg_unconditional=cfg_unconditional,
                        negative_prompts=negative_prompts,
                        stats=stats,
                        progress=progress,
                        max_nfe=max_nfe,
                        block_starts=(
                            torch.minimum(
                                current_block_starts,
                                torch.where(
                                    pending_revision.any(dim=1),
                                    pending_revision.float().argmax(dim=1),
                                    current_block_starts,
                                ),
                            )
                            if pending_revision.any()
                            else current_block_starts
                        ),
                        block_ends=current_block_ends,
                    )
                    suppressed_ids = {mask_id}
                    if pad_id not in stop_ids:
                        suppressed_ids.add(pad_id)
                    for token_id in suppressed_ids:
                        logits[..., token_id] = -torch.inf
                    if stop_ids and min_new_tokens:
                        for row, prompt_length in enumerate(prompt_lengths):
                            stop = min(
                                prompt_length + min_new_tokens,
                                prompt_length + max_new_tokens,
                            )
                            for stop_id in stop_ids:
                                logits[row, prompt_length:stop, stop_id] = -torch.inf

                    predictions = self._prediction_batch(
                        logits,
                        active,
                        prompt_lengths=prompt_lengths,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        sampling_method=sampling_method,
                        sampling_precision=sampling_precision,
                        commit_policy=commit_policy,
                        uncode_base_policy=uncode_base_policy,
                        uncode_position_lambda=uncode_position_lambda,
                        uncode_information_alpha=uncode_information_alpha,
                        uncode_trivial_penalty=uncode_trivial_penalty,
                        token_frequency_file=token_frequency_file,
                        sampling_chunk_size=sampling_chunk_size,
                    )
                    predicted_stop = torch.zeros_like(active)
                    for stop_id in stop_ids:
                        predicted_stop |= predictions.token_ids.eq(stop_id)
                    eos_streak[active & predicted_stop] += 1
                    eos_streak[active & ~predicted_stop] = 0
                    if local_iteration + 1 < iteration_budget:
                        unstable_eos = (
                            active
                            & predicted_stop
                            & eos_streak.lt(eos_stability_steps)
                        )
                        predictions.scores[unstable_eos] = -torch.inf

                    selected = pending_revision.clone()
                    remaining_iterations = max(1, iteration_budget - local_iteration)
                    for row in range(batch_size):
                        unresolved = (
                            active[row]
                            & block_region[row]
                            & ~pending_revision[row]
                        ).nonzero(as_tuple=False).flatten()
                        if unresolved.numel() == 0:
                            continue
                        required = math.ceil(unresolved.numel() / remaining_iterations)
                        row_scores = predictions.scores[row, unresolved]
                        finite = torch.isfinite(row_scores)
                        if commit_schedule == "threshold":
                            # The threshold always means maximum token
                            # probability. Policies such as UNCODE may rescale
                            # the ranking score, but must not silently change
                            # the meaning of --confidence-threshold.
                            row_confidence = predictions.candidate_probabilities[
                                row, unresolved
                            ]
                            above = finite & row_confidence.ge(confidence_threshold)
                            count = max(required, min_commit, int(above.sum().item()))
                            if max_commit is not None:
                                count = max(required, min(count, max_commit))
                        else:
                            count = required
                        count = min(count, unresolved.numel())
                        candidate_positions = unresolved[finite]
                        candidate_scores = row_scores[finite]
                        if candidate_positions.numel() < count:
                            if local_iteration + 1 < iteration_budget:
                                count = candidate_positions.numel()
                            else:
                                candidate_positions = unresolved
                                candidate_scores = row_scores
                                count = unresolved.numel()
                        if count:
                            ordering = candidate_scores.argsort(descending=True)
                            selected[row, candidate_positions[ordering[:count]]] = True

                    committed_count = 0
                    revised_count = 0
                    for row in range(batch_size):
                        row_selected = selected[row].nonzero(as_tuple=False).flatten()
                        if row_selected.numel() == 0:
                            continue
                        row_revision = pending_revision[row, row_selected]
                        if row_revision.any():
                            revision_positions = row_selected[row_revision]
                            old_tokens = old_revision_tokens[row, revision_positions]
                            accepted = torch.ones(
                                revision_positions.shape,
                                dtype=torch.bool,
                                device=tokens.device,
                            )
                            if remask_accept == "improve":
                                old_probabilities = _selected_probabilities(
                                    logits[row, revision_positions],
                                    old_tokens,
                                )
                                candidate_probabilities = _selected_probabilities(
                                    logits[row, revision_positions],
                                    predictions.token_ids[row, revision_positions],
                                )
                                accepted = candidate_probabilities >= old_probabilities
                                rejected_positions = revision_positions[~accepted]
                                if rejected_positions.numel():
                                    predictions.token_ids[row, rejected_positions] = (
                                        old_revision_tokens[row, rejected_positions]
                                    )
                                    predictions.candidate_probabilities[
                                        row,
                                        rejected_positions,
                                    ] = old_probabilities[~accepted]
                            accepted_positions = revision_positions[accepted]
                            if accepted_positions.numel():
                                revised_count += int(
                                    predictions.token_ids[row, accepted_positions]
                                    .ne(old_revision_tokens[row, accepted_positions])
                                    .sum()
                                    .item()
                                )
                        tokens[row, row_selected] = predictions.token_ids[row, row_selected]
                        commit_confidence[row, row_selected] = (
                            predictions.candidate_probabilities[row, row_selected]
                        )
                        last_commit_iteration[row, row_selected] = global_iteration
                        committed_count += row_selected.numel()

                    stats.iterations += 1
                    stats.tokens_committed += committed_count
                    stats.tokens_revised += revised_count
                    record_event(
                        "commit",
                        block_index=block_index,
                        committed=committed_count,
                        revised=revised_count,
                    )

                    if stop_on_eos and not remask_eos and stop_ids:
                        for row, prompt_length in enumerate(prompt_lengths):
                            if finished[row]:
                                continue
                            generated = tokens[
                                row,
                                prompt_length : prompt_length + max_new_tokens,
                            ]
                            stop_position = next(
                                (
                                    index
                                    for index, token_id in enumerate(generated.tolist())
                                    if token_id in stop_ids
                                ),
                                None,
                            )
                            if stop_position is None:
                                continue
                            if generated[: stop_position + 1].eq(mask_id).any():
                                continue
                            absolute_stop = prompt_length + stop_position
                            tail_start = absolute_stop + 1
                            tail_stop = prompt_length + max_new_tokens
                            tokens[row, tail_start:tail_stop] = pad_id
                            attention_mask[row, tail_start:tail_stop] = 0
                            finished[row] = True
                            stats.sequences_finished_by_eos += 1

                    block_still_masked = (
                        tokens.eq(mask_id) & block_region & ~finished[:, None]
                    ).any()
                    if not block_still_masked and remask_policy == "none":
                        break

                unresolved_block = (
                    tokens.eq(mask_id) & block_region & ~finished[:, None]
                )
                if unresolved_block.any():
                    raise RuntimeError(
                        f"Block {block_index} ended with unresolved masks. "
                        "Increase --steps/--max-nfe or relax EOS stability."
                    )

        unresolved = tokens.eq(mask_id) & response_positions
        if unresolved.any():
            raise RuntimeError("Sampler ended with unresolved response mask tokens.")
        stats.elapsed_seconds = time.perf_counter() - started_at
        return SamplerOutput(
            sequences=tokens,
            prompt_lengths=prompt_lengths,
            histories=histories,
            events=events,
            stats=stats,
        )


def decode_generations(tokenizer: Any, output: SamplerOutput) -> list[str]:
    """Decode only generated spans, stopping at EOS or end-of-turn."""
    stop_ids = {
        token_id
        for token_id in (
            tokenizer.eos_token_id,
            getattr(tokenizer, "eot_token_id", None),
        )
        if token_id is not None
    }
    texts: list[str] = []
    for row, prompt_length in zip(
        output.sequences.tolist(),
        output.prompt_lengths,
        strict=True,
    ):
        generated = row[prompt_length:]
        for index, token_id in enumerate(generated):
            if token_id in stop_ids:
                generated = generated[:index]
                break
        texts.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return texts
