"""Iterative masked-diffusion sampling and prompt formatting.

Run ``python -m diffusion_llm generate --help`` for one-shot inference or
``python -m diffusion_llm chat --help`` for an interactive session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffusion_llm.schedule import reveal_counts
from diffusion_llm.tokenization import token_id_list


@dataclass
class SamplerOutput:
    """Token sequences and optional denoising history."""

    sequences: torch.Tensor
    prompt_lengths: list[int]
    histories: list[torch.Tensor] | None = None


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


def _sample_predictions(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    uniform = torch.rand_like(logits, dtype=torch.float32).clamp_(1e-6, 1 - 1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    return (logits.float() + temperature * gumbel).argmax(dim=-1)


class MaskedDiffusionSampler:
    """Generate blocks of tokens by repeatedly predicting and revealing masks."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def sample(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int = 64,
        steps: int = 64,
        block_size: int = 32,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        return_history: bool = False,
        show_progress: bool = False,
    ) -> SamplerOutput:
        """Denoise a fixed mask canvas while preserving every prompt token."""
        if not prompts:
            raise ValueError("At least one prompt is required.")
        if max_new_tokens < 1 or steps < 1 or block_size < 1:
            raise ValueError("max_new_tokens, steps, and block_size must be positive.")
        if remasking not in {"low_confidence", "random"}:
            raise ValueError("remasking must be 'low_confidence' or 'random'.")

        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id
        if mask_id is None or pad_id is None:
            raise ValueError("Tokenizer must define mask and padding tokens.")

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
        for row, prompt in enumerate(prompt_tensors):
            length = prompt_lengths[row]
            tokens[row, :length] = prompt
            tokens[row, length : length + max_new_tokens] = mask_id
            attention_mask[row, : length + max_new_tokens] = 1

        histories = [tokens.clone()] if return_history else None
        number_of_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = max(1, math.ceil(steps / number_of_blocks))
        block_lengths = [
            min(block_size, max_new_tokens - block_index * block_size)
            for block_index in range(number_of_blocks)
        ]
        total_denoising_steps = sum(
            min(steps_per_block, block_length) for block_length in block_lengths
        )
        suppressed = {mask_id, pad_id}

        with tqdm(
            total=total_denoising_steps,
            desc="Denoising",
            unit="step",
            disable=not show_progress,
            dynamic_ncols=True,
        ) as progress:
            for block_index in range(number_of_blocks):
                eligible = torch.zeros_like(tokens, dtype=torch.bool)
                for row, prompt_length in enumerate(prompt_lengths):
                    start = prompt_length + block_index * block_size
                    stop = min(start + block_size, prompt_length + max_new_tokens)
                    eligible[row, start:stop] = True

                plan = reveal_counts((eligible & tokens.eq(mask_id)).sum(dim=1), steps_per_block)
                for step_index in range(plan.shape[1]):
                    current_masks = eligible & tokens.eq(mask_id)
                    if not current_masks.any():
                        break
                    logits = self.model(
                        input_ids=tokens,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits
                    for token_id in suppressed:
                        logits[..., token_id] = -torch.inf
                    predictions = _sample_predictions(logits, temperature)

                    if remasking == "low_confidence":
                        probabilities = F.softmax(logits.float(), dim=-1)
                        confidence = probabilities.gather(
                            -1,
                            predictions.unsqueeze(-1),
                        ).squeeze(-1)
                    else:
                        confidence = torch.rand(tokens.shape, device=self.device)
                    confidence = confidence.masked_fill(~current_masks, -torch.inf)

                    for row in range(batch_size):
                        count = min(
                            int(plan[row, step_index].item()),
                            int(current_masks[row].sum().item()),
                        )
                        if count == 0:
                            continue
                        selected = confidence[row].topk(count).indices
                        tokens[row, selected] = predictions[row, selected]
                    if histories is not None:
                        histories.append(tokens.clone())
                    progress.update()

        if tokens.eq(mask_id).any():
            raise RuntimeError("Sampler ended with unresolved mask tokens.")
        return SamplerOutput(tokens, prompt_lengths, histories)


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
