"""Bidirectional versions of supported decoder-only Transformers models.

These classes retain the source architecture and parameter names while adding
full or block-causal diffusion attention and optional zero-initialized time
conditioning. Run ``python -m diffusion_llm convert --help`` to create a
checkpoint that uses one of these classes.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    LlamaConfig,
    Qwen2Config,
    Qwen3Config,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2PreTrainedModel,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
)

from diffusion_llm.attention import create_block_causal_mask


class DiffusionTimeConditioner(nn.Module):
    """Zero-initialized additive embedding of the current mask fraction."""

    def __init__(self, hidden_size: int, embedding_dim: int):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("diffusion-time-embedding-dim must be at least 2.")
        self.embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, diffusion_time: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        denominator = max(half - 1, 1)
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(
                half,
                device=diffusion_time.device,
                dtype=torch.float32,
            )
            / denominator
        )
        angles = diffusion_time.float()[:, None] * 1_000 * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.embedding_dim:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return self.projection(embedding.to(self.projection[0].weight.dtype))


def _initialize_time_conditioner(model: nn.Module, config: Any) -> None:
    kind = getattr(config, "diffusion_time_conditioning", "none")
    if kind == "none":
        model.diffusion_time_conditioner = None
        return
    if kind != "additive":
        raise ValueError(f"Unknown diffusion time conditioning: {kind}.")
    embedding_dim = int(
        getattr(config, "diffusion_time_embedding_dim", 256)
    )
    model.diffusion_time_conditioner = DiffusionTimeConditioner(
        config.hidden_size,
        embedding_dim,
    )


def _zero_time_conditioner_output(model: nn.Module) -> None:
    conditioner = getattr(model, "diffusion_time_conditioner", None)
    if conditioner is not None:
        nn.init.zeros_(conditioner.projection[-1].weight)
        nn.init.zeros_(conditioner.projection[-1].bias)


def configure_time_conditioning(
    model: nn.Module,
    *,
    kind: str,
    embedding_dim: int,
) -> None:
    """Enable optional time conditioning on a loaded or reclassified model."""
    if kind not in {"none", "additive"}:
        raise ValueError("time-conditioning must be 'none' or 'additive'.")
    if embedding_dim < 2:
        raise ValueError("time-embedding-dim must be at least 2.")
    core = getattr(model, "model", model)
    current = getattr(core, "diffusion_time_conditioner", None)
    current_dim = getattr(current, "embedding_dim", None)
    if kind == "none":
        core.diffusion_time_conditioner = None
    elif current is None or current_dim != embedding_dim:
        conditioner = DiffusionTimeConditioner(
            core.config.hidden_size,
            embedding_dim,
        )
        reference = core.embed_tokens.weight
        core.diffusion_time_conditioner = conditioner.to(
            device=reference.device,
            dtype=reference.dtype,
        )
    model.config.diffusion_time_conditioning = kind
    model.config.diffusion_time_embedding_dim = embedding_dim
    core.config = model.config


def _condition_embeddings(
    model: nn.Module,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    diffusion_time: torch.Tensor | None,
) -> torch.Tensor:
    conditioner = getattr(model, "diffusion_time_conditioner", None)
    if conditioner is None:
        return inputs_embeds
    if diffusion_time is None:
        mask_token_id = getattr(model.config, "mask_token_id", None)
        if input_ids is None or mask_token_id is None:
            diffusion_time = torch.zeros(
                inputs_embeds.shape[0],
                device=inputs_embeds.device,
            )
        else:
            valid = (
                attention_mask.bool()
                if attention_mask is not None and not isinstance(attention_mask, dict)
                else torch.ones_like(input_ids, dtype=torch.bool)
            )
            diffusion_time = (
                input_ids.eq(mask_token_id).logical_and(valid).sum(dim=1)
                / valid.sum(dim=1).clamp_min(1)
            )
    if diffusion_time.ndim == 0:
        diffusion_time = diffusion_time.expand(inputs_embeds.shape[0])
    if diffusion_time.shape != (inputs_embeds.shape[0],):
        raise ValueError("diffusion_time must contain one scalar per sequence.")
    return inputs_embeds + conditioner(diffusion_time)[:, None, :]


def _positions(
    inputs_embeds: torch.Tensor,
    past_key_values: Cache | None,
    position_ids: torch.LongTensor | None,
) -> torch.LongTensor:
    """Return monotonically increasing position IDs for the current tokens."""
    if position_ids is not None:
        return position_ids
    past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    positions = torch.arange(
        past_length,
        past_length + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )
    return positions.unsqueeze(0)


def _full_or_block_mask(
    config: Any,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    block_starts: torch.Tensor | None,
    block_ends: torch.Tensor | None,
) -> torch.Tensor | None:
    pattern = getattr(
        config,
        "diffusion_attention_pattern",
        "full-bidirectional",
    )
    if pattern == "full-bidirectional" or block_starts is None or block_ends is None:
        return create_bidirectional_mask(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
    if pattern != "block-causal":
        raise ValueError(f"Unknown diffusion attention pattern: {pattern}.")
    if past_key_values is not None and past_key_values.get_seq_length() > 0:
        raise ValueError("Block-causal training masks do not support a populated KV cache.")
    if attention_mask is None:
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=inputs_embeds.device,
        )
    return create_block_causal_mask(
        attention_mask,
        block_starts,
        block_ends,
        shifted_prediction=(
            getattr(
                config,
                "diffusion_prediction_parameterization",
                "same-position",
            )
            == "shifted"
        ),
    )


class DiffusionLlamaConfig(LlamaConfig):
    """Configuration for a Llama-family masked diffusion model."""

    model_type = "diffusion-llama"


class DiffusionLlamaModel(LlamaModel):
    """Llama decoder with bidirectional, padding-aware attention."""

    config_class = DiffusionLlamaConfig

    def __init__(self, config: DiffusionLlamaConfig):
        super().__init__(config)
        _initialize_time_conditioner(self, config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        diffusion_time: torch.Tensor | None = None,
        diffusion_block_starts: torch.LongTensor | None = None,
        diffusion_block_ends: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = _condition_embeddings(
            self,
            inputs_embeds,
            input_ids,
            attention_mask,
            diffusion_time,
        )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        position_ids = _positions(inputs_embeds, past_key_values, position_ids)
        full_mask = _full_or_block_mask(
            self.config,
            inputs_embeds,
            attention_mask,
            past_key_values,
            diffusion_block_starts,
            diffusion_block_ends,
        )
        kwargs["is_causal"] = False

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=full_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class DiffusionLlamaForMaskedLM(LlamaForCausalLM):
    """Llama language-model head backed by :class:`DiffusionLlamaModel`."""

    config_class = DiffusionLlamaConfig

    def __init__(self, config: DiffusionLlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = DiffusionLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        _zero_time_conditioner_output(self.model)


class DiffusionQwen2Config(Qwen2Config):
    """Configuration for a Qwen2/Qwen2.5 masked diffusion model."""

    model_type = "diffusion-qwen2"


class DiffusionQwen2Model(Qwen2Model):
    """Qwen2 decoder with bidirectional, padding-aware attention."""

    config_class = DiffusionQwen2Config

    def __init__(self, config: DiffusionQwen2Config):
        super().__init__(config)
        _initialize_time_conditioner(self, config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        diffusion_time: torch.Tensor | None = None,
        diffusion_block_starts: torch.LongTensor | None = None,
        diffusion_block_ends: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = _condition_embeddings(
            self,
            inputs_embeds,
            input_ids,
            attention_mask,
            diffusion_time,
        )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        position_ids = _positions(inputs_embeds, past_key_values, position_ids)

        if isinstance(attention_mask, dict):
            if diffusion_block_starts is not None or diffusion_block_ends is not None:
                raise ValueError("Block boundaries cannot be combined with a mask mapping.")
            mask_mapping = attention_mask
        else:
            full_mask = _full_or_block_mask(
                self.config,
                inputs_embeds,
                attention_mask,
                past_key_values,
                diffusion_block_starts,
                diffusion_block_ends,
            )
            mask_mapping = {"full_attention": full_mask}
            if self.has_sliding_layers:
                mask_mapping["sliding_attention"] = full_mask
        kwargs["is_causal"] = False

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for index, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask_mapping[self.config.layer_types[index]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class DiffusionQwen2ForMaskedLM(Qwen2ForCausalLM):
    """Qwen2 language-model head backed by :class:`DiffusionQwen2Model`."""

    config_class = DiffusionQwen2Config

    def __init__(self, config: DiffusionQwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = DiffusionQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        _zero_time_conditioner_output(self.model)


class DiffusionQwen3Config(Qwen3Config):
    """Configuration for a Qwen3 masked diffusion model."""

    model_type = "diffusion-qwen3"


class DiffusionQwen3Model(Qwen3Model):
    """Qwen3 decoder with bidirectional, padding-aware attention."""

    config_class = DiffusionQwen3Config

    def __init__(self, config: DiffusionQwen3Config):
        super().__init__(config)
        _initialize_time_conditioner(self, config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        diffusion_time: torch.Tensor | None = None,
        diffusion_block_starts: torch.LongTensor | None = None,
        diffusion_block_ends: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = _condition_embeddings(
            self,
            inputs_embeds,
            input_ids,
            attention_mask,
            diffusion_time,
        )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        position_ids = _positions(inputs_embeds, past_key_values, position_ids)

        if isinstance(attention_mask, dict):
            if diffusion_block_starts is not None or diffusion_block_ends is not None:
                raise ValueError("Block boundaries cannot be combined with a mask mapping.")
            mask_mapping = attention_mask
        else:
            full_mask = _full_or_block_mask(
                self.config,
                inputs_embeds,
                attention_mask,
                past_key_values,
                diffusion_block_starts,
                diffusion_block_ends,
            )
            mask_mapping = {"full_attention": full_mask}
            if self.has_sliding_layers:
                mask_mapping["sliding_attention"] = full_mask
        kwargs["is_causal"] = False

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for index, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask_mapping[self.config.layer_types[index]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class DiffusionQwen3ForMaskedLM(Qwen3ForCausalLM):
    """Qwen3 language-model head backed by :class:`DiffusionQwen3Model`."""

    config_class = DiffusionQwen3Config

    def __init__(self, config: DiffusionQwen3Config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = DiffusionQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        _zero_time_conditioner_output(self.model)


CONFIG_BY_SOURCE_TYPE = {
    "llama": DiffusionLlamaConfig,
    "qwen2": DiffusionQwen2Config,
    "qwen3": DiffusionQwen3Config,
}

MODEL_BY_SOURCE_TYPE = {
    "llama": DiffusionLlamaForMaskedLM,
    "qwen2": DiffusionQwen2ForMaskedLM,
    "qwen3": DiffusionQwen3ForMaskedLM,
}

BASE_MODEL_BY_SOURCE_TYPE = {
    "llama": DiffusionLlamaModel,
    "qwen2": DiffusionQwen2Model,
    "qwen3": DiffusionQwen3Model,
}


def register_diffusion_models() -> None:
    """Register all custom configs and model classes with Transformers."""
    registrations = (
        (DiffusionLlamaConfig, DiffusionLlamaForMaskedLM),
        (DiffusionQwen2Config, DiffusionQwen2ForMaskedLM),
        (DiffusionQwen3Config, DiffusionQwen3ForMaskedLM),
    )
    for config_class, model_class in registrations:
        AutoConfig.register(config_class.model_type, config_class, exist_ok=True)
        AutoModel.register(config_class, model_class, exist_ok=True)
        AutoModelForMaskedLM.register(config_class, model_class, exist_ok=True)
