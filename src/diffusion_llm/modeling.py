"""Bidirectional versions of supported decoder-only Transformers models.

These classes retain the source architecture and parameter names. The only
substantive change is replacing the decoder's causal mask with a bidirectional,
padding-aware mask. Run ``python -m diffusion_llm convert --help`` to create a
checkpoint that uses one of these classes.
"""

from __future__ import annotations

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


class DiffusionLlamaConfig(LlamaConfig):
    """Configuration for a Llama-family masked diffusion model."""

    model_type = "diffusion-llama"


class DiffusionLlamaModel(LlamaModel):
    """Llama decoder with bidirectional, padding-aware attention."""

    config_class = DiffusionLlamaConfig

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        position_ids = _positions(inputs_embeds, past_key_values, position_ids)
        full_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
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


class DiffusionQwen2Config(Qwen2Config):
    """Configuration for a Qwen2/Qwen2.5 masked diffusion model."""

    model_type = "diffusion-qwen2"


class DiffusionQwen2Model(Qwen2Model):
    """Qwen2 decoder with bidirectional, padding-aware attention."""

    config_class = DiffusionQwen2Config

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        position_ids = _positions(inputs_embeds, past_key_values, position_ids)

        if isinstance(attention_mask, dict):
            mask_mapping = attention_mask
        else:
            full_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
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


class DiffusionQwen3Config(Qwen3Config):
    """Configuration for a Qwen3 masked diffusion model."""

    model_type = "diffusion-qwen3"


class DiffusionQwen3Model(Qwen3Model):
    """Qwen3 decoder with bidirectional, padding-aware attention."""

    config_class = DiffusionQwen3Config

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        position_ids = _positions(inputs_embeds, past_key_values, position_ids)

        if isinstance(attention_mask, dict):
            mask_mapping = attention_mask
        else:
            full_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
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
