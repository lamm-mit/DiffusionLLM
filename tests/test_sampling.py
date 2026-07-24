"""Sampler tests. Run with ``PYTHONPATH=src pytest tests/test_sampling.py``."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from diffusion_llm import sampling
from diffusion_llm.sampling import MaskedDiffusionSampler


class ToyTokenizer:
    mask_token_id = 9
    pad_token_id = 0
    eos_token_id = 1
    eot_token_id = None

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = {1: "", 4: "alpha", 5: "beta", 6: ".", 7: "science", 9: "<mask>"}
        return "".join(pieces.get(token_id, "") for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id):
        return f"token-{token_id}"


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_position_embeddings=32)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 10), -20.0, device=input_ids.device)
        logits[..., 4] = 20.0
        return SimpleNamespace(logits=logits)


def test_sampler_preserves_prompt_and_resolves_masks() -> None:
    sampler = MaskedDiffusionSampler(ToyModel(), ToyTokenizer())
    output = sampler.sample(
        [[2, 3]],
        max_new_tokens=5,
        steps=3,
        block_size=5,
        return_history=True,
    )
    assert output.sequences[0, :2].tolist() == [2, 3]
    assert output.sequences[0, 2:].tolist() == [4, 4, 4, 4, 4]
    assert not output.sequences.eq(ToyTokenizer.mask_token_id).any()
    assert output.histories is not None
    assert len(output.histories) == 4
    assert output.histories[0][0, 2:].eq(ToyTokenizer.mask_token_id).all()
    assert not output.histories[-1].eq(ToyTokenizer.mask_token_id).any()


def test_progress_tracks_actual_forward_passes(monkeypatch) -> None:
    state = {"total": None, "updates": 0, "closed": False}

    class RecordingProgress:
        def __init__(self, *, total, **kwargs):
            state["total"] = total
            assert kwargs["desc"] == "Denoising"
            assert kwargs["unit"] == "forward"
            assert not kwargs["disable"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            state["closed"] = True

        def update(self, count=1):
            state["updates"] += count

    monkeypatch.setattr(sampling, "tqdm", RecordingProgress)
    MaskedDiffusionSampler(ToyModel(), ToyTokenizer()).sample(
        [[2, 3]],
        max_new_tokens=5,
        steps=4,
        block_size=2,
        show_progress=True,
    )

    assert state == {"total": 5, "updates": 5, "closed": True}


def test_threshold_schedule_commits_multiple_tokens_per_forward() -> None:
    output = MaskedDiffusionSampler(ToyModel(), ToyTokenizer()).sample(
        [[2, 3]],
        max_new_tokens=5,
        steps=5,
        block_size=5,
        commit_schedule="threshold",
        confidence_threshold=0.9,
        max_commit=5,
    )

    assert output.stats is not None
    assert output.stats.forward_evaluations == 1
    assert output.stats.iterations == 1
    assert output.stats.tokens_committed == 5
    assert output.stats.committed_tokens_per_forward == 5


def test_uncode_threshold_uses_probability_before_ranking_calibration() -> None:
    output = MaskedDiffusionSampler(ToyModel(), ToyTokenizer()).sample(
        [[2, 3]],
        max_new_tokens=5,
        steps=5,
        block_size=5,
        commit_policy="uncode",
        commit_schedule="threshold",
        confidence_threshold=0.9,
        max_commit=5,
    )

    assert output.stats is not None
    assert output.stats.forward_evaluations == 1


class ChangingModel(ToyModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        self.calls += 1
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 10), -20.0, device=input_ids.device)
        token_id = 4 if self.calls <= 2 else 5
        logits[..., token_id] = 20.0
        return SimpleNamespace(logits=logits)


def test_training_free_remasking_revises_committed_tokens() -> None:
    output = MaskedDiffusionSampler(ChangingModel(), ToyTokenizer()).sample(
        [[2, 3]],
        max_new_tokens=2,
        steps=4,
        block_size=2,
        remask_policy="confidence",
        remask_start_fraction=0.5,
        remask_rate=0.5,
        max_remasks_per_step=1,
        max_revisions_per_token=2,
        remask_accept="always",
        return_history=True,
    )

    assert output.stats is not None
    assert output.stats.tokens_remasked == 2
    assert output.stats.tokens_revised >= 1
    assert output.events is not None
    assert any(event.kind == "remask" for event in output.events)
    assert output.histories is not None
    assert any(
        history[0, 2:].eq(ToyTokenizer.mask_token_id).any()
        for history in output.histories[1:]
    )
    assert 5 in output.sequences[0, 2:].tolist()


def test_rescore_remasking_uses_only_reserved_nfe_budget() -> None:
    output = MaskedDiffusionSampler(ChangingModel(), ToyTokenizer()).sample(
        [[2, 3]],
        max_new_tokens=2,
        steps=4,
        max_nfe=6,
        block_size=2,
        remask_policy="rescore",
        remask_start_fraction=0.5,
        remask_rate=0.5,
        max_remasks_per_step=1,
        max_revisions_per_token=2,
        remask_candidate_pool=2,
        remask_accept="always",
    )

    assert output.stats is not None
    assert output.stats.forward_evaluations == 6
    assert output.stats.tokens_remasked == 2


class RecordingModel(ToyModel):
    def __init__(self):
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        self.inputs.append(input_ids.detach().clone())
        return super().forward(input_ids, attention_mask=attention_mask, use_cache=use_cache)


def test_cfg_masks_only_the_prompt_and_counts_both_forwards() -> None:
    model = RecordingModel()
    output = MaskedDiffusionSampler(model, ToyTokenizer()).sample(
        [[2, 3]],
        max_new_tokens=2,
        steps=1,
        block_size=2,
        cfg_scale=1.0,
    )

    assert output.stats is not None
    assert output.stats.forward_evaluations == 2
    assert len(model.inputs) == 2
    assert model.inputs[0][0, :2].tolist() == [2, 3]
    assert model.inputs[1][0, :2].tolist() == [9, 9]
    assert model.inputs[1][0, 2:].eq(9).all()


class PolicyModel(ToyModel):
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 10), -10.0, device=input_ids.device)
        logits[..., 4] = 10.0
        logits[:, 2, :] = -10.0
        logits[:, 2, 6] = 12.0
        logits[:, 3, :] = -10.0
        logits[:, 3, 7] = 10.0
        return SimpleNamespace(logits=logits)


def test_uncode_penalizes_trivial_punctuation_commitment() -> None:
    output = MaskedDiffusionSampler(PolicyModel(), ToyTokenizer()).sample(
        [[2, 3]],
        max_new_tokens=2,
        steps=2,
        block_size=2,
        commit_policy="uncode",
        uncode_position_lambda=0.0,
        uncode_trivial_penalty=0.01,
        return_history=True,
    )

    assert output.histories is not None
    first_commit = output.histories[1][0, 2:].tolist()
    assert first_commit == [9, 7]


class EosTokenizer(ToyTokenizer):
    pad_token_id = 1


class EosModel(ToyModel):
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 10), -20.0, device=input_ids.device)
        logits[..., 4] = 20.0
        logits[:, 4, 1] = 30.0
        return SimpleNamespace(logits=logits)


def test_eos_can_be_generated_when_pad_and_eos_share_an_id() -> None:
    output = MaskedDiffusionSampler(EosModel(), EosTokenizer()).sample(
        [[2, 3]],
        max_new_tokens=5,
        steps=5,
        block_size=5,
        stop_on_eos=True,
    )

    assert output.stats is not None
    assert output.stats.sequences_finished_by_eos == 1
    assert output.sequences[0, 4].item() == 1
    assert output.sequences[0, 5:].eq(1).all()


class AllEosModel(ToyModel):
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 10), -20.0, device=input_ids.device)
        logits[..., 1] = 20.0
        return SimpleNamespace(logits=logits)


def test_eos_is_remaskable_when_pad_and_eos_share_an_id() -> None:
    output = MaskedDiffusionSampler(AllEosModel(), EosTokenizer()).sample(
        [[2, 3]],
        max_new_tokens=1,
        steps=2,
        block_size=1,
        remask_policy="confidence",
        remask_start_fraction=0.0,
        max_remasks_per_step=1,
        max_revisions_per_token=1,
        remask_eos=True,
        remask_accept="always",
    )

    assert output.stats is not None
    assert output.stats.tokens_remasked == 1
    assert output.sequences[0, 2].item() == EosTokenizer.eos_token_id
