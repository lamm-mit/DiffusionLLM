"""CLI integration tests. Run with ``PYTHONPATH=src pytest tests/test_cli.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from diffusion_llm import cli
from diffusion_llm.sampling import SamplerOutput


class FakeEncoding:
    def __init__(self, ids: list[int]):
        self.ids = ids


class ToyTokenizer:
    mask_token_id = 9
    pad_token_id = 0
    eos_token_id = 1
    eot_token_id = None
    chat_template = "present"

    def __init__(self):
        self.last_messages = None

    def encode(self, text, add_special_tokens=True):
        return [2, 3]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.last_messages = messages
        assert tokenize
        assert add_generation_prompt
        return FakeEncoding([2, 3])

    def decode(self, token_ids, skip_special_tokens=False):
        pieces = {2: "Prompt", 3: ":", 4: " generated", 9: "<mask>"}
        return "".join(pieces.get(token_id, "") for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id):
        return f"token-{token_id}"


class ToySampler:
    def sample(self, prompts, *, max_new_tokens, return_history, **kwargs):
        prompt = prompts[0]
        initial = torch.tensor([prompt + [9] * max_new_tokens])
        middle = torch.tensor([prompt + [4] * (max_new_tokens // 2) + [9] * 2])
        final = torch.tensor([prompt + [4] * max_new_tokens])
        return SamplerOutput(
            sequences=final,
            prompt_lengths=[len(prompt)],
            histories=[initial, middle, final] if return_history else None,
        )


def test_generate_cli_writes_gif(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_sampler_from_args", lambda args: (ToyTokenizer(), ToySampler()))
    gif_path = tmp_path / "denoising.gif"

    cli.main(
        [
            "generate",
            "--model",
            "unused",
            "--prompt",
            "Prompt:",
            "--chat-template",
            "--max-new-tokens",
            "4",
            "--steps",
            "2",
            "--block-size",
            "4",
            "--gif",
            str(gif_path),
            "--gif-frame-duration-ms",
            "40",
        ]
    )

    output = capsys.readouterr().out
    assert "generated" in output
    assert str(gif_path) in output
    assert gif_path.exists()


def test_generate_cli_applies_optional_system_message(
    monkeypatch,
    capsys,
) -> None:
    tokenizer = ToyTokenizer()
    monkeypatch.setattr(
        cli,
        "_sampler_from_args",
        lambda args: (tokenizer, ToySampler()),
    )

    cli.main(
        [
            "generate",
            "--model",
            "unused",
            "--prompt",
            "Question",
            "--system-prompt",
            "Scientific system instruction",
            "--chat-template",
            "--max-new-tokens",
            "4",
            "--steps",
            "2",
            "--block-size",
            "4",
            "--no-progress",
            "--json",
        ]
    )

    assert tokenizer.last_messages == [
        {"role": "system", "content": "Scientific system instruction"},
        {"role": "user", "content": "Question"},
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["system_prompt"] == "Scientific system instruction"


def test_generate_system_message_requires_chat_template(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli,
        "_sampler_from_args",
        lambda args: pytest.fail("model should not be loaded"),
    )

    with pytest.raises(SystemExit):
        cli.main(
            [
                "generate",
                "--model",
                "unused",
                "--prompt",
                "Question",
                "--system-prompt",
                "Scientific system instruction",
            ]
        )

    assert "--system-prompt requires --chat-template" in capsys.readouterr().err


def test_train_cli_parses_hub_options() -> None:
    args = cli.build_parser().parse_args(
        [
            "train",
            "--model",
            "base",
            "--dataset",
            "data",
            "--output",
            "output",
            "--push-to-hub",
            "--hub-model-id",
            "lamm-mit/classroom-diffusion",
            "--hub-private",
            "--hub-strategy",
            "checkpoint",
            "--warmup-steps",
            "0.05",
            "--report-to",
            "wandb",
            "--wandb-project",
            "DiffusionLLM",
            "--wandb-entity",
            "lamm-mit",
            "--run-name",
            "classroom-run",
        ]
    )

    assert args.push_to_hub
    assert args.hub_model_id == "lamm-mit/classroom-diffusion"
    assert args.hub_private
    assert args.hub_strategy == "checkpoint"
    assert args.warmup_steps == pytest.approx(0.05)
    assert args.report_to == "wandb"
    assert args.wandb_project == "DiffusionLLM"
    assert args.wandb_entity == "lamm-mit"
    assert args.run_name == "classroom-run"


def test_train_cli_preserves_legacy_defaults_and_parses_v2_objective() -> None:
    parser = cli.build_parser()
    legacy = parser.parse_args(
        [
            "train",
            "--model",
            "base",
            "--dataset",
            "data",
            "--output",
            "output",
        ]
    )
    v2 = parser.parse_args(
        [
            "train",
            "--model",
            "base",
            "--dataset",
            "data",
            "--output",
            "output",
            "--objective",
            "mdlm-v2",
            "--time-sampling",
            "stratified",
            "--mask-sampling",
            "uniform-count",
            "--loss-normalization",
            "sequence",
        ]
    )

    assert legacy.objective == "legacy-mdlm"
    assert legacy.time_sampling == "uniform"
    assert legacy.mask_sampling == "bernoulli"
    assert legacy.loss_normalization == "token"
    assert v2.objective == "mdlm-v2"
    assert v2.time_sampling == "stratified"
    assert v2.mask_sampling == "uniform-count"
    assert v2.loss_normalization == "sequence"

    block = parser.parse_args(
        [
            "train",
            "--model",
            "base",
            "--dataset",
            "data",
            "--output",
            "output",
            "--objective",
            "block-hybrid",
            "--prediction-parameterization",
            "shifted",
            "--attention-pattern",
            "block-causal",
            "--train-block-sizes",
            "16,32,64",
            "--full-mdlm-ratio",
            "0.25",
            "--ar-loss-weight",
            "0.1",
        ]
    )
    assert block.objective == "block-hybrid"
    assert block.prediction_parameterization == "shifted"
    assert block.attention_pattern == "block-causal"
    assert block.train_block_sizes == "16,32,64"
    assert block.ar_loss_weight == pytest.approx(0.1)


def test_evaluate_cli_parses_heldout_generation_options() -> None:
    args = cli.build_parser().parse_args(
        [
            "evaluate",
            "--model",
            "checkpoint-4400",
            "--dataset",
            "lamm-mit/diffusion-chat-mixture-1024",
            "--dataset-config",
            "chatmix_2m",
            "--split",
            "validation",
            "--num-samples",
            "32",
            "--batch-size",
            "2",
            "--max-total-tokens",
            "1024",
            "--max-new-tokens",
            "512",
            "--steps",
            "512",
            "--block-size",
            "8",
            "--temperature",
            "0",
            "--output",
            "artifacts/heldout.jsonl",
            "--no-progress",
        ]
    )

    assert args.dataset_config == "chatmix_2m"
    assert args.split == "validation"
    assert args.num_samples == 32
    assert args.batch_size == 2
    assert args.max_total_tokens == 1024
    assert args.max_new_tokens == 512
    assert args.output == "artifacts/heldout.jsonl"
    assert not args.progress


def test_build_mixture_cli_uses_safe_upload_default() -> None:
    args = cli.build_parser().parse_args(
        [
            "build-mixture",
            "--manifest",
            "mixture.json",
            "--target-train-rows",
            "2000000",
            "--save-to-disk",
            "artifacts/chatmix-2m",
            "--push-to-hub",
            "--hub-dataset-id",
            "lamm-mit/diffusion-chat-mixture-1024",
            "--hub-config-name",
            "chatmix_2m",
            "--num-proc",
            "16",
        ]
    )

    assert args.target_train_rows == 2_000_000
    assert args.num_proc == 16
    assert args.upload_num_proc == 1
    assert args.require_final_assistant

    retry = cli.build_parser().parse_args(
        [
            "upload-mixture",
            "--dataset",
            "artifacts/chatmix-2m",
            "--hub-dataset-id",
            "lamm-mit/diffusion-chat-mixture-1024",
        ]
    )
    assert retry.num_proc == 1


def test_inference_progress_is_on_by_default_and_can_be_disabled() -> None:
    parser = cli.build_parser()
    defaults = parser.parse_args(
        ["generate", "--model", "model", "--prompt", "Prompt:"]
    )
    disabled = parser.parse_args(
        ["generate", "--model", "model", "--prompt", "Prompt:", "--no-progress"]
    )

    assert defaults.progress
    assert not disabled.progress
    assert cli._sample_kwargs(defaults)["show_progress"]
    assert not cli._sample_kwargs(disabled)["show_progress"]


def test_generate_cli_parses_advanced_sampler_options() -> None:
    args = cli.build_parser().parse_args(
        [
            "generate",
            "--model",
            "model",
            "--prompt",
            "Prompt:",
            "--max-new-tokens",
            "512",
            "--steps",
            "768",
            "--max-nfe",
            "900",
            "--block-size",
            "16",
            "--temperature",
            "0.2",
            "--top-p",
            "0.95",
            "--sampling-method",
            "multinomial",
            "--sampling-precision",
            "float64",
            "--commit-policy",
            "uncode",
            "--commit-schedule",
            "threshold",
            "--confidence-threshold",
            "0.9",
            "--max-commit",
            "8",
            "--cfg-scale",
            "0.5",
            "--remask-policy",
            "rescore",
            "--remask-rate",
            "0.05",
            "--max-remasks-per-step",
            "2",
            "--remask-window",
            "previous",
            "--min-new-tokens",
            "32",
            "--eos-stability-steps",
            "2",
            "--gif",
            "trajectory.gif",
            "--gif-max-frames",
            "80",
        ]
    )

    values = cli._sample_kwargs(args)
    assert values["max_nfe"] == 900
    assert values["commit_policy"] == "uncode"
    assert values["commit_schedule"] == "threshold"
    assert values["cfg_scale"] == pytest.approx(0.5)
    assert values["remask_policy"] == "rescore"
    assert values["remask_window"] == "previous"
    assert values["eos_stability_steps"] == 2
    assert args.gif_max_frames == 80
