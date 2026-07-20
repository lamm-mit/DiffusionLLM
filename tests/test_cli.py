"""CLI integration tests. Run with ``PYTHONPATH=src pytest tests/test_cli.py``."""

from __future__ import annotations

from pathlib import Path

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

    def encode(self, text, add_special_tokens=True):
        return [2, 3]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "Prompt:"}]
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
        assert return_history
        prompt = prompts[0]
        initial = torch.tensor([prompt + [9] * max_new_tokens])
        middle = torch.tensor([prompt + [4] * (max_new_tokens // 2) + [9] * 2])
        final = torch.tensor([prompt + [4] * max_new_tokens])
        return SamplerOutput(
            sequences=final,
            prompt_lengths=[len(prompt)],
            histories=[initial, middle, final],
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
        ]
    )

    assert args.push_to_hub
    assert args.hub_model_id == "lamm-mit/classroom-diffusion"
    assert args.hub_private
    assert args.hub_strategy == "checkpoint"


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
