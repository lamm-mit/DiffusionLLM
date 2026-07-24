"""GIF rendering tests. Run with ``PYTHONPATH=src pytest tests/test_visualization.py``."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from diffusion_llm.sampling import SamplerOutput, TrajectoryEvent
from diffusion_llm.visualization import save_denoising_gif


class ToyTokenizer:
    mask_token_id = 9

    def decode(self, token_ids, skip_special_tokens=False):
        pieces = {2: "Hello", 3: " world", 4: "!", 9: "<mask>"}
        return "".join(pieces.get(token_id, f"<{token_id}>") for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id):
        return f"token-{token_id}"


def test_save_denoising_gif(tmp_path: Path) -> None:
    histories = [
        torch.tensor([[2, 3, 9, 9, 9]]),
        torch.tensor([[2, 3, 4, 9, 9]]),
        torch.tensor([[2, 3, 4, 3, 9]]),
        torch.tensor([[2, 3, 4, 3, 4]]),
    ]
    output = SamplerOutput(
        sequences=histories[-1],
        prompt_lengths=[2],
        histories=histories,
    )
    path = save_denoising_gif(
        ToyTokenizer(),
        output,
        tmp_path / "trajectory.gif",
        prompt="Hello world",
        frame_duration_ms=40,
        text_columns=24,
    )

    assert path.exists()
    with Image.open(path) as image:
        assert image.format == "GIF"
        assert image.n_frames == len(histories)
        assert image.width == 1280
        assert image.height > 300


def test_long_prompt_and_128_token_result_fit_canvas(tmp_path: Path) -> None:
    prompt = " ".join(
        [
            "Summarize the relationship between masked diffusion, parallel token",
            "prediction, confidence-based token commitment, and iterative refinement",
        ]
        * 5
    )
    final_tokens = [2, 3, 4, 3] * 32
    histories = []
    for resolved in (0, 16, 48, 80, 112, 128):
        generated = final_tokens[:resolved] + [9] * (128 - resolved)
        histories.append(torch.tensor([[2, 3, *generated]]))
    output = SamplerOutput(
        sequences=histories[-1],
        prompt_lengths=[2],
        histories=histories,
    )
    path = save_denoising_gif(
        ToyTokenizer(),
        output,
        tmp_path / "long-trajectory.gif",
        prompt=prompt,
        frame_duration_ms=40,
    )

    with Image.open(path) as image:
        assert image.n_frames == len(histories)
        assert image.width == 1280
        assert image.height <= 1200


def test_long_revision_history_is_downsampled(tmp_path: Path) -> None:
    histories = []
    events = []
    for index in range(40):
        generated = [4] * min(index, 8) + [9] * max(0, 8 - index)
        histories.append(torch.tensor([[2, 3, *generated]]))
        events.append(
            TrajectoryEvent(
                kind="remask" if index == 20 else "commit",
                iteration=index,
                forward_evaluations=index,
                block_index=0,
                remasked=1 if index == 20 else 0,
            )
        )
    output = SamplerOutput(
        sequences=histories[-1],
        prompt_lengths=[2],
        histories=histories,
        events=events,
    )
    path = save_denoising_gif(
        ToyTokenizer(),
        output,
        tmp_path / "downsampled.gif",
        prompt="Hello world",
        frame_duration_ms=40,
        max_frames=10,
        text_columns=24,
    )

    with Image.open(path) as image:
        assert image.n_frames <= 10
