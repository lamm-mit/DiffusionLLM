"""Presentation-ready visualizations of the masked denoising process."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from diffusion_llm.sampling import SamplerOutput

_BACKGROUND = "#080C18"
_PANEL = "#11182A"
_TEXT = "#F4F7FF"
_MUTED = "#9BA9C5"
_ACCENT = "#7C5CFC"
_MASK = "#AE8CFF"
_NEW = "#FFD166"


def _font(size: int, *, bold: bool = False):
    names = (
        "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _decoded_piece(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=True)
    if text:
        return text
    token = tokenizer.convert_ids_to_tokens(token_id)
    return "" if token is None else str(token)


def _styled_characters(
    tokenizer: Any,
    token_ids: list[int],
    previous_ids: list[int] | None,
    mask_token_id: int,
) -> list[tuple[str, str]]:
    characters: list[tuple[str, str]] = []
    for index, token_id in enumerate(token_ids):
        if token_id == mask_token_id:
            piece = "· "
            color = _MASK
        else:
            piece = _decoded_piece(tokenizer, token_id)
            is_new = previous_ids is not None and previous_ids[index] == mask_token_id
            color = _NEW if is_new else _TEXT
        piece = piece.replace("\t", "    ").replace("\r", "")
        characters.extend((character, color) for character in piece)
    return characters


def _wrap_styled(
    characters: list[tuple[str, str]],
    *,
    columns: int,
    max_lines: int,
) -> list[list[tuple[str, str]]]:
    lines: list[list[tuple[str, str]]] = [[]]
    for character, color in characters:
        if character == "\n":
            lines.append([])
        else:
            if len(lines[-1]) >= columns:
                split_at = next(
                    (
                        index
                        for index in range(len(lines[-1]) - 1, max(-1, len(lines[-1]) - 20), -1)
                        if lines[-1][index][0].isspace()
                    ),
                    None,
                )
                if split_at is not None:
                    carry = lines[-1][split_at + 1 :]
                    lines[-1] = lines[-1][:split_at]
                    lines.append(carry)
                else:
                    lines.append([])
            lines[-1].append((character, color))
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            ellipsis_color = lines[-1][-1][1] if lines[-1] else _TEXT
            lines[-1] = lines[-1][: max(0, columns - 1)] + [("…", ellipsis_color)]
            break
    return lines or [[]]


def _prompt_lines(prompt: str, *, columns: int, max_lines: int) -> list[str]:
    normalized = prompt.replace("\t", "    ").replace("\r", "")
    lines: list[str] = []
    for paragraph in normalized.splitlines() or [""]:
        lines.extend(
            textwrap.wrap(
                paragraph,
                width=columns,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            or [""]
        )
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = f"{lines[-1][: max(0, columns - 1)]}…"
    return lines


def _draw_styled_line(
    draw: ImageDraw.ImageDraw,
    line: list[tuple[str, str]],
    *,
    x: int,
    y: int,
    font: Any,
    character_width: float,
) -> None:
    run_start = 0
    while run_start < len(line):
        color = line[run_start][1]
        run_stop = run_start + 1
        while run_stop < len(line) and line[run_stop][1] == color:
            run_stop += 1
        text = "".join(character for character, _ in line[run_start:run_stop])
        draw.text((x + run_start * character_width, y), text, fill=color, font=font)
        run_start = run_stop


def _render_frame(
    *,
    tokenizer: Any,
    token_ids: list[int],
    previous_ids: list[int] | None,
    mask_token_id: int,
    prompt_lines: list[str],
    result_line_capacity: int,
    step: int,
    total_steps: int,
    columns: int,
) -> Image.Image:
    width = 1280
    margin = 58
    line_height = 30
    prompt_padding = 28
    result_padding = 32
    gap = 24
    footer_height = 52
    prompt_height = max(92, 2 * prompt_padding + len(prompt_lines) * line_height)
    result_height = max(250, 2 * result_padding + result_line_capacity * line_height)
    height = margin + prompt_height + gap + result_height + footer_height + margin

    image = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    prompt_font = _font(19)
    result_font = _font(20)
    small_font = _font(13)
    character_box = draw.textbbox((0, 0), "M", font=result_font)
    character_width = character_box[2] - character_box[0]

    prompt_top = margin
    draw.rounded_rectangle(
        (margin, prompt_top, width - margin, prompt_top + prompt_height),
        radius=18,
        fill=_PANEL,
    )
    for line_index, line in enumerate(prompt_lines):
        draw.text(
            (margin + prompt_padding, prompt_top + prompt_padding + line_index * line_height),
            line,
            fill=_MUTED,
            font=prompt_font,
        )

    result_top = prompt_top + prompt_height + gap
    draw.rounded_rectangle(
        (margin, result_top, width - margin, result_top + result_height),
        radius=18,
        fill=_PANEL,
    )
    styled = _styled_characters(tokenizer, token_ids, previous_ids, mask_token_id)
    result_lines = _wrap_styled(styled, columns=columns, max_lines=result_line_capacity)
    for line_index, line in enumerate(result_lines):
        _draw_styled_line(
            draw,
            line,
            x=margin + result_padding,
            y=result_top + result_padding + line_index * line_height,
            font=result_font,
            character_width=character_width,
        )

    resolved = sum(token_id != mask_token_id for token_id in token_ids)
    progress = resolved / max(1, len(token_ids))
    bar_top = result_top + result_height + 23
    bar_width = width - 2 * margin
    draw.rounded_rectangle(
        (margin, bar_top, width - margin, bar_top + 8),
        radius=4,
        fill="#1C2740",
    )
    if progress:
        draw.rounded_rectangle(
            (margin, bar_top, margin + int(bar_width * progress), bar_top + 8),
            radius=4,
            fill=_ACCENT,
        )
    status = f"{step} / {total_steps}"
    status_box = draw.textbbox((0, 0), status, font=small_font)
    draw.text(
        (width - margin - (status_box[2] - status_box[0]), bar_top + 17),
        status,
        fill=_MUTED,
        font=small_font,
    )
    return image


def save_denoising_gif(
    tokenizer: Any,
    output: SamplerOutput,
    path: str | Path,
    *,
    prompt: str,
    frame_duration_ms: int = 220,
    row: int = 0,
    text_columns: int = 92,
    max_prompt_lines: int = 8,
    max_result_lines: int = 24,
) -> Path:
    """Render a sampler history as a looping prompt/result animation."""
    if not output.histories:
        raise ValueError("Sampler output has no history; pass return_history=True.")
    if frame_duration_ms < 20:
        raise ValueError("GIF frame duration must be at least 20 ms.")
    if text_columns < 20:
        raise ValueError("GIF text columns must be at least 20.")
    if row < 0 or row >= output.sequences.shape[0]:
        raise ValueError("GIF row is outside the sampler batch.")

    prompt_length = output.prompt_lengths[row]
    history = [
        frame[row, prompt_length:].detach().cpu().tolist() for frame in output.histories
    ]
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer must define a mask token.")

    wrapped_prompt = _prompt_lines(
        prompt,
        columns=text_columns,
        max_lines=max_prompt_lines,
    )
    final_styled = _styled_characters(tokenizer, history[-1], None, mask_token_id)
    result_line_capacity = min(
        max_result_lines,
        max(
            3,
            len(
                _wrap_styled(
                    final_styled,
                    columns=text_columns,
                    max_lines=max_result_lines,
                )
            ),
        ),
    )
    total_steps = len(history) - 1
    frames = [
        _render_frame(
            tokenizer=tokenizer,
            token_ids=token_ids,
            previous_ids=history[index - 1] if index else None,
            mask_token_id=mask_token_id,
            prompt_lines=wrapped_prompt,
            result_line_capacity=result_line_capacity,
            step=index,
            total_steps=total_steps,
            columns=text_columns,
        )
        for index, token_ids in enumerate(history)
    ]
    durations = [frame_duration_ms] * len(frames)
    durations[0] = max(frame_duration_ms, 900)
    durations[-1] = max(frame_duration_ms, 1500)

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return output_path
