"""Command-line interface for conversion, training, and diffusion inference.

Run ``python -m diffusion_llm --help`` or the installed ``diffusion-llm``
console command.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import accelerate
import datasets
import peft
import torch
import transformers

from diffusion_llm import __version__
from diffusion_llm.conversion import convert_checkpoint
from diffusion_llm.loading import choose_device, load_model, load_tokenizer
from diffusion_llm.sampling import MaskedDiffusionSampler, decode_generations, encode_prompt
from diffusion_llm.training import TrainConfig, train


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Diffusion checkpoint or LoRA adapter.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64, help="Total denoising-step budget.")
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Tokens generated together; use max-new-tokens for pure diffusion.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--remasking",
        choices=("low_confidence", "random"),
        default="low_confidence",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI parser."""
    parser = argparse.ArgumentParser(
        prog="diffusion-llm",
        description=(
            "Convert a supported decoder-only LLM to bidirectional attention, "
            "train it with masked diffusion, and run iterative denoising inference."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser("convert", help="Convert an AR checkpoint.")
    convert.add_argument("--source", required=True, help="HF model ID or local AR checkpoint.")
    convert.add_argument("--output", required=True)
    convert.add_argument("--mask-token", default="<|diffusion_mask|>")
    convert.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    convert.add_argument("--random-init", action="store_true")
    convert.add_argument("--trust-remote-code", action="store_true")
    convert.add_argument("--overwrite", action="store_true")
    convert.set_defaults(handler=_run_convert)

    train_parser = commands.add_parser("train", help="Pretrain or SFT a converted model.")
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--mode", choices=("pretrain", "sft"), default="sft")
    train_parser.add_argument("--dataset-config")
    train_parser.add_argument("--train-split", default="train")
    train_parser.add_argument("--eval-split")
    train_parser.add_argument("--validation-fraction", type=float, default=0.02)
    train_parser.add_argument("--text-field", default="text")
    train_parser.add_argument("--max-length", type=int, default=512)
    train_parser.add_argument(
        "--append-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument(
        "--mask-prompt-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument("--num-proc", type=int, default=1)
    train_parser.add_argument("--max-train-samples", type=int)
    train_parser.add_argument("--max-eval-samples", type=int)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--epochs", type=float, default=3.0)
    train_parser.add_argument("--max-steps", type=int, default=-1)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--eval-batch-size", type=int, default=4)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.03)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--logging-steps", type=int, default=10)
    train_parser.add_argument("--save-steps", type=int, default=250)
    train_parser.add_argument("--eval-steps", type=int, default=250)
    train_parser.add_argument("--save-total-limit", type=int, default=2)
    train_parser.add_argument("--time-epsilon", type=float, default=1e-3)
    train_parser.add_argument(
        "--loss-weighting",
        choices=("schedule", "uniform"),
        default="schedule",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--bf16", action="store_true")
    train_parser.add_argument("--fp16", action="store_true")
    train_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_parser.add_argument("--lora", action="store_true")
    train_parser.add_argument("--lora-rank", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=32)
    train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    train_parser.add_argument("--report-to", default="none", help="none, wandb, tensorboard, ...")
    train_parser.add_argument("--resume-from-checkpoint")
    train_parser.set_defaults(handler=_run_train)

    generate = commands.add_parser("generate", help="Run one-shot diffusion generation.")
    _add_inference_arguments(generate)
    generate.add_argument("--prompt", required=True)
    generate.add_argument(
        "--chat-template",
        action="store_true",
        help="Wrap the prompt as a user message with the tokenizer's chat template.",
    )
    generate.add_argument(
        "--gif",
        metavar="PATH",
        help="Save an animated GIF showing the iterative denoising trajectory.",
    )
    generate.add_argument(
        "--gif-frame-duration-ms",
        type=int,
        default=220,
        help="Duration of intermediate GIF frames in milliseconds.",
    )
    generate.add_argument("--json", action="store_true", help="Emit a machine-readable result.")
    generate.set_defaults(handler=_run_generate)

    chat = commands.add_parser("chat", help="Start an interactive diffusion chat.")
    _add_inference_arguments(chat)
    chat.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Do not use or maintain the tokenizer's chat template.",
    )
    chat.set_defaults(handler=_run_chat)

    doctor = commands.add_parser("doctor", help="Inspect dependencies and checkpoint metadata.")
    doctor.add_argument("--model", help="Optional converted checkpoint to inspect.")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=_run_doctor)
    return parser


def _run_convert(args: argparse.Namespace) -> None:
    output = convert_checkpoint(
        args.source,
        args.output,
        mask_token=args.mask_token,
        dtype=args.dtype,
        random_init=args.random_init,
        trust_remote_code=args.trust_remote_code,
        overwrite=args.overwrite,
    )
    print(f"Converted diffusion checkpoint: {output}")


def _run_train(args: argparse.Namespace) -> None:
    values = vars(args).copy()
    values.pop("command")
    values.pop("handler")
    output = train(TrainConfig(**values))
    print(f"Final training checkpoint: {output}")


def _sampler_from_args(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    model = load_model(
        args.model,
        dtype=args.dtype,
        device=args.device,
    )
    tokenizer = load_tokenizer(args.model)
    return tokenizer, MaskedDiffusionSampler(model, tokenizer)


def _sample_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "block_size": args.block_size,
        "temperature": args.temperature,
        "remasking": args.remasking,
    }


def _run_generate(args: argparse.Namespace) -> None:
    tokenizer, sampler = _sampler_from_args(args)
    prompt_ids = encode_prompt(
        tokenizer,
        args.prompt,
        chat_template=args.chat_template,
    )
    output = sampler.sample(
        [prompt_ids],
        return_history=args.gif is not None,
        **_sample_kwargs(args),
    )
    text = decode_generations(tokenizer, output)[0]
    gif_path = None
    if args.gif:
        from diffusion_llm.visualization import save_denoising_gif

        gif_path = save_denoising_gif(
            tokenizer,
            output,
            args.gif,
            prompt=args.prompt,
            frame_duration_ms=args.gif_frame_duration_ms,
        )
    if args.json:
        print(
            json.dumps(
                {
                    "prompt": args.prompt,
                    "text": text,
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens": args.max_new_tokens,
                    "gif": str(gif_path) if gif_path else None,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(text)
        if gif_path:
            print(f"\nDenoising GIF: {gif_path}")


def _run_chat(args: argparse.Namespace) -> None:
    tokenizer, sampler = _sampler_from_args(args)
    use_template = not args.raw_prompt
    if use_template and not getattr(tokenizer, "chat_template", None):
        print("Tokenizer has no chat template; falling back to raw single-turn prompts.")
        use_template = False
    messages: list[dict[str, str]] = []
    print("Diffusion chat. Type /clear to reset or /quit to exit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt == "/quit":
            break
        if prompt == "/clear":
            messages.clear()
            print("Conversation cleared.")
            continue
        if use_template:
            messages.append({"role": "user", "content": prompt})
            prompt_ids = encode_prompt(tokenizer, messages=messages, chat_template=True)
        else:
            prompt_ids = encode_prompt(tokenizer, prompt)
        output = sampler.sample([prompt_ids], **_sample_kwargs(args))
        response = decode_generations(tokenizer, output)[0]
        print(f"model> {response}")
        if use_template:
            messages.append({"role": "assistant", "content": response})


def _doctor_payload(model_path: str | None) -> dict[str, object]:
    device = choose_device("auto")
    payload: dict[str, object] = {
        "diffusion_llm": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "accelerate": accelerate.__version__,
        "peft": peft.__version__,
        "selected_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if model_path:
        adapter_base = None
        if (Path(model_path) / "adapter_config.json").exists():
            adapter_base = peft.PeftConfig.from_pretrained(model_path).base_model_name_or_path
        config = transformers.AutoConfig.from_pretrained(adapter_base or model_path)
        tokenizer = load_tokenizer(model_path)
        payload["checkpoint"] = {
            "path": str(Path(model_path).expanduser()),
            "adapter_base": adapter_base,
            "model_type": config.model_type,
            "architecture": getattr(config, "architectures", None),
            "diffusion_method": getattr(config, "diffusion_method", None),
            "source_model": getattr(config, "source_model_name_or_path", None),
            "vocab_size": config.vocab_size,
            "mask_token": tokenizer.mask_token,
            "mask_token_id": tokenizer.mask_token_id,
        }
    return payload


def _run_doctor(args: argparse.Namespace) -> None:
    payload = _doctor_payload(args.model)
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for nested_key, nested_value in value.items():
                print(f"  {nested_key}: {nested_value}")
        else:
            print(f"{key}: {value}")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and execute the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
