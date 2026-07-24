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
from diffusion_llm.evaluation import GenerationEvalConfig, evaluate_checkpoint
from diffusion_llm.loading import choose_device, load_model, load_tokenizer
from diffusion_llm.mixture import (
    MixtureBuildConfig,
    build_and_write_mixture,
    upload_saved_mixture,
)
from diffusion_llm.sampling import MaskedDiffusionSampler, decode_generations, encode_prompt
from diffusion_llm.tokenization import token_id_list
from diffusion_llm.training import TrainConfig, train


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Diffusion checkpoint or LoRA adapter.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Denoising-iteration budget; extra iterations enable revision when remasking.",
    )
    parser.add_argument(
        "--max-nfe",
        type=int,
        help="Optional hard budget on model forward evaluations, including CFG and rescoring.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Tokens generated together; use max-new-tokens for pure diffusion.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k filtering.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--sampling-method",
        choices=("multinomial", "gumbel"),
        default="multinomial",
        help="Categorical sampler; multinomial avoids full-vocabulary Gumbel noise.",
    )
    parser.add_argument(
        "--sampling-precision",
        choices=("float32", "float64"),
        default="float64",
        help="Precision used for categorical sampling.",
    )
    parser.add_argument(
        "--sampling-chunk-size",
        type=int,
        default=64,
        help="Positions sampled together; reduce this to lower peak memory.",
    )
    commitment = parser.add_argument_group("token commitment")
    commitment.add_argument(
        "--commit-policy",
        choices=("max-prob", "margin", "entropy", "random", "left-to-right", "uncode"),
        default="max-prob",
    )
    commitment.add_argument(
        "--commit-schedule",
        choices=("fixed", "threshold"),
        default="fixed",
    )
    commitment.add_argument("--confidence-threshold", type=float, default=0.9)
    commitment.add_argument("--min-commit", type=int, default=1)
    commitment.add_argument("--max-commit", type=int)
    commitment.add_argument(
        "--remasking",
        choices=("low_confidence", "random"),
        help=argparse.SUPPRESS,
    )
    uncode = parser.add_argument_group("UNCODE calibration")
    uncode.add_argument(
        "--uncode-base-policy",
        choices=("max-prob", "margin", "entropy"),
        default="max-prob",
    )
    uncode.add_argument("--uncode-position-lambda", type=float, default=1.0)
    uncode.add_argument("--uncode-information-alpha", type=float, default=10.0)
    uncode.add_argument("--uncode-trivial-penalty", type=float, default=0.35)
    uncode.add_argument(
        "--token-frequency-file",
        help="Optional JSON token-frequency object/list for UNCODE self-information.",
    )
    guidance = parser.add_argument_group("classifier-free guidance")
    guidance.add_argument("--cfg-scale", type=float, default=0.0)
    guidance.add_argument(
        "--cfg-unconditional",
        choices=("mask", "pad"),
        default="mask",
    )
    guidance.add_argument(
        "--negative-prompt",
        help="Optional raw negative condition placed in the unconditional prompt span.",
    )
    revision = parser.add_argument_group("training-free remasking")
    revision.add_argument(
        "--remask-policy",
        choices=("none", "confidence", "rescore", "random"),
        default="none",
    )
    revision.add_argument("--remask-rate", type=float, default=0.05)
    revision.add_argument("--remask-start-fraction", type=float, default=0.5)
    revision.add_argument("--max-remasks-per-step", type=int, default=4)
    revision.add_argument("--max-revisions-per-token", type=int, default=2)
    revision.add_argument(
        "--remask-window",
        choices=("current", "previous", "global"),
        default="current",
    )
    revision.add_argument("--remask-cooldown", type=int, default=1)
    revision.add_argument("--remask-candidate-pool", type=int, default=16)
    revision.add_argument(
        "--remask-accept",
        choices=("always", "improve"),
        default="improve",
    )
    revision.add_argument(
        "--remask-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Permit EOS revision; this disables safe early stopping.",
    )
    length = parser.add_argument_group("length and EOS")
    length.add_argument("--min-new-tokens", type=int, default=0)
    length.add_argument("--eos-stability-steps", type=int, default=1)
    length.add_argument(
        "--stop-on-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show denoising progress (disable with --no-progress).",
    )
    parser.add_argument(
        "--sampling-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report NFE, commitments, remasks, revisions, and elapsed time to stderr.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI parser."""
    parser = argparse.ArgumentParser(
        prog="diffusion-llm",
        description=(
            "Convert a supported decoder-only LLM to bidirectional attention, "
            "build chat mixtures, train with masked diffusion, and run iterative "
            "denoising inference."
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
    convert.add_argument(
        "--prediction-parameterization",
        choices=("same-position", "shifted"),
        default="same-position",
    )
    convert.add_argument(
        "--attention-pattern",
        choices=("full-bidirectional", "block-causal"),
        default="full-bidirectional",
    )
    convert.add_argument(
        "--time-conditioning",
        choices=("none", "additive"),
        default="none",
    )
    convert.add_argument("--time-embedding-dim", type=int, default=256)
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
    train_parser.add_argument(
        "--warmup-steps",
        type=float,
        default=0.03,
        help="Absolute steps when >=1, or a fraction of total steps when in [0,1).",
    )
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
    train_parser.add_argument(
        "--objective",
        choices=("legacy-mdlm", "mdlm-v2", "block-mdlm", "block-hybrid"),
        default="legacy-mdlm",
        help="Legacy behavior is unchanged; mdlm-v2 enables explicit v2 corruption.",
    )
    train_parser.add_argument(
        "--time-sampling",
        choices=("uniform", "stratified"),
        default="uniform",
    )
    train_parser.add_argument(
        "--mask-sampling",
        choices=("bernoulli", "uniform-count", "progressive"),
        default="bernoulli",
    )
    train_parser.add_argument("--progressive-stages", type=int, default=8)
    train_parser.add_argument(
        "--progressive-mask-probability",
        type=float,
        default=1.0,
        help="Mix progressive states with uniform-count states.",
    )
    train_parser.add_argument(
        "--loss-normalization",
        choices=("token", "sequence"),
        default="token",
    )
    train_parser.add_argument(
        "--prediction-parameterization",
        choices=("same-position", "shifted"),
        default="same-position",
    )
    train_parser.add_argument(
        "--attention-pattern",
        choices=("full-bidirectional", "block-causal"),
        default="full-bidirectional",
    )
    train_parser.add_argument("--train-block-sizes", default="16,32,64")
    train_parser.add_argument("--full-mdlm-ratio", type=float, default=0.25)
    train_parser.add_argument("--ar-loss-weight", type=float, default=0.0)
    train_parser.add_argument("--condition-dropout", type=float, default=0.0)
    train_parser.add_argument(
        "--condition-dropout-mode",
        choices=("mask", "pad"),
        default="mask",
    )
    train_parser.add_argument(
        "--mask-tail-augmentation",
        type=float,
        default=0.0,
        help="Probability of appending masked canvas tokens during training.",
    )
    train_parser.add_argument("--mask-tail-max-tokens", type=int, default=64)
    train_parser.add_argument("--mask-consistency-weight", type=float, default=0.0)
    train_parser.add_argument(
        "--time-conditioning",
        choices=("none", "additive"),
        default="none",
    )
    train_parser.add_argument("--time-embedding-dim", type=int, default=256)
    train_parser.add_argument(
        "--self-conditioning-probability",
        type=float,
        default=0.0,
    )
    train_parser.add_argument(
        "--draft-commit-probability",
        type=float,
        default=0.5,
    )
    train_parser.add_argument("--draft-loss-weight", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--bf16", action="store_true")
    train_parser.add_argument("--fp16", action="store_true")
    train_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_parser.add_argument("--lora", action="store_true")
    train_parser.add_argument("--lora-rank", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=32)
    train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    train_parser.add_argument("--report-to", default="none", help="none, wandb, tensorboard, ...")
    train_parser.add_argument(
        "--run-name",
        help="Experiment run name; W&B defaults to the output directory name.",
    )
    train_parser.add_argument(
        "--wandb-project",
        default="DiffusionLLM",
        help="W&B project used when --report-to includes wandb.",
    )
    train_parser.add_argument(
        "--wandb-entity",
        help="Optional W&B user or team entity.",
    )
    train_parser.add_argument("--resume-from-checkpoint")
    train_parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload saved and final models to the Hugging Face Hub.",
    )
    train_parser.add_argument(
        "--hub-model-id",
        help="Destination model repository, for example username/model-name.",
    )
    train_parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create the destination repository as private.",
    )
    train_parser.add_argument(
        "--hub-strategy",
        choices=("end", "every_save", "checkpoint", "all_checkpoints"),
        default="every_save",
        help="What to upload and when; every_save keeps the repository root current.",
    )
    train_parser.set_defaults(handler=_run_train)

    mixture = commands.add_parser(
        "build-mixture",
        help="Build an exact-size chat dataset from a JSON source manifest.",
    )
    mixture.add_argument(
        "--manifest",
        required=True,
        help="JSON file describing datasets, configurations, splits, caps, and licenses.",
    )
    mixture.add_argument("--target-train-rows", type=int, required=True)
    mixture.add_argument(
        "--save-to-disk",
        metavar="DIRECTORY",
        help="Write a recoverable datasets directory before any Hub upload.",
    )
    mixture.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the completed DatasetDict to the Hugging Face Hub.",
    )
    mixture.add_argument(
        "--hub-dataset-id",
        help="Destination dataset repository, for example username/dataset-name.",
    )
    mixture.add_argument("--hub-config-name", default="default")
    mixture.add_argument("--hub-private", action="store_true")
    mixture.add_argument("--max-shard-size", default="500MB")
    mixture.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Workers used to validate and normalize source rows.",
    )
    mixture.add_argument(
        "--upload-num-proc",
        type=int,
        default=1,
        help="Save/upload workers; one is the safest default on Python 3.13.",
    )
    mixture.add_argument("--seed", type=int, default=42)
    mixture.add_argument("--cache-dir")
    mixture.add_argument(
        "--validation-rows-per-source",
        type=int,
        help="Override every source's validation_rows manifest value.",
    )
    mixture.add_argument(
        "--max-validation-rows",
        type=int,
        help="Cap the final combined validation split after shuffling.",
    )
    mixture.add_argument(
        "--allowed-role",
        action="append",
        dest="allowed_roles",
        help="Allowed message role; repeat to replace system,user,assistant defaults.",
    )
    mixture.add_argument(
        "--require-final-assistant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every retained conversation to end in an assistant message.",
    )
    mixture.set_defaults(handler=_run_build_mixture)

    upload_mixture = commands.add_parser(
        "upload-mixture",
        help="Retry Hub upload from a locally saved chat mixture.",
    )
    upload_mixture.add_argument("--dataset", required=True, help="Saved DatasetDict directory.")
    upload_mixture.add_argument("--hub-dataset-id", required=True)
    upload_mixture.add_argument("--hub-config-name", default="default")
    upload_mixture.add_argument("--hub-private", action="store_true")
    upload_mixture.add_argument("--max-shard-size", default="500MB")
    upload_mixture.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Upload workers; one avoids stdin/spawn multiprocessing failures.",
    )
    upload_mixture.set_defaults(handler=_run_upload_mixture)

    generate = commands.add_parser("generate", help="Run one-shot diffusion generation.")
    _add_inference_arguments(generate)
    generate.add_argument("--prompt", required=True)
    generate.add_argument(
        "--chat-template",
        action="store_true",
        help="Format the prompt as chat, optionally with --system-prompt.",
    )
    generate.add_argument(
        "--system-prompt",
        help="Optional system message; requires --chat-template.",
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
    generate.add_argument(
        "--gif-max-frames",
        type=int,
        default=120,
        help="Downsample long trajectories while retaining remask/revision frames.",
    )
    generate.add_argument("--gif-text-columns", type=int, default=92)
    generate.add_argument("--gif-max-prompt-lines", type=int, default=8)
    generate.add_argument("--gif-max-result-lines", type=int, default=32)
    generate.add_argument("--json", action="store_true", help="Emit a machine-readable result.")
    generate.set_defaults(handler=_run_generate)

    evaluate = commands.add_parser(
        "evaluate",
        help="Generate answers for a reproducible sample from a held-out dataset split.",
    )
    _add_inference_arguments(evaluate)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--dataset-config")
    evaluate.add_argument("--split", default="validation")
    evaluate.add_argument("--num-samples", type=int, default=32)
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument(
        "--max-total-tokens",
        type=int,
        help="Skip prompts whose prompt plus generation would exceed this length.",
    )
    evaluate.add_argument("--source-field", default="source")
    evaluate.add_argument("--output", required=True, help="Detailed JSONL result path.")
    evaluate.add_argument(
        "--summary-output",
        help="Summary JSON path; defaults to OUTPUT with a .summary.json suffix.",
    )
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=_run_evaluate)

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
        prediction_parameterization=args.prediction_parameterization,
        attention_pattern=args.attention_pattern,
        time_conditioning=args.time_conditioning,
        time_embedding_dim=args.time_embedding_dim,
    )
    print(f"Converted diffusion checkpoint: {output}")


def _run_train(args: argparse.Namespace) -> None:
    values = vars(args).copy()
    values.pop("command")
    values.pop("handler")
    output = train(TrainConfig(**values))
    print(f"Final training checkpoint: {output}")


def _run_build_mixture(args: argparse.Namespace) -> None:
    allowed_roles = tuple(args.allowed_roles or ("system", "user", "assistant"))
    config = MixtureBuildConfig(
        manifest=args.manifest,
        target_train_rows=args.target_train_rows,
        save_to_disk=args.save_to_disk,
        push_to_hub=args.push_to_hub,
        hub_dataset_id=args.hub_dataset_id,
        hub_config_name=args.hub_config_name,
        hub_private=args.hub_private,
        max_shard_size=args.max_shard_size,
        num_proc=args.num_proc,
        upload_num_proc=args.upload_num_proc,
        seed=args.seed,
        cache_dir=args.cache_dir,
        validation_rows_per_source=args.validation_rows_per_source,
        max_validation_rows=args.max_validation_rows,
        allowed_roles=allowed_roles,
        require_final_assistant=args.require_final_assistant,
    )
    mixture = build_and_write_mixture(config)
    destinations = []
    if config.save_to_disk:
        destinations.append(str(Path(config.save_to_disk).expanduser().resolve()))
    if config.push_to_hub:
        destinations.append(f"{config.hub_dataset_id}:{config.hub_config_name}")
    print(f"Mixture complete ({len(mixture['train']):,} rows): {', '.join(destinations)}")


def _run_upload_mixture(args: argparse.Namespace) -> None:
    mixture = upload_saved_mixture(
        args.dataset,
        hub_dataset_id=args.hub_dataset_id,
        hub_config_name=args.hub_config_name,
        hub_private=args.hub_private,
        max_shard_size=args.max_shard_size,
        num_proc=args.num_proc,
    )
    print(
        f"Mixture upload complete ({len(mixture['train']):,} rows): "
        f"{args.hub_dataset_id}:{args.hub_config_name}"
    )


def _sampler_from_args(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    model = load_model(
        args.model,
        dtype=args.dtype,
        device=args.device,
    )
    tokenizer = load_tokenizer(args.model)
    return tokenizer, MaskedDiffusionSampler(model, tokenizer)


def _sample_kwargs(
    args: argparse.Namespace,
    tokenizer=None,
    *,
    batch_size: int = 1,
) -> dict[str, object]:
    values: dict[str, object] = {
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "max_nfe": args.max_nfe,
        "block_size": args.block_size,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "sampling_method": args.sampling_method,
        "sampling_precision": args.sampling_precision,
        "sampling_chunk_size": args.sampling_chunk_size,
        "commit_policy": args.commit_policy,
        "commit_schedule": args.commit_schedule,
        "confidence_threshold": args.confidence_threshold,
        "min_commit": args.min_commit,
        "max_commit": args.max_commit,
        "uncode_base_policy": args.uncode_base_policy,
        "uncode_position_lambda": args.uncode_position_lambda,
        "uncode_information_alpha": args.uncode_information_alpha,
        "uncode_trivial_penalty": args.uncode_trivial_penalty,
        "token_frequency_file": args.token_frequency_file,
        "cfg_scale": args.cfg_scale,
        "cfg_unconditional": args.cfg_unconditional,
        "remask_policy": args.remask_policy,
        "remask_rate": args.remask_rate,
        "remask_start_fraction": args.remask_start_fraction,
        "max_remasks_per_step": args.max_remasks_per_step,
        "max_revisions_per_token": args.max_revisions_per_token,
        "remask_window": args.remask_window,
        "remask_cooldown": args.remask_cooldown,
        "remask_candidate_pool": args.remask_candidate_pool,
        "remask_accept": args.remask_accept,
        "remask_eos": args.remask_eos,
        "min_new_tokens": args.min_new_tokens,
        "eos_stability_steps": args.eos_stability_steps,
        "stop_on_eos": args.stop_on_eos,
        "remasking": args.remasking,
        "show_progress": args.progress,
    }
    if args.negative_prompt is not None:
        if tokenizer is None:
            raise ValueError("A tokenizer is required to encode --negative-prompt.")
        negative_ids = token_id_list(
            tokenizer.encode(args.negative_prompt, add_special_tokens=True)
        )
        values["negative_prompts"] = [negative_ids.copy() for _ in range(batch_size)]
    return values


def _run_generate(args: argparse.Namespace) -> None:
    if args.system_prompt is not None and not args.chat_template:
        raise ValueError("--system-prompt requires --chat-template.")
    tokenizer, sampler = _sampler_from_args(args)
    messages = None
    if args.system_prompt is not None:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
    prompt_ids = encode_prompt(
        tokenizer,
        args.prompt,
        messages=messages,
        chat_template=args.chat_template,
    )
    output = sampler.sample(
        [prompt_ids],
        return_history=args.gif is not None,
        **_sample_kwargs(args, tokenizer),
    )
    text = decode_generations(tokenizer, output)[0]
    if args.sampling_stats and output.stats:
        stats = output.stats
        print(
            f"Sampling: {stats.forward_evaluations} NFE, "
            f"{stats.tokens_committed} commitments, "
            f"{stats.tokens_remasked} remasks, "
            f"{stats.tokens_revised} revisions, "
            f"{stats.elapsed_seconds:.2f}s",
            file=sys.stderr,
        )
    gif_path = None
    if args.gif:
        from diffusion_llm.visualization import save_denoising_gif

        gif_path = save_denoising_gif(
            tokenizer,
            output,
            args.gif,
            prompt=args.prompt,
            frame_duration_ms=args.gif_frame_duration_ms,
            max_frames=args.gif_max_frames,
            text_columns=args.gif_text_columns,
            max_prompt_lines=args.gif_max_prompt_lines,
            max_result_lines=args.gif_max_result_lines,
        )
    if args.json:
        print(
            json.dumps(
                {
                    "prompt": args.prompt,
                    "system_prompt": args.system_prompt,
                    "text": text,
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens": len(
                        tokenizer.encode(text, add_special_tokens=False)
                    ),
                    "sampler": output.stats.to_dict() if output.stats else None,
                    "gif": str(gif_path) if gif_path else None,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(text)
        if gif_path:
            print(f"\nDenoising GIF: {gif_path}")


def _run_evaluate(args: argparse.Namespace) -> None:
    values = vars(args).copy()
    values.pop("command")
    values.pop("handler")
    output, summary_output, summary = evaluate_checkpoint(GenerationEvalConfig(**values))
    metrics = summary["metrics"]
    print(f"Generation records: {output}")
    print(f"Evaluation summary: {summary_output}")
    print(
        f"Samples: {summary['samples']} | "
        f"non-empty: {metrics['nonempty_rate']:.1%} | "
        f"exact match: {metrics['exact_match_rate']:.1%} | "
        f"lexical token F1: {metrics['mean_lexical_token_f1']:.3f}"
    )
    sampler_totals = summary.get("sampler_totals", {})
    if args.sampling_stats and sampler_totals:
        print(
            f"Sampler totals: {int(sampler_totals.get('forward_evaluations', 0))} NFE, "
            f"{int(sampler_totals.get('tokens_remasked', 0))} remasks, "
            f"{int(sampler_totals.get('tokens_revised', 0))} revisions"
        )


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
        output = sampler.sample(
            [prompt_ids],
            **_sample_kwargs(args, tokenizer),
        )
        response = decode_generations(tokenizer, output)[0]
        print(f"model> {response}")
        if args.sampling_stats and output.stats:
            print(
                f"       [{output.stats.forward_evaluations} NFE, "
                f"{output.stats.tokens_remasked} remasks, "
                f"{output.stats.tokens_revised} revisions]"
            )
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
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
