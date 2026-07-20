# DiffusionLLM

[![CI](https://github.com/lamm-mit/DiffusionLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/lamm-mit/DiffusionLLM/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A focused, course-oriented codebase for converting a small decoder-only
autoregressive language model into a masked diffusion language model (MDLM),
training it, and generating text by iterative denoising.

> Conversion is only an initialization. Removing the causal mask preserves
> useful AR weights, but the resulting checkpoint must be trained with the
> diffusion objective before it becomes a useful diffusion generator.

## What is included

- AR-to-diffusion conversion for Qwen2/Qwen2.5, Qwen3, and Llama models.
- Padding-aware bidirectional attention with the original pretrained weights.
- Continuous-time absorbing-mask MDLM training.
- Raw-text continual pretraining and supervised fine-tuning.
- Full-parameter and LoRA training.
- Parallel and blockwise iterative denoising.
- Chat-template-aware SFT and inference.
- Animated GIF export of the complete denoising trajectory.
- Local files, saved `datasets` directories, and Hugging Face Hub datasets.
- Offline unit tests, attention tests, and an end-to-end training test.

## Installation

Python 3.10+ is required. A CUDA GPU is strongly recommended for real models.

### Recommended: uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/lamm-mit/DiffusionLLM.git
cd DiffusionLLM

uv sync --python 3.12
uv run diffusion-llm doctor
```

`uv sync` creates `.venv` and installs the project. Commands below use
`uv run`, so manual activation is unnecessary.

For development:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

### Compatible alternative: venv and pip

```bash
git clone https://github.com/lamm-mit/DiffusionLLM.git
cd DiffusionLLM

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
diffusion-llm doctor
```

The package targets `transformers>=5.13,<6`. Its attention implementation is
version-sensitive.

## Small end-to-end example

This converts Qwen2.5-0.5B-Instruct, runs a short LoRA SFT job, generates one
answer, and records the denoising process. It demonstrates the pipeline; it is
not intended to produce a strong checkpoint.

### 1. Convert the AR checkpoint

```bash
uv run diffusion-llm convert \
  --source Qwen/Qwen2.5-0.5B-Instruct \
  --output artifacts/qwen2.5-0.5b-diffusion-base \
  --dtype bfloat16
```

This command:

1. loads the AR model and tokenizer;
2. adds `<|diffusion_mask|>` and resizes the embeddings when necessary;
3. replaces causal attention with padding-aware bidirectional attention;
4. retains the pretrained weights and parameter names; and
5. saves a self-contained diffusion initialization.

Inspect it:

```bash
uv run diffusion-llm doctor \
  --model artifacts/qwen2.5-0.5b-diffusion-base
```

### 2. Run a short SFT smoke test

```bash
uv run diffusion-llm train \
  --model artifacts/qwen2.5-0.5b-diffusion-base \
  --dataset tatsu-lab/alpaca \
  --mode sft \
  --output artifacts/qwen2.5-0.5b-diffusion-smoke \
  --max-length 256 \
  --max-train-samples 5000 \
  --max-eval-samples 250 \
  --max-steps 250 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --lora \
  --gradient-checkpointing \
  --bf16
```

The SFT loader automatically applies the tokenizer chat template to
`messages`, `instruction/output`, and `prompt/response` records. Prompt tokens
remain visible as context but use label `-100`, so they are neither corrupted
nor included in the loss.

### 3. Generate and create a GIF

```bash
uv run diffusion-llm generate \
  --model artifacts/qwen2.5-0.5b-diffusion-smoke \
  --prompt "Explain masked diffusion in one concise paragraph." \
  --chat-template \
  --max-new-tokens 64 \
  --steps 24 \
  --block-size 64 \
  --temperature 0.2 \
  --gif artifacts/denoising.gif \
  --gif-frame-duration-ms 180
```

The animation contains only two clean text panels: the prompt and the evolving
result. Unresolved tokens appear as purple dots, newly committed text appears
in gold, and older committed text becomes white. A small progress bar and step
counter show the trajectory. The initial and final frames pause automatically.

For the clearest parallel-denoising demonstration, make `--block-size` equal to
`--max-new-tokens`. GIF history is collected only when `--gif` is supplied.
The renderer is tested with a long prompt and a 128-token result; the canvas
grows with wrapped output up to 24 result lines.

## Substantial training recipe

For materially better results, use a stronger instruction-tuned
initialization, high-quality conversational data, full-model training, and
enough steps for every attention layer to adapt to bidirectional denoising.

This baseline uses:

- [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct);
- the 207k-conversation `train_sft` split of
  [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k);
- three epochs at sequence length 1024;
- effective batch size 64 across four GPUs; and
- full-parameter training rather than LoRA.

Convert the larger initialization:

```bash
uv run diffusion-llm convert \
  --source Qwen/Qwen2.5-1.5B-Instruct \
  --output artifacts/qwen2.5-1.5b-diffusion-base \
  --dtype bfloat16
```

Launch distributed training:

```bash
uv run accelerate launch \
  --num_processes 4 \
  --module diffusion_llm train \
  --model artifacts/qwen2.5-1.5b-diffusion-base \
  --dataset HuggingFaceH4/ultrachat_200k \
  --train-split train_sft \
  --eval-split test_sft \
  --mode sft \
  --output artifacts/qwen2.5-1.5b-diffusion-ultrachat \
  --max-length 1024 \
  --epochs 3 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --warmup-ratio 0.03 \
  --weight-decay 0.1 \
  --max-eval-samples 2000 \
  --logging-steps 10 \
  --save-steps 500 \
  --eval-steps 500 \
  --save-total-limit 3 \
  --num-proc 16 \
  --gradient-checkpointing \
  --bf16
```

With four processes, per-device batch 2 and gradient accumulation 8 give an
effective batch of `4 × 2 × 8 = 64`. The complete run is roughly 9,700
optimizer steps before filtering or truncation effects.

Plan on approximately four 80 GB training GPUs for this conservative recipe.
Memory depends on the GPU, kernels, and example lengths. Reduce
`--batch-size` first if necessary. On one 80 GB GPU, start with batch 1 and
gradient accumulation 64.

Generate a realistic long response and animation:

```bash
uv run diffusion-llm generate \
  --model artifacts/qwen2.5-1.5b-diffusion-ultrachat \
  --prompt "Explain how masked diffusion generates text in parallel and compare it with autoregressive decoding." \
  --chat-template \
  --max-new-tokens 128 \
  --steps 48 \
  --block-size 128 \
  --temperature 0.2 \
  --gif artifacts/ultrachat-denoising.gif \
  --gif-frame-duration-ms 140
```

This is a serious baseline with a reasonable chance of producing a useful
classroom model, but it is not a quality guarantee. Evaluate held-out MDLM
loss, response quality, repetition, instruction following, and speed against
the original AR checkpoint.

## Other modes

The SFT loader accepts:

- `messages`: a list of `{role, content}` records;
- Alpaca `instruction`, optional `input`, and `output`;
- `prompt` plus `response` or `completion`; or
- `text`, treated as an already formatted sequence.

Add `--lora` for a lower-memory experiment. LoRA checkpoints record their
converted base model and can be passed directly to `generate` or `chat`.

For raw-text continual pretraining:

```bash
uv run diffusion-llm train \
  --model artifacts/qwen2.5-0.5b-diffusion-base \
  --dataset Trelis/tiny-shakespeare \
  --mode pretrain \
  --text-field Text \
  --output artifacts/qwen2.5-0.5b-diffusion-shakespeare \
  --max-length 128 \
  --max-steps 500 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --bf16
```

Interactive chat uses the tokenizer chat template by default:

```bash
uv run diffusion-llm chat \
  --model artifacts/qwen2.5-1.5b-diffusion-ultrachat \
  --max-new-tokens 96 \
  --steps 48 \
  --block-size 32
```

Use `/clear` to reset and `/quit` to exit. Pass `--raw-prompt` when the model
was trained without a chat template.

## Exact method

This repository implements a **continuous-time masked diffusion language model
(MDLM) with an absorbing mask state**.

### Model conversion

The converter preserves the tokenizer, token embeddings, transformer blocks,
LM head, pretrained weights, and the source model's rotary token positions. It
replaces lower-triangular causal attention with padding-aware bidirectional
attention and adds one mask token.

### Forward corruption and objective

For every sequence:

1. Sample one scalar diffusion time
   $t \sim U(\epsilon,1)$, with $\epsilon=10^{-3}$ by default.
2. Use the linear survival schedule $\alpha(t)=1-t$.
3. Independently replace each trainable token by the mask token with
   probability $1-\alpha(t)=t$.
4. Predict the clean token at the same position from the bidirectional noised
   sequence. There is no causal right shift.
5. Apply the continuous-time schedule weight
   $-\alpha'(t)/(1-\alpha(t))=1/t$ only to corrupted target positions.

```math
\mathcal{L}
=
\mathbb{E}_{t,x_t}
\left[
\frac{1}{t}
\sum_{i \in M_t}
-\log p_\theta(x_{0,i}\mid x_t)
\right].
```

### Time encoding

**The network receives no explicit diffusion-time encoding.** Time $t$ is used
by the corruption sampler and loss weighting, but is not embedded, added to
token states, or passed to the Transformer. The denoiser conditions on $x_t$
itself, particularly the locations and fraction of mask tokens.

The original Qwen/Llama rotary position embeddings remain active. They encode
token position in the sequence, not diffusion time.

### Reverse process

Generation starts with a fixed canvas of masks. At every step the network:

1. predicts all currently unresolved positions in one bidirectional pass;
2. excludes the mask and padding tokens as output candidates;
3. ranks predictions by confidence; and
4. permanently commits a scheduled number of tokens.

The process repeats until no masks remain. With
`block-size < max-new-tokens`, blocks are completed from left to right.
`--remasking random` replaces confidence ranking with random selection as an
ablation.

This is iterative absorbing-mask denoising. It is not Gaussian diffusion, does
not use a learned time embedding, and is not an exact ancestral sampler for a
general discrete transition matrix.

See [docs/method.md](docs/method.md) for the longer derivation and
[docs/classroom-lab.md](docs/classroom-lab.md) for a teaching lab.

## Command reference

```text
diffusion-llm convert   AR checkpoint -> bidirectional checkpoint
diffusion-llm train     continual pretraining or SFT
diffusion-llm generate  one-shot denoising, optionally with GIF export
diffusion-llm chat      interactive multi-turn inference
diffusion-llm doctor    dependency, device, and checkpoint inspection
```

Every subcommand provides detailed help:

```bash
uv run diffusion-llm generate --help
```

`block-size` controls generation:

- `block-size == max-new-tokens`: denoise the entire output together;
- smaller blocks: semi-autoregressive, block-by-block generation; and
- `block-size == 1`: an expensive AR-like limiting case.

`steps` is a total budget divided across blocks. Every mask is guaranteed to
resolve even when there are fewer steps than output tokens.

## Verification

```bash
uv run pytest
uv run ruff check .
```

The suite checks:

- future tokens affect earlier logits;
- padding does not change real-token logits;
- converted checkpoints reload through `AutoModelForMaskedLM`;
- mask-token embedding sizes match the tokenizer;
- reveal schedules fill every mask;
- prompts remain immutable;
- long denoising histories render as valid multi-frame GIFs;
- diffusion loss is finite and differentiable; and
- conversion, local data, training, save, reload, and generation work together.

## Project layout

```text
DiffusionLLM/
├── src/diffusion_llm/
│   ├── cli.py           # all CLI commands
│   ├── conversion.py    # AR checkpoint conversion
│   ├── modeling.py      # bidirectional Llama/Qwen model classes
│   ├── data.py          # local and Hub datasets
│   ├── training.py      # continuous-time MDLM objective
│   ├── sampling.py      # iterative unmasking
│   ├── schedule.py      # corruption and reveal schedules
│   ├── visualization.py # denoising GIF renderer
│   └── loading.py       # full-checkpoint and LoRA loading
├── tests/
├── docs/
└── examples/
```

## Scope and limitations

- Short classroom runs do not reproduce published diffusion-LLM quality.
- Only Qwen2/Qwen2.5, Qwen3, and Llama-family decoder layouts are supported.
- Diffusion inference performs repeated full-sequence forward passes and is
  slower than optimized production kernels.
- This project intentionally excludes BD3LM, GRPO, lm-evaluation-harness,
  DeepSpeed recipes, Slurm wrappers, and architecture-specific production
  samplers.

## Attribution

The design was informed by the Apache-2.0
[`ZHZisZZ/dllm`](https://github.com/ZHZisZZ/dllm) repository, especially its
`examples/a2d` pipeline, at commit
[`ca176752`](https://github.com/ZHZisZZ/dllm/commit/ca176752fbceec49c6b4777a2c18ae88e4eb10ed).
See [docs/upstream-review.md](docs/upstream-review.md) and [NOTICE](NOTICE).

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). This project
is distributed under the [Apache License 2.0](LICENSE).
