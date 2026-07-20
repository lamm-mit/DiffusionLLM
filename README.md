# DiffusionLLM

[![CI](https://github.com/lamm-mit/DiffusionLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/lamm-mit/DiffusionLLM/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A small, course-oriented codebase for converting a decoder-only autoregressive
language model into a masked diffusion language model (MDLM), training it, and
generating text by iterative denoising.

This project is deliberately narrower than the upstream
[dLLM repository](https://github.com/ZHZisZZ/dllm): it implements one complete
AR-to-diffusion path with one CLI and no dependency on the upstream package.
Every implementation file lives in this directory.

> Converting attention does **not** by itself make the model a useful diffusion
> generator. Conversion preserves the AR weights as an initialization. The
> converted checkpoint must then be trained with the diffusion objective.

## What is included

- In-place, memory-conscious conversion for `qwen2` (including Qwen2.5),
  `qwen3`, and `llama`-family Hugging Face checkpoints.
- Padding-aware bidirectional attention using Transformers 5 mask utilities.
- Continuous-time masked diffusion loss with linear masking schedule.
- Raw-text continual pretraining and supervised fine-tuning (SFT).
- Full fine-tuning or LoRA.
- Blockwise iterative unmasking, one-shot generation, and interactive chat.
- Local JSON/JSONL/CSV/text datasets, saved `datasets` directories, or Hub IDs.
- Offline unit tests, attention invariance tests, and an end-to-end training test.

## Installation

Python 3.10+ is required. A CUDA GPU is strongly recommended for real models.

### Install from a clone

```bash
git clone https://github.com/lamm-mit/DiffusionLLM.git
cd DiffusionLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
diffusion-llm doctor
```

For development and tests, install the `dev` extra:

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=src pytest
ruff check .
```

### Install directly from GitHub

```bash
python -m pip install \
  "git+https://github.com/lamm-mit/DiffusionLLM.git"
diffusion-llm doctor
```

The package targets `transformers>=5.13,<6`. The attention implementation is
version-sensitive, so do not silently downgrade to the upstream repository's
older Transformers pin.

## Quick start

### 1. Convert an AR checkpoint

Qwen2.5-0.5B is a manageable classroom-scale default:

```bash
diffusion-llm convert \
  --source Qwen/Qwen2.5-0.5B \
  --output artifacts/qwen2.5-0.5b-diffusion-base \
  --dtype bfloat16
```

The command:

1. loads the source model and tokenizer;
2. adds `<|diffusion_mask|>` and resizes embeddings if necessary;
3. changes causal attention to bidirectional attention;
4. retains the pretrained weights and parameter names; and
5. saves a self-contained diffusion checkpoint.

Inspect it with:

```bash
diffusion-llm doctor --model artifacts/qwen2.5-0.5b-diffusion-base
```

### 2A. Continual pretraining on raw text

```bash
diffusion-llm train \
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

For multiple GPUs, launch the same module through Accelerate:

```bash
accelerate launch --module diffusion_llm train \
  --model artifacts/qwen2.5-0.5b-diffusion-base \
  --dataset Trelis/tiny-shakespeare \
  --mode pretrain \
  --text-field Text \
  --output artifacts/qwen2.5-0.5b-diffusion-shakespeare \
  --max-length 128 \
  --max-steps 500 \
  --batch-size 4 \
  --bf16
```

### 2B. Supervised fine-tuning

The SFT loader accepts:

- `messages`: a list of `{role, content}` objects;
- Alpaca fields: `instruction`, optional `input`, and `output`;
- `prompt` plus `response` or `completion`; or
- `text`, treated as an already formatted sequence.

```bash
diffusion-llm train \
  --model artifacts/qwen2.5-0.5b-diffusion-base \
  --dataset tatsu-lab/alpaca \
  --mode sft \
  --output artifacts/qwen2.5-0.5b-diffusion-alpaca \
  --max-length 256 \
  --max-steps 1000 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --bf16
```

By default, prompt tokens remain visible but do not contribute to the loss.
Use `--no-mask-prompt-loss` to train on the whole formatted conversation.

For lower memory use, add:

```bash
--lora --gradient-checkpointing
```

LoRA checkpoints record the converted model as their base and can be passed
directly to `generate` or `chat`.

### 3. Generate with iterative denoising

For a raw-text model:

```bash
diffusion-llm generate \
  --model artifacts/qwen2.5-0.5b-diffusion-shakespeare \
  --prompt "First Citizen: Before we proceed any further" \
  --max-new-tokens 64 \
  --steps 64 \
  --block-size 64 \
  --temperature 0.3
```

For an instruction-tuned model, add `--chat-template`:

```bash
diffusion-llm generate \
  --model artifacts/qwen2.5-0.5b-diffusion-alpaca \
  --prompt "Explain masked diffusion in two paragraphs." \
  --chat-template \
  --max-new-tokens 96 \
  --steps 96 \
  --block-size 32
```

`block-size` controls the generation regime:

- `block-size == max-new-tokens`: all output tokens are denoised together;
- a smaller block: semi-autoregressive generation, left block to right block;
- `block-size == 1`: an expensive AR-like limiting case.

`steps` is a total budget divided across blocks. Every mask is guaranteed to be
resolved even when there are fewer steps than tokens.

### 4. Interactive chat

```bash
diffusion-llm chat \
  --model artifacts/qwen2.5-0.5b-diffusion-alpaca \
  --max-new-tokens 96 \
  --steps 96 \
  --block-size 32
```

Use `/clear` to reset and `/quit` to exit. Pass `--raw-prompt` for a model that
was trained without a chat template.

## Command reference

```text
diffusion-llm convert   AR checkpoint -> bidirectional checkpoint
diffusion-llm train     continual pretraining or SFT
diffusion-llm generate  one-shot iterative denoising
diffusion-llm chat      interactive multi-turn inference
diffusion-llm doctor    dependency, device, and checkpoint inspection
```

Every subcommand has detailed help:

```bash
diffusion-llm train --help
```

## Method in one page

For a clean token sequence $x_0$, sample one diffusion time
$t \sim U(\epsilon, 1)$. Under the linear schedule,
$\alpha(t)=1-t$, so each trainable token is independently replaced by the
mask token with probability $t$. The bidirectional model predicts the clean
token at every masked position. The continuous-time objective is:

$$
\mathcal{L}
=
\mathbb{E}_{t,x_t}
\left[
\frac{1}{t}
\sum_{i \in M_t}
-\log p_\theta(x_{0,i}\mid x_t)
\right].
$$

During generation, the model starts with a fixed canvas of masks. Each step
predicts every unresolved position, ranks the predictions by confidence, and
commits a scheduled subset. The process repeats until no masks remain.

See [docs/method.md](docs/method.md) for the full explanation and
[docs/classroom-lab.md](docs/classroom-lab.md) for a lab sequence.

## Verification

```bash
PYTHONPATH=src pytest
ruff check .
```

The suite checks:

- a future token affects past logits (attention is truly bidirectional);
- padding does not affect real-token logits;
- conversion reloads through `AutoModelForMaskedLM`;
- mask-token embedding size matches the tokenizer;
- the reveal schedule fills every mask;
- prompts are immutable during sampling;
- diffusion loss is finite and differentiable; and
- conversion → local dataset → one training step → save works end to end.

## Project layout

```text
DiffusionLLM/
├── src/diffusion_llm/
│   ├── cli.py          # all CLI commands
│   ├── conversion.py   # AR checkpoint conversion
│   ├── modeling.py     # bidirectional Llama/Qwen model classes
│   ├── data.py         # local/Hub datasets and tokenization
│   ├── training.py     # MDLM objective and Trainer
│   ├── sampling.py     # iterative unmasking
│   ├── schedule.py     # forward and reverse schedules
│   └── loading.py      # full-checkpoint and LoRA loading
├── tests/
├── docs/
└── examples/
```

## Scope and limitations

- This is a teaching implementation, not a claim that short classroom runs
  reproduce published diffusion LLM quality.
- Only Qwen2/Qwen2.5, Qwen3, and Llama-family decoder layouts are supported.
  Other architectures need an explicit bidirectional model subclass and tests.
- Diffusion inference uses repeated full-sequence forward passes; it is much
  slower than optimized production kernels.
- The project intentionally excludes BD3LM, GRPO, lm-evaluation-harness,
  DeepSpeed recipes, Slurm wrappers, and model-specific production samplers.

## Attribution

The design was informed by the Apache-2.0
[ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) repository, especially its
`examples/a2d` pipeline, at commit
[`ca176752`](https://github.com/ZHZisZZ/dllm/commit/ca176752fbceec49c6b4777a2c18ae88e4eb10ed).
See [docs/upstream-review.md](docs/upstream-review.md) and [NOTICE](NOTICE).

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). This project
is distributed under the [Apache License 2.0](LICENSE).
