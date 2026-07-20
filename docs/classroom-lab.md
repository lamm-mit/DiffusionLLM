# Getting started: turn a small AR model into a diffusion model

## Objectives

These examples allow you to:

1. explain why a causal decoder cannot denoise arbitrary masked positions;
2. distinguish checkpoint conversion from diffusion training;
3. derive the linear MDLM mask probability and loss weight;
4. trace the forward corruption and reverse reveal processes; and
5. measure how steps and block size affect speed and output quality.

## Part 1: verify the environment

```bash
python -m pip install -e ".[dev]"
diffusion-llm doctor
PYTHONPATH=src pytest
```

Discuss the two attention tests before continuing. The most important assertion
is that modifying token $j>i$ changes the logits at $i$.

## Part 2: create the initialization

```bash
diffusion-llm convert \
  --source Qwen/Qwen2.5-0.5B \
  --output artifacts/base \
  --dtype bfloat16

diffusion-llm doctor --model artifacts/base
```

Ask students to compare `config.json` before and after conversion. The parameter
shapes should be unchanged except for a possible one-token vocabulary increase.

## Part 3: run a deliberately short training job

```bash
diffusion-llm train \
  --model artifacts/base \
  --dataset Trelis/tiny-shakespeare \
  --mode pretrain \
  --text-field Text \
  --output artifacts/shakespeare-100 \
  --max-length 128 \
  --max-steps 100 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --bf16
```

This is a pipeline exercise, not a quality target. Hardware and dataset size
determine the appropriate full experiment.

## Part 4: run sampling ablations

Whole-span diffusion:

```bash
diffusion-llm generate \
  --model artifacts/shakespeare-100 \
  --prompt "First Citizen:" \
  --max-new-tokens 64 \
  --steps 64 \
  --block-size 64
```

Four semi-autoregressive blocks:

```bash
diffusion-llm generate \
  --model artifacts/shakespeare-100 \
  --prompt "First Citizen:" \
  --max-new-tokens 64 \
  --steps 64 \
  --block-size 16
```

Fewer network evaluations:

```bash
diffusion-llm generate \
  --model artifacts/shakespeare-100 \
  --prompt "First Citizen:" \
  --max-new-tokens 64 \
  --steps 16 \
  --block-size 16
```

Record:

| Run | Steps | Block size | Wall time | Repetition | Local coherence |
|---|---:|---:|---:|---:|---:|
| A | 64 | 64 | | | |
| B | 64 | 16 | | | |
| C | 16 | 16 | | | |

## Part 5: inspect the objective

In `training.py`, locate:

- the sampled `time`;
- `mask_probability(time)`;
- the forced-mask safeguard;
- same-position cross-entropy; and
- `loss_weight(time)`.

Then train a second run with `--loss-weighting uniform`. Compare validation loss
carefully: the objectives have different weighting, so raw values are not a
complete apples-to-apples quality metric.

## Discussion questions

1. Why can the AR initialization still help after removing the causal mask?
2. Why are target logits not shifted right?
3. What happens as $t\to 0$, and why is `time_epsilon` needed?
4. Why might a small block outperform whole-span diffusion after short training?
5. Why is conversion support architecture-specific even though the loss is not?
6. Which computations could be cached or sparsified in a production sampler?
