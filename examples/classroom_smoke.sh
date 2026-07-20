#!/usr/bin/env bash
# Run from the DiffusionLLM root after installing the package.
set -euo pipefail

BASE="${BASE:-artifacts/qwen2.5-0.5b-diffusion-base}"
TRAINED="${TRAINED:-artifacts/qwen2.5-0.5b-diffusion-smoke}"

diffusion-llm convert \
  --source Qwen/Qwen2.5-0.5B \
  --output "${BASE}" \
  --dtype bfloat16

diffusion-llm train \
  --model "${BASE}" \
  --dataset Trelis/tiny-shakespeare \
  --mode pretrain \
  --text-field Text \
  --output "${TRAINED}" \
  --max-length 128 \
  --max-steps 25 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --bf16

diffusion-llm generate \
  --model "${TRAINED}" \
  --prompt "First Citizen:" \
  --max-new-tokens 32 \
  --steps 32 \
  --block-size 16
