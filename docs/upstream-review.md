# Review of `ZHZisZZ/dllm`

## Review scope

The repository was cloned and reviewed at commit
`ca176752fbceec49c6b4777a2c18ae88e4eb10ed` (2026-04-14), the head of its
default branch during this work. The checkout contained 204 files and used the
Apache License 2.0.

The review covered:

- root documentation, packaging, dependency pins, license, and contributor
  instructions;
- all package and example paths in the repository tree;
- reusable schedulers, trainers, samplers, evaluation adapters, collators,
  dataset loaders, model loaders, and utilities;
- the complete `dllm/pipelines/a2d` and `examples/a2d` implementations;
- A2D attention, padding-invariance, KV-cache, and BD3LM equivalence tests;
- neighboring LLaDA, LLaDA2/2.1, Dream, BERT, EditFlow, Fast-dLLM, and GRPO
  pipelines to determine what was generic and what was unrelated to this task;
  and
- Accelerate, DeepSpeed, FSDP, Slurm, preprocessing, and evaluation entry
  points.

## Repository architecture

The upstream repository is a broad diffusion-language-model toolkit:

- `dllm/core`: generic MDLM/BD3LM training, schedules, sampling, and evaluation;
- `dllm/data`: named dataset adapters and dataset-composition helpers;
- `dllm/pipelines`: model-family and algorithm-specific implementations;
- `dllm/utils`: model/tokenizer loading, PEFT, collators, chat, data, and
  visualization;
- `examples`: thin executable recipes for every supported pipeline;
- `scripts`: Accelerate configs, Slurm launchers, and tests.

That breadth is useful for research reproduction but creates a large teaching
surface: students must understand several model families, algorithms, launch
systems, data conventions, and loader fallbacks before seeing the minimal
AR-to-diffusion loop.

## A2D conversion

The A2D converter supports source `model_type` values `llama`, `qwen2`, and
`qwen3`. For each architecture it:

1. loads `AutoModelForCausalLM`;
2. creates a new configuration with an A2D-specific `model_type`;
3. creates a second target model;
4. loads the source state dict into that model; and
5. saves the target and source tokenizer.

The custom model subclasses largely copy the Hugging Face decoder forward pass
and replace causal-mask construction with a full, padding-aware mask.

The key attention test changes a future token and asserts that earlier logits
change. Additional tests compare padded and unpadded logits and validate the
special block masks used by BD3LM.

## MDLM training

The MDLM trainer:

1. samples one time $t$ per sequence;
2. masks eligible tokens independently with probability $1-\alpha(t)$;
3. predicts clean token IDs at the same positions;
4. applies either schedule or uniform weighting; and
5. normalizes over token, sequence, or batch.

The default linear schedule yields $\alpha(t)=1-t$ and loss weight
$1/t$. SFT labels use `-100` to prevent prompt corruption and loss.

## MDLM inference

The sampler appends a fixed span of masks, repeatedly predicts every masked
position, then commits a scheduled number of positions. It supports
confidence-based or random remasking, Gumbel sampling, classifier-free
guidance, infilling, blockwise generation, and denoising histories.

## BD3LM

BD3LM concatenates a noised and clean copy during training and applies a custom
mask combining block-diagonal, offset block-causal, and block-causal regions.
Its inference loop uses blockwise KV caching. It is interesting but materially
more complex and doubles the training sequence, so it was excluded from this
course-focused implementation.

## What this project retained

- AR weights as the initialization.
- Explicit bidirectional attention classes for the same three architecture
  families.
- Linear continuous-time masked corruption.
- Same-position cross-entropy on corrupted targets.
- Schedule weighting and fixed prompt conditioning.
- Iterative confidence-based unmasking.
- Blockwise generation as a bridge between pure diffusion and AR ordering.
- Attention and padding invariance as required tests.

## What this project changed

- One MDLM method rather than a multi-pipeline framework.
- One `diffusion-llm` CLI instead of separate scripts for every mode/algorithm.
- Transformers 5.13 public bidirectional-mask utilities.
- In-place class conversion, avoiding simultaneous source and target models.
- Mask-token insertion and embedding/LM-head resizing during conversion.
- A guaranteed mask for every trainable batch row.
- Numerically standard additive Gumbel sampling.
- A reveal allocator that guarantees no unresolved masks.
- Direct local-file dataset support and common SFT-schema normalization.
- Explicit full-checkpoint/LoRA loading.
- An offline tiny-checkpoint integration test.

## What was intentionally omitted

- LLaDA, LLaDA-MoE, LLaDA2/2.1, Dream, BERT-Chat, and EditFlow model stacks.
- BD3LM and specialized cache masks.
- GRPO/reward training.
- lm-evaluation-harness adapters.
- Fast-dLLM production acceleration.
- video/terminal visualization machinery.
- named dataset-specific adapters when schema normalization is sufficient.
- DeepSpeed, FSDP, Slurm, and cluster policy files.

These omissions keep the causal chain visible to students:

```text
AR weights
  -> bidirectional attention
  -> mask corruption
  -> denoising loss
  -> iterative unmasking
```
