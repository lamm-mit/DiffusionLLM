# End-to-end training recipes

This guide builds a 2,000,000-example chat mixture and provides two 1,024-token
training tracks: an exact legacy-compatible continuation of the published
Qwen2.5-1.5B diffusion checkpoint, and the recommended new block-shift model
with inference-aligned training. Both tracks support Hub upload, deterministic
evaluation, system-prompted generation, and denoising animations.

The recommended curriculum is:

1. start from `lamm-mit/qwen2.5-1.5b-diffusion-ultrachat`, which has already
   learned the basic masked-diffusion objective at length 512;
2. train once through a broader 2M-example instruction mixture at length 1024;
3. optionally continue at a lower learning rate on a focused domain dataset;
4. evaluate checkpoints on a fixed prompt suite before doing more epochs.

The earlier UltraChat run saw 207,865 conversation rows for three epochs, or
about 624k example presentations. With effective batch 128, it performed about
4,872 optimizer updates. The recipe below sees 2M mixture rows once, uses the
same effective batch, and performs about 15,625 optimizer updates. Its lower
learning rate is intentional because it starts from an already trained
diffusion model.

## 1. Install

On a normal Linux CUDA workstation:

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl git python3 python3-dev

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone https://github.com/lamm-mit/DiffusionLLM.git
cd DiffusionLLM
uv sync --python 3.12
uv run diffusion-llm doctor
```

Authenticate once with a Hugging Face token that can create and update model
and dataset repositories:

```bash
uv run hf auth login
```

For an NVIDIA DGX Spark, install its CUDA-compatible PyTorch wheel before the
package and use `--no-sync` so `uv` does not replace that wheel:

```bash
git clone https://github.com/lamm-mit/DiffusionLLM.git
cd DiffusionLLM

uv venv --python python3
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install -e .
uv run --no-sync diffusion-llm doctor
uv run --no-sync hf auth login
```

Use `uv run --no-sync` instead of `uv run` in the remaining commands when
using this DGX Spark installation.

## 2. Build and publish the 2M-example mixture

The mixture combines the full training splits of:

| Source | Approximate train rows | License recorded per row |
| --- | ---: | --- |
| UltraChat 200k | 207,865 | MIT |
| Smol-Magpie-Ultra | 409,537 | Apache-2.0 |
| Smol-Constraints | 34,424 | Apache-2.0 |
| Smol-Rewrite | 53,342 | Apache-2.0 |
| Smol-Summarize | 96,356 | Apache-2.0 |
| Dolci-Instruct-SFT-No-Tools | sampled to reach 2M | ODC-BY-1.0 |

The exact count is computed after structural filtering, so the output contains
exactly 2,000,000 training rows even if a source is revised. The included
manifest requests 3,000 examples across the core test splits and 2,000
disjoint Dolci examples. All rows use only `messages`, `source`, and `license`
columns.

Large dataset operations need substantial cache space. On a server, point the
Hugging Face cache at a large local or scratch filesystem before running:

```bash
export HF_HOME=/path/with/at-least-30GB-free/huggingface
```

Create the `chatmix_2m` configuration in
`lamm-mit/diffusion-chat-mixture-1024` with the packaged builder:

```bash
uv run diffusion-llm build-mixture \
  --manifest examples/chatmix_2m.json \
  --target-train-rows 2000000 \
  --save-to-disk artifacts/diffusion-chat-mixture-1024-chatmix-2m \
  --push-to-hub \
  --hub-dataset-id lamm-mit/diffusion-chat-mixture-1024 \
  --hub-config-name chatmix_2m \
  --num-proc 16 \
  --upload-num-proc 1
```

`--num-proc` controls parallel filtering and normalization.
`--upload-num-proc` separately controls local serialization and Hub upload and
defaults to one, avoiding `<stdin>`/multiprocessing failures. The local copy is
written before upload, so an interrupted upload can be retried without
reconstructing the 2M rows.

Retry only the upload with:

```bash
uv run diffusion-llm upload-mixture \
  --dataset artifacts/diffusion-chat-mixture-1024-chatmix-2m \
  --hub-dataset-id lamm-mit/diffusion-chat-mixture-1024 \
  --hub-config-name chatmix_2m \
  --num-proc 1
```

The JSON manifest accepts these fields for each entry in `sources`:

- `dataset` (required), `config`, `train_split`, and `validation_split`;
- `license` and optional `label`, written into every output row;
- `max_train_rows`, which caps a source before mixing;
- `validation_rows`, sampled from `validation_split`, or reserved disjointly
  from training when `validation_split` is `null`;
- `fill: true` on at most one source, which supplies the rows needed to hit
  `--target-train-rows` exactly;
- `preserve_original_source`, which appends upstream provenance to `source`;
  and
- `revision`, which pins a source to a Hub revision.

If no source has `fill: true`, the combined fixed sources are deterministically
downsampled to the requested total. The command rejects an undersized mixture
instead of silently returning fewer rows. Use `--validation-rows-per-source`
or `--max-validation-rows` for command-line validation overrides.

<details>
<summary>Expanded inline implementation (the CLI above is recommended)</summary>

```bash
uv run python - <<'PY'
from collections import Counter

from datasets import (
    DatasetDict,
    Features,
    List,
    Value,
    concatenate_datasets,
    load_dataset,
)

TARGET_TRAIN_ROWS = 2_000_000
DOLCI_VALIDATION_ROWS = 2_000
CORE_VALIDATION_ROWS_PER_SOURCE = 600
SEED = 42
NUM_PROC = 8
DATASET_REPO = "lamm-mit/diffusion-chat-mixture-1024"
CONFIG_NAME = "chatmix_2m"

FEATURES = Features(
    {
        "messages": List(
            {
                "role": Value("string"),
                "content": Value("string"),
            }
        ),
        "source": Value("string"),
        "license": Value("string"),
    }
)

CORE_SOURCES = [
    {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "config": None,
        "train_split": "train_sft",
        "eval_split": "test_sft",
        "source": "HuggingFaceH4/ultrachat_200k",
        "license": "MIT",
    },
    {
        "dataset": "HuggingFaceTB/smoltalk",
        "config": "smol-magpie-ultra",
        "train_split": "train",
        "eval_split": "test",
        "source": "HuggingFaceTB/smoltalk:smol-magpie-ultra",
        "license": "Apache-2.0",
    },
    {
        "dataset": "HuggingFaceTB/smoltalk",
        "config": "smol-constraints",
        "train_split": "train",
        "eval_split": "test",
        "source": "HuggingFaceTB/smoltalk:smol-constraints",
        "license": "Apache-2.0",
    },
    {
        "dataset": "HuggingFaceTB/smoltalk",
        "config": "smol-rewrite",
        "train_split": "train",
        "eval_split": "test",
        "source": "HuggingFaceTB/smoltalk:smol-rewrite",
        "license": "Apache-2.0",
    },
    {
        "dataset": "HuggingFaceTB/smoltalk",
        "config": "smol-summarize",
        "train_split": "train",
        "eval_split": "test",
        "source": "HuggingFaceTB/smoltalk:smol-summarize",
        "license": "Apache-2.0",
    },
]

ALLOWED_ROLES = {"system", "user", "assistant"}


def is_valid(row):
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    if not isinstance(messages[-1], dict):
        return False
    if messages[-1].get("role") != "assistant":
        return False
    return all(
        isinstance(message, dict)
        and message.get("role") in ALLOWED_ROLES
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for message in messages
    )


def clean_messages(messages):
    return [
        {
            "role": str(message["role"]),
            "content": str(message["content"]).strip(),
        }
        for message in messages
    ]


def normalize_fixed(row, source_name, license_name):
    return {
        "messages": clean_messages(row["messages"]),
        "source": source_name,
        "license": license_name,
    }


def normalize_dolci(row):
    original_source = str(row.get("source") or "unknown")
    return {
        "messages": clean_messages(row["messages"]),
        "source": f"allenai/Dolci-Instruct-SFT-No-Tools:{original_source}",
        "license": "ODC-BY-1.0",
    }


def clean_fixed(dataset, source_name, license_name):
    original_columns = dataset.column_names
    return dataset.filter(
        is_valid,
        num_proc=NUM_PROC,
        desc=f"Validating {source_name}",
    ).map(
        normalize_fixed,
        fn_kwargs={"source_name": source_name, "license_name": license_name},
        remove_columns=original_columns,
        features=FEATURES,
        num_proc=NUM_PROC,
        desc=f"Normalizing {source_name}",
    )


core_train_parts = []
core_validation_parts = []

for spec in CORE_SOURCES:
    train_part = load_dataset(
        spec["dataset"],
        spec["config"],
        split=spec["train_split"],
    )
    train_part = clean_fixed(train_part, spec["source"], spec["license"])
    core_train_parts.append(train_part)

    validation_part = load_dataset(
        spec["dataset"],
        spec["config"],
        split=spec["eval_split"],
    )
    validation_part = clean_fixed(
        validation_part,
        spec["source"],
        spec["license"],
    )
    validation_part = validation_part.select(
        range(min(CORE_VALIDATION_ROWS_PER_SOURCE, len(validation_part)))
    )
    core_validation_parts.append(validation_part)

core_train = concatenate_datasets(core_train_parts)
dolci_train_rows = TARGET_TRAIN_ROWS - len(core_train)
if dolci_train_rows <= 0:
    raise RuntimeError(
        f"Core sources already contain {len(core_train):,} valid rows; "
        "increase TARGET_TRAIN_ROWS."
    )

dolci = load_dataset(
    "allenai/Dolci-Instruct-SFT-No-Tools",
    split="train",
)
dolci_columns = dolci.column_names
dolci = dolci.filter(
    is_valid,
    num_proc=NUM_PROC,
    desc="Validating Dolci",
).map(
    normalize_dolci,
    remove_columns=dolci_columns,
    features=FEATURES,
    num_proc=NUM_PROC,
    desc="Normalizing Dolci",
).shuffle(seed=SEED)

required_dolci_rows = dolci_train_rows + DOLCI_VALIDATION_ROWS
if len(dolci) < required_dolci_rows:
    raise RuntimeError(
        f"Need {required_dolci_rows:,} valid Dolci rows, found {len(dolci):,}."
    )

dolci_train = dolci.select(range(dolci_train_rows))
dolci_validation = dolci.select(
    range(dolci_train_rows, required_dolci_rows)
)

train = concatenate_datasets([core_train, dolci_train]).shuffle(seed=SEED)
validation = concatenate_datasets(
    [*core_validation_parts, dolci_validation]
).shuffle(seed=SEED)

assert len(train) == TARGET_TRAIN_ROWS
mixture = DatasetDict({"train": train, "validation": validation})

print(mixture)
print("train sources:", Counter(train["source"]))
print("validation sources:", Counter(validation["source"]))

mixture.push_to_hub(
    DATASET_REPO,
    config_name=CONFIG_NAME,
    max_shard_size="500MB",
)
PY
```

</details>

Do not label the resulting dataset repository as Apache-2.0 only. It is a
mixed-license collection and includes ODC-BY-1.0 data; retain the per-row
`license` and `source` fields and comply with every upstream dataset's terms.

Check the uploaded configuration before starting an expensive job:

```bash
uv run python - <<'PY'
from collections import Counter
from datasets import load_dataset

dataset = load_dataset(
    "lamm-mit/diffusion-chat-mixture-1024",
    "chatmix_2m",
)
print(dataset)
print(Counter(dataset["train"]["license"]))
print(dataset["train"][0]["messages"])
PY
```

## 3. Run a two-step smoke test

Use the same model, dataset configuration, context length, precision, and GPU
as the real run. This catches authentication, schema, tokenization, and memory
problems without committing to the full job:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
uv run diffusion-llm train \
  --model lamm-mit/qwen2.5-1.5b-diffusion-ultrachat \
  --dataset lamm-mit/diffusion-chat-mixture-1024 \
  --dataset-config chatmix_2m \
  --train-split train \
  --eval-split validation \
  --mode sft \
  --output artifacts/chatmix-1024-smoke \
  --max-length 1024 \
  --max-train-samples 256 \
  --max-eval-samples 32 \
  --max-steps 2 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-5 \
  --mask-prompt-loss \
  --num-proc 16 \
  --gradient-checkpointing \
  --bf16 \
  --report-to none
```

Because `CUDA_VISIBLE_DEVICES=1` exposes only the physical RTX 6000 Ada, that
GPU is named `cuda:0` inside the process. The RTX 5000 Ada remains available to
another terminal as physical GPU 0.

## 4. Train the legacy-compatible 1,024-token model

This run starts from the existing 512-token diffusion model. Qwen rotary
positions support length 1024 without changing the architecture. The longer
context comes from `--max-length 1024`; generation `--block-size` is independent
and is not a training parameter.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
uv run diffusion-llm train \
  --model lamm-mit/qwen2.5-1.5b-diffusion-ultrachat \
  --dataset lamm-mit/diffusion-chat-mixture-1024 \
  --dataset-config chatmix_2m \
  --train-split train \
  --eval-split validation \
  --mode sft \
  --output artifacts/qwen2.5-1.5b-diffusion-chatmix-1024-2m \
  --max-length 1024 \
  --epochs 1 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 64 \
  --learning-rate 1e-5 \
  --warmup-steps 0.03 \
  --weight-decay 0.1 \
  --mask-prompt-loss \
  --max-eval-samples 512 \
  --logging-steps 10 \
  --save-steps 1000 \
  --eval-steps 1000 \
  --save-total-limit 3 \
  --num-proc 16 \
  --gradient-checkpointing \
  --bf16 \
  --report-to none \
  --push-to-hub \
  --hub-model-id lamm-mit/qwen2.5-1.5b-diffusion-chatmix-1024-2m \
  --hub-strategy every_save
```

To report this run to W&B, first install the optional SDK and authenticate:

```bash
uv pip install wandb
uv run --no-sync wandb login
```

Then replace `--report-to none` in the training command with:

```bash
  --report-to wandb \
  --wandb-project DiffusionLLM \
  --run-name qwen2.5-1.5b-chatmix-1024-2m
```

The project defaults to `DiffusionLLM`, and the run name defaults to the output
directory basename. A process launched with `--report-to none` cannot be
switched to live W&B reporting without restarting it.

Per-device batch 2 and gradient accumulation 64 give effective batch 128.
There are approximately 15,625 optimizer steps in one epoch before examples
whose assistant target is completely truncated are removed. Training examples
sample a new continuous diffusion time and a new random mask pattern on every
visit, so one epoch still covers varying corruption levels across the 2M rows.

If the process is interrupted, resume from the latest complete checkpoint by
rerunning the same command with:

```bash
  --resume-from-checkpoint \
  artifacts/qwen2.5-1.5b-diffusion-chatmix-1024-2m/checkpoint-N
```

Do not change the batch, accumulation, learning-rate schedule, dataset order,
or output directory while resuming an optimizer state. DiffusionLLM compares
the new command with the saved `training_config.json` and rejects
training-critical drift before starting. Logging/save cadence, Hub settings,
and run labels may change. `--allow-resume-mismatch` exists for controlled
research only; it does not make a changed objective, optimizer schedule, batch,
or training horizon mathematically equivalent.

To start from the original AR model instead, first convert it and replace the
training command's `--model` value. This is a colder start and is expected to
need more optimization than continuing the published diffusion checkpoint:

```bash
uv run diffusion-llm convert \
  --source Qwen/Qwen2.5-1.5B-Instruct \
  --output artifacts/qwen2.5-1.5b-diffusion-base \
  --dtype bfloat16
```

## 5. Train the recommended block-shift model

This is the new model/trainer path. It deliberately starts from the original
AR checkpoint because shifted prediction preserves the AR next-token alignment.
Do not change a mature same-position diffusion checkpoint to shifted prediction
mid-run and assume the optimizer state remains comparable.

Create a block-causal, shifted, time-conditioned initialization:

```bash
uv run --no-sync diffusion-llm convert \
  --source Qwen/Qwen2.5-1.5B-Instruct \
  --output artifacts/qwen2.5-1.5b-diffusion-block-shift-base \
  --dtype bfloat16 \
  --prediction-parameterization shifted \
  --attention-pattern block-causal \
  --time-conditioning additive \
  --time-embedding-dim 256
```

First run the complete objective for two steps. Keep every method flag identical
to the long run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
uv run --no-sync diffusion-llm train \
  --model artifacts/qwen2.5-1.5b-diffusion-block-shift-base \
  --dataset lamm-mit/diffusion-chat-mixture-1024 \
  --dataset-config chatmix_2m \
  --train-split train \
  --eval-split validation \
  --mode sft \
  --output artifacts/qwen2.5-1.5b-block-shift-v3-smoke \
  --max-length 1024 \
  --max-train-samples 512 \
  --max-eval-samples 32 \
  --max-steps 2 \
  --batch-size 4 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --learning-rate 3e-5 \
  --objective block-hybrid \
  --prediction-parameterization shifted \
  --attention-pattern block-causal \
  --train-block-sizes 8,16,32,64 \
  --full-mdlm-ratio 0.20 \
  --ar-loss-weight 0.10 \
  --time-sampling stratified \
  --mask-sampling progressive \
  --progressive-stages 8 \
  --progressive-mask-probability 0.50 \
  --loss-normalization sequence \
  --condition-dropout 0.10 \
  --condition-dropout-mode mask \
  --mask-tail-augmentation 0.10 \
  --mask-tail-max-tokens 64 \
  --mask-consistency-weight 0.05 \
  --time-conditioning additive \
  --time-embedding-dim 256 \
  --self-conditioning-probability 0.25 \
  --draft-commit-probability 0.25 \
  --draft-loss-weight 0.10 \
  --mask-prompt-loss \
  --num-proc 16 \
  --gradient-checkpointing \
  --bf16 \
  --report-to none
```

If the smoke test fits, launch the full DGX Spark run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
uv run --no-sync diffusion-llm train \
  --model artifacts/qwen2.5-1.5b-diffusion-block-shift-base \
  --dataset lamm-mit/diffusion-chat-mixture-1024 \
  --dataset-config chatmix_2m \
  --train-split train \
  --eval-split validation \
  --mode sft \
  --output artifacts/qwen2.5-1.5b-diffusion-chatmix-1024-2m-block-shift-v3 \
  --max-length 1024 \
  --epochs 1 \
  --batch-size 4 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 3e-5 \
  --warmup-steps 500 \
  --weight-decay 0.1 \
  --objective block-hybrid \
  --prediction-parameterization shifted \
  --attention-pattern block-causal \
  --train-block-sizes 8,16,32,64 \
  --full-mdlm-ratio 0.20 \
  --ar-loss-weight 0.10 \
  --time-sampling stratified \
  --mask-sampling progressive \
  --progressive-stages 8 \
  --progressive-mask-probability 0.50 \
  --loss-normalization sequence \
  --condition-dropout 0.10 \
  --condition-dropout-mode mask \
  --mask-tail-augmentation 0.10 \
  --mask-tail-max-tokens 64 \
  --mask-consistency-weight 0.05 \
  --time-conditioning additive \
  --time-embedding-dim 256 \
  --self-conditioning-probability 0.25 \
  --draft-commit-probability 0.25 \
  --draft-loss-weight 0.10 \
  --mask-prompt-loss \
  --max-eval-samples 512 \
  --logging-steps 10 \
  --save-steps 1000 \
  --eval-steps 1000 \
  --save-total-limit 5 \
  --num-proc 16 \
  --gradient-checkpointing \
  --bf16 \
  --report-to wandb \
  --wandb-project DiffusionLLM \
  --run-name qwen2.5-1.5b-chatmix-block-shift-v3 \
  --push-to-hub \
  --hub-model-id lamm-mit/qwen2.5-1.5b-diffusion-chatmix-1024-2m-block-shift-v3 \
  --hub-strategy every_save
```

Batch 4 and accumulation 32 give effective batch 128 and approximately 15,625
optimizer updates over 2M examples. This run is intentionally more expensive
than legacy MDLM: progressive and draft states sometimes add a no-gradient
proposal forward, while tail consistency sometimes adds another reference
forward. Proposal work is shared when progressive masking and
self-conditioning occur together.

The new pieces have distinct purposes:

| Mechanism | Purpose | Primary cost |
| --- | --- | --- |
| shifted + block-causal | retain AR next-token skill while denoising a block | no extra forward |
| block-hybrid | match block generation while retaining some full-span ability | no extra forward |
| stratified/exact-count base | cover corruption levels more evenly | negligible |
| progressive masks | train on confidence-ordered intermediate sampler states | proposal forward |
| condition dropout | make CFG unconditional states familiar | negligible |
| masked-tail consistency | resist long unresolved generation canvases | padding + occasional reference |
| additive time encoding | expose the active masked fraction explicitly | small MLP |
| draft self-conditioning | learn with plausible visible model predictions | proposal forward |

`--mask-tail-augmentation` pads training batches to `--max-length`, so it can
raise memory use even when the sampled augmentation probability is small. If
the smoke test is out of memory, first set tail augmentation and consistency to
zero; next reduce batch 4 to batch 2 and increase accumulation from 32 to 64.
Keep effective batch 128.

All advanced features are opt-in. Training inherits architecture and objective
metadata from the checkpoint; an old checkpoint therefore still runs the exact
legacy path when these flags are omitted. `training_config.json` records every
resolved flag and `run_manifest.json` records the method, dataset fingerprints,
effective batch, git revision, software/CUDA environment, and completion
status.

## 6. Test a checkpoint on the other GPU

Run this while training uses physical GPU 1. The test process exposes physical
GPU 0, which is again named `cuda:0` inside that process:

```bash
MODEL=lamm-mit/qwen2.5-1.5b-diffusion-chatmix-1024-2m
SYSTEM_PROMPT='Answer accurately, explain the mechanism, and state important assumptions.'
PROMPT='Explain why masked diffusion can revise several tokens in parallel, and contrast it with autoregressive decoding.'

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
uv run diffusion-llm generate \
  --model "$MODEL" \
  --system-prompt "$SYSTEM_PROMPT" \
  --prompt "$PROMPT" \
  --chat-template \
  --max-new-tokens 512 \
  --steps 512 \
  --block-size 16 \
  --temperature 0.2 \
  --device cuda:0 \
  --gif artifacts/chatmix-1024-denoising.gif \
  --gif-frame-duration-ms 100
```

The terminal progress bar reports the actual denoising forward passes. With
512 output tokens and block size 16, there are 32 blocks and at most 16 useful
reveal iterations per block, so 512 steps fully uses this configuration. A
larger `--steps` value is capped by the reveal schedule and does not add useful
passes.

For the new block-shift checkpoint, no inference architecture flags are needed:
the loader reads shifted prediction, block attention, and additive time
conditioning from `config.json`. Condition dropout makes modest CFG strengths
a trained use case, and draft training makes revision states less out of
distribution:

```bash
MODEL=lamm-mit/qwen2.5-1.5b-diffusion-chatmix-1024-2m-block-shift-v3

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
uv run --no-sync diffusion-llm generate \
  --model "$MODEL" \
  --system-prompt "Answer accurately and explain the mechanism." \
  --prompt "Why can a masked diffusion language model revise an earlier token?" \
  --chat-template \
  --max-new-tokens 512 \
  --steps 768 \
  --max-nfe 1024 \
  --block-size 16 \
  --temperature 0.2 \
  --top-p 0.95 \
  --commit-policy uncode \
  --cfg-scale 0.5 \
  --remask-policy confidence \
  --remask-rate 0.05 \
  --max-remasks-per-step 2 \
  --max-revisions-per-token 2 \
  --remask-window previous \
  --remask-accept improve \
  --device cuda:0 \
  --gif artifacts/block-shift-v3-denoising.gif \
  --gif-frame-duration-ms 100
```

Compare `--cfg-scale 0`, `0.5`, `1.0`, and `1.5`, and compare remasking with
`--remask-policy none`; the new training makes those states familiar but does
not guarantee that every sampler policy improves every task.

## 7. Evaluate denoising and fixed held-out generations

First compare model reconstruction independently of sampler policy:

```bash
MODEL=lamm-mit/qwen2.5-1.5b-diffusion-chatmix-1024-2m-block-shift-v3

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
uv run --no-sync diffusion-llm evaluate-denoising \
  --model "$MODEL" \
  --dataset lamm-mit/diffusion-chat-mixture-1024 \
  --dataset-config chatmix_2m \
  --split validation \
  --mode sft \
  --max-length 1024 \
  --num-samples 512 \
  --batch-size 4 \
  --mask-probabilities 0.15,0.30,0.50,0.70,0.90 \
  --block-size 16 \
  --calibration-bins 10 \
  --dtype bfloat16 \
  --device cuda:0 \
  --seed 1729 \
  --output artifacts/block-shift-v3-denoising-metrics.json
```

This command masks an exact deterministic count per sequence, so repeated
checkpoint comparisons use the same corrupted tokens. It reports NLL,
perplexity, top-1 clean-token reconstruction accuracy, confidence, and ECE at
each mask fraction. A block-causal checkpoint evaluates deterministic active
blocks of `--block-size`; a full checkpoint evaluates the whole target span.

The periodic `eval_loss` already measures the diffusion objective on 512
held-out rows. Because evaluation samples new diffusion times and corruption
masks, individual loss measurements are noisy. Complement them with a fixed
generation suite:

```bash
MODEL=artifacts/qwen2.5-1.5b-diffusion-chatmix-1024-2m-from-base/checkpoint-4400

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
uv run --no-sync diffusion-llm evaluate \
  --model "$MODEL" \
  --dataset lamm-mit/diffusion-chat-mixture-1024 \
  --dataset-config chatmix_2m \
  --split validation \
  --num-samples 32 \
  --batch-size 2 \
  --max-total-tokens 1024 \
  --max-new-tokens 512 \
  --steps 512 \
  --block-size 8 \
  --temperature 0 \
  --dtype bfloat16 \
  --device cuda:0 \
  --output artifacts/checkpoint-4400-heldout.jsonl
```

The evaluator shuffles deterministically with `--seed 42` by default. It uses
every message before the final assistant message as the prompt, including any
system message, and treats that final assistant message as the reference.
`--max-total-tokens 1024` keeps the prompt plus generation canvas within the
training sequence length and reports how many longer rows were skipped.

Repeat the command for checkpoints near steps 4,400 and 8,000 and for the final
model, changing only `--model` and `--output`. Compare the detailed JSONL files
blindly. Exact match and lexical token F1 are diagnostics, not reliable measures
of semantic quality for open-ended answers; use human inspection or a blinded
LLM judge for the primary comparison.

## 8. Continue on the scientific-design dataset

After the broad 1024-token stage, specialize at a lower learning rate. Three
epochs over the 9k-example scientific split are a sensible first run; compare
held-out generations at each checkpoint before deciding whether more epochs
help.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
uv run diffusion-llm train \
  --model lamm-mit/qwen2.5-1.5b-diffusion-chatmix-1024-2m \
  --dataset lamm-mit/scientific-sft-grpo-data \
  --dataset-config scientific_design_sft_L_messages \
  --train-split train \
  --eval-split validation \
  --mode sft \
  --output artifacts/qwen2.5-1.5b-diffusion-scientific-1024 \
  --max-length 1024 \
  --epochs 3 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 5e-6 \
  --warmup-steps 0.03 \
  --weight-decay 0.1 \
  --mask-prompt-loss \
  --max-eval-samples 256 \
  --logging-steps 10 \
  --save-steps 100 \
  --eval-steps 100 \
  --save-total-limit 3 \
  --num-proc 16 \
  --gradient-checkpointing \
  --bf16 \
  --report-to none \
  --push-to-hub \
  --hub-model-id lamm-mit/qwen2.5-1.5b-diffusion-scientific-1024 \
  --hub-strategy every_save
```

Test it with the same system role used by the scientific `messages` subset:

```bash
MODEL=lamm-mit/qwen2.5-1.5b-diffusion-scientific-1024
SYSTEM_PROMPT='Solve the self-contained scientific problem-solving task. Develop several distinct candidate ideas, identify the governing scientific principles and constraints, synthesize the strongest direction, and give a concise final answer.'
PROMPT='Develop mechanistic hypotheses for why a porous catalyst loses activity during repeated wet-dry cycles, distinguish transport from chemical-deactivation explanations, and propose one decisive experiment.'

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
uv run diffusion-llm generate \
  --model "$MODEL" \
  --system-prompt "$SYSTEM_PROMPT" \
  --prompt "$PROMPT" \
  --chat-template \
  --max-new-tokens 512 \
  --steps 512 \
  --block-size 16 \
  --temperature 0.2 \
  --device cuda:0 \
  --gif artifacts/scientific-porous-catalyst.gif \
  --gif-frame-duration-ms 100
```
