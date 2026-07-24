# Method: autoregressive initialization to masked diffusion

## 1. What changes during conversion

A decoder-only AR model normally uses the factorization

```math
p(x)=\prod_i p(x_i\mid x_{<i}),
```

implemented by a lower-triangular causal attention mask. Masked diffusion needs
each noised position to use information on both sides:

```math
p_\theta(x_{0,i}\mid x_t).
```

The converter keeps the tokenizer, embeddings, transformer blocks, LM head,
and pretrained weights. It changes the model configuration and replaces the
causal mask construction with a bidirectional padding mask. A dedicated mask
token is added. If this expands the vocabulary, both input embeddings and the
LM head are resized with mean-based initialization.

The converted model is only an initialization. Its weights were optimized for
left-to-right prediction and have not yet learned to use future context or to
denoise a mask token.

## 2. Forward corruption

Let $x_0$ be a clean sequence and $t\in[\epsilon,1]$ a sampled time. The
linear schedule is

```math
\alpha(t)=1-t.
```

At each trainable position $i$, independently sample:

```math
x_{t,i} =
\begin{cases}
x_{0,i} & \text{with probability } \alpha(t),\\
\texttt{[MASK]} & \text{with probability } 1-\alpha(t)=t.
\end{cases}
```

SFT prompt tokens use label `-100`. They remain clean conditioning context and
are never selected for corruption or loss. Padding is likewise excluded.

The implementation forces at least one mask in every batch row that has a
target. This avoids zero-loss examples when $t$ is small or the response is
short.

## 3. Training objective

For a continuous-time masked diffusion model, the schedule weight is

```math
w(t)=\frac{-\alpha'(t)}{1-\alpha(t)}.
```

With the linear schedule, $w(t)=1/t$. The estimator used here is:

```math
\mathcal{L}
=
\frac{1}{N}
\sum_{i\in M_t}
w(t)\,\mathrm{CE}\left(
p_\theta(\cdot\mid x_t)_i,
x_{0,i}
\right),
```

where $M_t$ contains corrupted target positions and $N$ is the number of
non-padding target tokens. `--loss-weighting uniform` is available as a useful
classroom ablation but is not the schedule-weighted MDLM objective.

No right shift is applied: logits at position $i$ predict the clean token at
position $i$. That is different from causal language modeling.

## 4. Time and position encoding

Diffusion time is not an input to the network. It controls the sampled mask
probability and schedule weight in the loss, but there is no learned or
sinusoidal time embedding and no time-conditioned normalization layer. The
denoiser sees the corrupted token sequence and can infer its effective noise
level from the mask pattern.

The source model's rotary position embeddings remain unchanged. They encode
each token's sequence position, not diffusion time.

## 5. Reverse process

Generation begins with:

```text
prompt tokens | MASK MASK MASK ... MASK
```

At every denoising step:

1. run one bidirectional forward pass;
2. predict a token for every unresolved mask;
3. suppress invalid special tokens and apply optional top-k/top-p filtering;
4. sample categorically with multinomial or Gumbel-max sampling;
5. score predictions by probability, top-two margin, normalized entropy,
   left-to-right position, random order, or UNCODE calibration;
6. commit a fixed-count or confidence-thresholded subset; and
7. optionally remask low-confidence committed response tokens and predict them
   again using newer bidirectional context.

The deterministic reveal allocator divides the remaining masks across the
remaining iterations and guarantees that every mask is filled. A threshold
schedule may commit more tokens when the model is confident, but always falls
back to the number needed to finish within its block budget.

### Categorical sampling

The default non-greedy sampler uses chunked float64 multinomial sampling.
Chunking avoids materializing float64 probabilities or Gumbel noise for every
position in a long output simultaneously. Float32, Gumbel-max, top-k, and
nucleus sampling remain explicit ablations.

### Genuine remasking

The model was trained to reconstruct arbitrary masked subsets of the assistant
response while preserving the prompt. Consequently, an inference sampler can
turn a committed response token back into `[MASK]` and ask the same checkpoint
to reconstruct it under newer context. Prompt and system positions are never
eligible.

Two confidence sources are supported:

- `confidence` remembers the candidate probability at commitment time;
- `rescore` selects a low-confidence candidate pool, masks that pool in a
  probe pass, and measures the probability of each old token under the current
  context.

With `remask-accept=improve`, the revision forward evaluates both the old and
new token under the same masked input and retains the old token when it remains
more probable. Revision counts and cooldowns prevent oscillation.

This is training-free self-correction, but it has a limitation: visible tokens
were clean ground truth during training and may be wrong during generation.
Later remasking-aware training can explicitly expose the model to plausible
wrong visible tokens and learn a correction policy.

### Classifier-free guidance

Optional guidance combines conditional and prompt-masked logits:

```math
\ell_\mathrm{guided}
=
\ell_\mathrm{conditional}
+
w\left(\ell_\mathrm{conditional}-\ell_\mathrm{unconditional}\right).
```

This costs a second model forward per iteration. Current SFT checkpoints did
not use condition dropout, so guidance strength must be validated rather than
assumed to help.

### EOS and length

EOS may be prevented before `min-new-tokens` and may require repeated
predictions at the same unresolved position. Once EOS is committed and every
earlier response position is resolved, later canvas positions are padded and
skipped. EOS can be made remaskable for research experiments, but doing so
disables safe early termination.

## 6. Blockwise generation

Pure diffusion over a long output can be hard for a small model. The sampler
therefore supports blocks. It fully denoises one block before moving right:

```text
prompt | block 1 | block 2 | block 3
         denoise   still     still
                   masked    masked
```

Within a block, attention remains bidirectional. Earlier completed blocks form
fixed context. This interpolates between whole-span diffusion and an
autoregressive ordering over blocks.

## 7. What students should measure

- AR initialization versus random initialization.
- Schedule-weighted versus uniform loss.
- Whole-span versus blockwise generation.
- Probability, margin, entropy, UNCODE, and random commitment.
- Fixed-count versus confidence-threshold commitment.
- No remasking versus stored-confidence and rescore remasking.
- CFG strength at matched NFE.
- Number of denoising steps at fixed block size.
- Full fine-tuning versus LoRA.

Loss alone is insufficient. Compare mask reconstruction accuracy on held-out
text and manually inspect coherence, repetition, prompt following, and the
stability of tokens across denoising steps.
