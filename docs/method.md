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
3. suppress the mask and padding tokens as output candidates;
4. score predictions by model confidence (or randomly for an ablation);
5. commit the scheduled highest-confidence subset; and
6. leave low-confidence positions masked for another prediction.

The deterministic reveal allocator divides the remaining masks across the
remaining steps and guarantees that every mask is filled.

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
- Confidence versus random remasking.
- Number of denoising steps at fixed block size.
- Full fine-tuning versus LoRA.

Loss alone is insufficient. Compare mask reconstruction accuracy on held-out
text and manually inspect coherence, repetition, prompt following, and the
stability of tokens across denoising steps.
