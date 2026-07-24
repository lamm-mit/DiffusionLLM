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
and pretrained weights. A dedicated mask token is added. If this expands the
vocabulary, both input embeddings and the LM head are resized with mean-based
initialization. Two attention/prediction parameterizations are supported:

- `same-position` plus `full-bidirectional`: every real query attends to every
  real key, and logit $i$ reconstructs token $i$;
- `shifted` plus `block-causal`: completed-prefix representations remain
  causal, active-block queries see the complete prefix and active block, and
  raw logit $i-1$ reconstructs token $i$. The final prefix query is extended
  into the active region so it can predict the block's first token.

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

The legacy implementation forces at least one mask in every batch row that has
a target. This avoids zero-loss examples when $t$ is small or the response is
short. The v2 engine additionally supports stratified $t$ samples and uniformly
sampled exact mask counts. Exact counts replace Monte Carlo Bernoulli variance
with a Rao-Blackwellized masked-token mean.

For block training, one target block is sampled per row. Earlier tokens remain
visible, the active block receives the sampled corruption, and later target
blocks become masks. Multiple `train-block-sizes` expose the same checkpoint to
several inference granularities.

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

The compatibility objective has no right shift. The shifted objective instead
uses the pretrained AR alignment: raw logit $i-1$ predicts clean token $i$.
An optional clean-prefix AR loss helps retain the source model's next-token
skill while block denoising is learned.

### Inference-aligned corruption

Progressive masking approximates intermediate sampler states:

1. fully mask the active target region;
2. obtain a no-gradient proposal;
3. measure the proposal probability assigned to every clean target token;
4. sample one of $K$ reveal phases;
5. reveal the corresponding number of highest-confidence clean tokens; and
6. train on the targets that remain masked.

This is a stateless PUMA-inspired rollout, not PUMA's stateful batch-streaming
implementation. It can be mixed with exact-count states to control compute.

Three other augmentations address train/inference mismatch:

- SFT condition dropout masks or pads the entire prompt on a fraction of rows,
  supplying the unconditional branch required by classifier-free guidance.
- Mask-tail augmentation appends unresolved canvas positions beyond the real
  sequence. A KL consistency term matches target predictions with and without
  that tail.
- Draft self-conditioning runs a no-gradient proposal, makes a random subset
  of proposed tokens visible, retains masked-token loss on the rest, and adds a
  smaller clean-target correction loss at draft positions.

## 4. Time and position encoding

By default, diffusion time is not an input to the network. It controls
corruption and weighting, while the denoiser infers noise from the mask
pattern. This reproduces the original method and follows the observation that
an absorbing-mask state already exposes its approximate noise level.

The opt-in additive conditioner uses the actual active-region mask fraction
$\tilde t=|M|/|T|$. It forms sinusoidal features, applies a two-layer MLP, and
adds the result to every input embedding:

```math
h_i^{(0)} = e(x_{t,i}) + \mathrm{MLP}(\mathrm{sinusoid}(\tilde t)).
```

The MLP's final projection is initialized to zero. A newly enabled conditioner
therefore changes no logits until training learns a nonzero projection. During
block generation, $\tilde t$ is recomputed only over the active block. CFG
conditional and unconditional forwards receive the same value so prompt masks
do not falsely change diffusion time.

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

This remains a training-free remasking policy. Draft self-conditioning reduces
the visible-token mismatch by exposing the denoiser to its own proposals, but
there is no learned token-quality/remask head. Learning that policy is the next
deliberately deferred stage.

### Classifier-free guidance

Optional guidance combines conditional and prompt-masked logits:

```math
\ell_\mathrm{guided}
=
\ell_\mathrm{conditional}
+
w\left(\ell_\mathrm{conditional}-\ell_\mathrm{unconditional}\right).
```

This costs a second model forward per iteration. Legacy checkpoints did not use
condition dropout. New checkpoints can train with `condition-dropout`, but
guidance strength must still be validated at matched NFE.

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
- Uniform versus stratified times and Bernoulli versus exact mask counts.
- Same-position/full attention versus shifted/block-causal training.
- Canonical versus progressive corruption at matched forward-pass budgets.
- Implicit versus additive time conditioning.
- No tail, tail augmentation, and tail consistency.
- No draft versus draft self-conditioning.
- Whole-span versus blockwise generation.
- Probability, margin, entropy, UNCODE, and random commitment.
- Fixed-count versus confidence-threshold commitment.
- No remasking versus stored-confidence and rescore remasking.
- CFG strength at matched NFE.
- Number of denoising steps at fixed block size.
- Full fine-tuning versus LoRA.

Stochastic training loss alone is insufficient. Use `evaluate-denoising` to
compare exact fixed corruption patterns, NLL, reconstruction accuracy,
confidence, and calibration. Then compare end-to-end generation at matched NFE
and inspect coherence, repetition, prompt following, format adherence, and
token stability across denoising steps.

The implementation is informed by
[LLaDA](https://github.com/ML-GSAI/LLaDA),
[Block Diffusion](https://arxiv.org/abs/2503.09573),
[Fast-dLLM v2](https://github.com/NVlabs/Fast-dLLM/tree/main/v2),
[PUMA](https://github.com/JaeyeonKim01/PUMA),
[Masks Can Be Distracting](https://openreview.net/forum?id=CdJwNTisx1), and
[self-conditioned masked diffusion](https://arxiv.org/abs/2604.26985). The
focused implementations here are adaptations, not claims of exact equivalence.
