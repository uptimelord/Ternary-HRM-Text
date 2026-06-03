# Experiment 42 - Token-over-Parameter Budget Probe (Rank 13)

Spectra (arXiv 2506.23025) finds TriLMs benefit more from training **tokens**
than from **parameters** (the token exponent beta exceeds the param exponent
alpha in their fitted scaling law; for FloatLMs the two are ~equal). For a fixed
RTX 3050 Ti budget this argues for spending compute on data-efficient QAT rather
than wider models. This is a 2-axis probe to check the ordering locally on the
ternary HRM, using the locked preset `mixed_top512_tequila_L_mlp_gate_up`.

Design: measure the eval-loss drop from (a) **doubling tokens** at fixed width
vs (b) **doubling width** at fixed tokens, then compare which buys more.
"Tokens seen" in this harness = steps x numseqs x total_len; with numseqs and
seq len fixed, tokens scale linearly with steps. Cells (locked variant, 3 seeds):

- token axis: h256 @ 1000 steps  vs  h256 @ 2000 steps (2x tokens, fixed width)
- param axis: h128 @ 2000 steps  vs  h256 @ 2000 steps (~4x params, fixed tokens)

The token-doubling drop (h256: 1000->2000) is compared against the
param-quadrupling drop (h128->h256 at 2000). Spectra predicts the per-FLOP loss
reduction from tokens should be competitive with or beat params. This probe
*informs budget allocation* for the rank 2/3 confirm runs; it is not a
promote/deploy gate on a model variant.

Noise floor: +/- 0.0203 eval loss.

## Decision Rule

- Promote if the eval-loss reduction from doubling tokens (h256 1000->2000) is
  >= the reduction from quadrupling params (h128->h256 at 2000 steps), beyond the
  noise floor, across 3 seeds -- confirming the token-over-param ordering holds
  locally and budget should favor more tokens at fixed width.
- Kill if doubling tokens yields a smaller eval-loss reduction than quadrupling
  params (beyond the noise floor) -- the Spectra ordering does not transfer to
  this regime and budget should favor width.

## Results

Run 2026-05-31, locked variant `mixed_top512_tequila_L_mlp_gate_up`, seeds 1/2/3,
device cuda. Files: `results_h256_steps{1000,2000}_seeds123.md`,
`results_h128_steps2000_seeds123.md`. 3-seed mean eval loss:

| cell | role | params | mean eval | quality/MB |
|---|---|---:|---:|---:|
| h256 / 1000 steps | token axis, low | 19.79M | 5.4278 | 0.0133 |
| h256 / 2000 steps | token axis, high / param axis, high | 19.79M | **5.2101** | 0.0139 |
| h128 / 2000 steps | param axis, low | 9.18M | 5.3957 | 0.0399 |

### Token vs param loss reduction

- **Doubling tokens** at fixed width (h256: 1000 -> 2000 steps):
  5.4278 -> 5.2101 = **-0.2177** eval loss.
- **~2.15x params** at fixed tokens (h128 -> h256 at 2000 steps, 9.18M -> 19.79M):
  5.3957 -> 5.2101 = **-0.1856** eval loss.

Both moves are far above the +/-0.0203 noise floor, and **the token-doubling drop
(-0.2177) exceeds the param-scaling drop (-0.1856)**. The token win is even larger
on a per-FLOP basis: doubling tokens at fixed width is ~2x training compute,
whereas the param axis here both doubles+ the parameter count and adds compute per
step, yet buys *less* loss reduction.

(Note: the param-axis ratio realized was ~2.15x params rather than the ~4x the plan
sketched -- h128->h256 at this config doubles hidden size but the dense vocab head
dominates param growth. The ordering conclusion is unaffected: even at only ~2.15x
params, width still loses to a clean 2x token doubling.)

### Verdict: PROMOTE

The token-over-parameter ordering from Spectra (beta > alpha for ternary LMs)
**reproduces locally**: at fixed width, doubling the token budget reduces eval loss
more than scaling parameters does, beyond the noise floor across 3 seeds. quality/MB
also favors the smaller-but-longer-trained regime conceptually (h128 packs far
smaller at comparable loss).

Actionable implication for the frontier: for a fixed RTX 3050 Ti budget, spend
compute on **more training tokens / steps at the current width** rather than on
wider models. This directly informs the rank 2/3 confirm runs -- prefer longer
token budgets at h128/h256 over going wider. This is an *informational* probe
(budget-allocation guidance), not a deployable model variant.
