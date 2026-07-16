# Lens Showdown — design spec

**Date:** 2026-07-14
**Status:** approved (design), pending implementation plan
**Author:** Sam Chen (with Claude)

## Goal

Demonstrate, quantitatively and across model families, the J-lens's core value
proposition: it reads a **coherent silent workspace** in the mid layers that the
logit lens cannot, and it does so **without the per-model training the tuned lens
requires**. Ship this as first-class package capability, not a one-off script.

This closes three gaps found during competitive review (2026-07-14):
1. The package has no tuned-lens baseline (already on the roadmap).
2. Current metrics (`topk_accuracy`, kurtosis, …) do **not** capture the
   coherence phenomenon that distinguishes the J-lens — its actual advantage.
3. No head-to-head comparison against the two baselines the field uses
   (logit lens, tuned lens).

Non-goals for this cut: pre-fit downloadable lens artifacts, hosted docs, and
plug-and-play hardening (post-norm warnings, init-time spec validation). Those
are a separate track.

## Key idea: three lenses, one operation

Every lens is "transport a layer-`ℓ` residual `h` into the final-layer basis,
then apply the model's own final-norm + unembedding." They differ only in the
transport `T_ℓ`:

| Lens   | `T_ℓ(h)`                     | Training            |
|--------|------------------------------|---------------------|
| logit  | `h` (identity)               | none                |
| tuned  | `(I + A_ℓ)·h + b_ℓ` (learned)| per-model, per-layer|
| J-lens | `J_ℓ·h` (averaged Jacobian)  | none                |

`logits_ℓ = to_logits(T_ℓ(h))`. Because all three share the same
`Adapter.to_logits` and produce the same `LensResult` shape, viz and metrics work
unchanged for all of them.

## Architecture

### 1. `Lens` protocol / transport abstraction (targeted refactor)

Today `JacobianLens` and the metrics are coupled: metrics reach into
`lens.jacobians[layer]`. Introduce a minimal common interface so all three lenses
are interchangeable in metrics, viz, and the harness.

```python
# jlenskit/core/types.py (or a new lens_base.py)
class Lens(Protocol):
    name: str                      # "logit" | "tuned" | "jacobian"
    layers: list[int]
    def transport(self, layer: int, h: Tensor) -> Tensor: ...   # h:[...,D] -> [...,D]
```

- `JacobianLens.transport(ℓ, h) = h @ J_ℓ.T` (extract from existing `apply`).
- `LogitLens.transport(ℓ, h) = h` (thin; replaces the standalone
  `baselines.logit_lens` function, which stays as a convenience wrapper).
- `TunedLens.transport(ℓ, h) = h + h @ A_ℓ.T + b_ℓ`.

`apply()` and `_decode()` move to a shared helper that takes any `Lens`, so we do
not duplicate the decode loop three times. `baselines.logit_lens(...)` keeps its
current signature for backward compatibility, delegating to `LogitLens`.

**Metrics refactor:** `_lens_logits_for_batch` and every metric switch from
`lens.jacobians[layer]` to `lens.transport(layer, h)`. Metrics that are
Jacobian-specific (`jspace_variance_explained`, `effective_dimension`) stay on
`JacobianLens` and are simply skipped for other lenses.

### 2. `TunedLens` baseline — `jlenskit/baselines/tuned.py`

Minimal in-package implementation of Belrose et al. (2023).

- **Parameters:** per fit layer `ℓ`, an affine translator `A_ℓ ∈ R^{D×D}`
  (initialized to 0) and bias `b_ℓ ∈ R^D` (initialized to 0). At init the tuned
  lens therefore equals the logit lens; training moves it.
- **Training objective:** for each layer independently, minimize
  `KL(softmax(model_final_logits) ‖ softmax(to_logits(T_ℓ(h))))`
  (forward KL to the model's own final distribution), averaged over corpus
  positions. This is distillation to the model's committed output — no labels.
- **Procedure:** capture residuals + final logits over the fit corpus once
  (no grad on the model), cache them, then optimize `{A_ℓ, b_ℓ}` with Adam for
  `n_steps` (default ~250) at a small lr. Train all layers together (shared
  batch, summed loss) for simplicity. Runs on the fit device.
- **Persistence:** `save`/`load` mirror `JacobianLens` — one `.safetensors`
  holding `A_ℓ`, `b_ℓ` per layer + `LensMeta` (reuse `LensMeta`, add
  `lens_type` field). Cache under `LensCache` keyed like the J-lens plus a
  `tuned` discriminator and the training config.
- **Correctness guard:** a test asserting a freshly-initialized `TunedLens`
  (zero A, zero b) produces logits identical to `LogitLens`, and that after
  training its forward-KL to the model is `<=` the logit lens's on held-out
  positions.

### 3. Coherence metrics — `jlenskit/metrics`

New per-layer metrics computed for **any** `Lens` (field-standard trajectory
metrics, matching what tuned-lens reports):

- `forward_kl(lens, ...)` — `KL(model_final ‖ lens_ℓ)` per layer, averaged over
  positions. Lower = the read-out is closer to the model's committed output.
- `entropy(lens, ...)` — mean Shannon entropy of the lens distribution per layer.
- Extend existing `topk_accuracy` to accept any `Lens` (via `transport`).

Headline scalar (computed in the harness, not a metric fn):
- **layers-to-coherence** `L*(τ)` = the smallest layer index `ℓ` such that
  `forward_kl(ℓ) <= τ` and stays below for all deeper layers, with `τ` a config
  value (default: a fraction, e.g. 0.5, of the layer-0 forward-KL, so it is
  scale-free across models). Report `L*` per lens; the J-lens should reach
  coherence at a much shallower layer than the logit lens.

### 4. Knowledge-probe suite — `jlenskit/data/probes/`

A small versioned JSON dataset shipped in-repo:

```json
{"prompt": "The country whose capital is Tokyo uses the currency called the",
 "answers": ["yen", "円"], "category": "two_hop"}
```

- ~20–30 prompts across categories: `factual` (capital/currency), `two_hop`
  (bridge entity), `antonym`, `language`. Each has a determinate answer and a
  list of acceptable answer surface forms (to tolerate tokenizer variants).
- Loader: `load_probes(name="core_v1") -> list[Probe]`. Versioned name in the
  filename so results are reproducible.
- **elicitation depth** metric (harness): for each lens and probe, the first
  layer whose top-`k` contains any answer surface form; report per-category
  medians per lens. The J-lens should surface answers at a shallower (or equal)
  layer than the logit lens.

### 5. `showdown` harness — new CLI subcommand `jlenskit showdown CONFIG`

Config gains a `showdown:` section:

```yaml
showdown:
  lenses: [logit, tuned, jacobian]   # which to compare
  probes: core_v1                    # probe suite name (or null to skip)
  coherence_tau: 0.5                 # fraction of layer-0 KL
  tuned:
    n_steps: 250
    lr: 1.0e-3
  top_k: 5
```

Flow:
1. Load model, resolve/fit the J-lens (existing path), build `LogitLens`, and
   fit-or-load the `TunedLens`.
2. **Corpus pass:** compute `forward_kl`, `entropy`, `topk_accuracy` per layer
   for each lens over the eval corpus.
3. **Probe pass:** compute elicitation depth per lens over the probe suite.
4. **Emit:**
   - `showdown_metrics.json` — nested `{lens: {metric: {layer: value}}}` plus
     the scalar summary (`layers_to_coherence`, per-category elicitation depth).
   - `showdown.md` — a human-readable comparison table (the artifact to look
     at): one row per lens, columns for `L*`, mean forward-KL over layers 0–15,
     final-layer top-5 accuracy, median elicitation depth.
   - `showdown.html` — a grouped multi-line viz (forward-KL vs layer, one line
     per lens) reusing/extending `jlenskit.viz`.
   - A reproducibility manifest via the existing `_finish` path.

### Scope of the first run

Two model families via two configs:
- **Qwen2-0.5B** (`configs/qwen05b_mps.yaml` extended) — cached J-lens exists.
- **GPT-2** (`configs/gpt2_demo.yaml` extended) — fast to fit; deliberately the
  logit lens's best case (tied embeddings, its canonical model). If the J-lens
  wins coherence even here, that is the strong result; a tie on GPT-2 is honest
  and expected, and the Qwen gap carries the headline.

A third family later is just another config — no code change.

## Data flow

```
config ─► adapter ─► {LogitLens, TunedLens(fit/cache), JacobianLens(fit/cache)}
                              │
             ┌────────────────┴───────────────┐
        corpus pass                       probe pass
     forward_kl/entropy/acc           elicitation depth
             └────────────────┬───────────────┘
                     showdown_metrics.json
                     showdown.md  (comparison table)
                     showdown.html (KL-vs-layer, per lens)
                     manifest.json
```

## Error handling

- Requesting a Jacobian-only metric on a non-Jacobian lens: skip with a logged
  note, do not crash.
- Tuned-lens fit on a model whose adapter autodetect failed: surfaces the same
  loud `ModelSpec` error the J-lens path already raises (shared adapter).
- Empty/malformed probe file: fail fast with the offending path.
- `coherence_tau` never reached by a lens (logit lens may stay incoherent):
  report `L* = None` (n/a) rather than the max layer, and say so in the table.

## Testing

- **Unit:** `LogitLens.transport` is identity; zero-init `TunedLens` == logit
  lens; `forward_kl`/`entropy` match hand-computed values on a tiny tensor;
  probe loader parses and validates the shipped suite.
- **Integration (toy model fixture):** run the full `showdown` flow on the
  random-init toy Llama fixture end-to-end; assert all output files are written
  and `showdown_metrics.json` has an entry per requested lens per metric.
- **Invariant:** after training, tuned lens forward-KL to the model is `<=` logit
  lens forward-KL averaged over the **fit** positions. This is the robust
  monotone-improvement guarantee (the logit lens is the zero-init special case of
  the tuned lens, so optimization cannot do worse on its own training set); we
  assert it on the fit corpus rather than held-out to avoid small-fixture
  overfitting flakiness.
- All new tests run on the toy fixture — no downloads, no GPU (consistent with
  the existing suite).

## Deliverable / definition of done

- `jlenskit showdown configs/qwen05b_mps.yaml` and `... configs/gpt2_demo.yaml`
  both run clean and produce `showdown.md` + `showdown.html` + metrics JSON.
- The Qwen `showdown.md` shows the J-lens reaching coherence at a materially
  shallower layer than the logit lens, with the tuned lens comparable to the
  J-lens (demonstrating "no training needed").
- The two scratch prototypes (`scratch_silent.py`, `scratch_curve.py`) are
  deleted, their logic absorbed into the harness.
- Tests + ruff pass.
```
