# jlenskit — a reproducible, modular Jacobian-lens toolkit

**Date:** 2026-07-08
**Status:** Approved (design)

## Purpose

Provide a reproducible, open-source Python package that runs the **Jacobian lens (J-lens)** —
from Anthropic's *"Verbalizable Representations Form a Global Workspace in Language Models"*
(Transformer Circuits, July 2026) — on arbitrary open-weights language models.

Three goals, in priority order:

1. **Modular** — any HuggingFace decoder model can be interpreted by writing a small
   capability spec (data), not code.
2. **Easy for researchers** — one-command, config-driven runs; a clean Python API; PyPI install.
3. **Accessible outputs** — every run produces structured, persisted, shareable results
   (JSON/parquet + a self-contained HTML viz) plus a reproducibility manifest.

## What the J-lens is (reference)

For a residual-stream activation `h` at layer `ℓ`:

```
lens_ℓ(h) = softmax( W_U · norm( J_ℓ · h ) )
J_ℓ       = E_{prompt, t, t'≥t} [ ∂h_final,t' / ∂h_ℓ,t ]
```

`J_ℓ` is the corpus-averaged input→output Jacobian mapping the residual stream at layer `ℓ`
to the final pre-norm residual stream. `norm` is the model's final norm; `W_U` its unembedding.
**J-space** is the set of points expressible as a sparse non-negative combination of the
per-token lens vectors.

### Key estimator insight

Because the model is autoregressive, `∂h_final,t' / ∂h_ℓ,t = 0` for `t' < t`. So the
constraint `t' ≥ t` is free. Define the scalar `S_i = Σ_{batch} Σ_{t'} h_final,t',i`.
A single backward pass from `S_i` yields `∂S_i/∂h_ℓ,t,j` for **all** layers `ℓ`, positions
`t`, and input dims `j` at once. Summing over positions/batch and dividing by the count gives
row `i` of `J_ℓ` for every layer simultaneously. Fitting therefore costs `d_model` backward
passes over the corpus (not `d_model × positions`), independent of sequence length.

## Architecture

Clean reimplementation (self-contained core), with numerical **parity tests against
`anthropics/jlens`** as the correctness oracle. The Jacobian math never touches HF internals
directly — it only talks to an `Adapter`. That indirection is what makes "any model" a data
problem.

```
jlenskit/
├── core/
│   ├── jacobian.py   # averaged-Jacobian estimator (VJP accumulation over corpus)
│   ├── lens.py       # JacobianLens: fit / apply / decode / save / load / merge
│   └── types.py      # LensResult, LensBundle, RunManifest, DecodedLayer dataclasses
├── adapters/
│   ├── base.py       # Adapter: capture(), to_logits(), patch(); residual-stream hooks
│   ├── spec.py       # ModelSpec capability config + autodetect from HF config
│   └── registry.py   # family specs: qwen, llama/mistral, gemma, gpt2/neox
├── metrics/          # topk_accuracy, excess_kurtosis, autocorrelation, effective_dim
├── jspace/           # sparse OMP/gradient-pursuit decomposition + interventions
├── viz/              # layer × position slice → self-contained HTML
├── data/             # pinned corpus loader + bundled prompt sets
├── io/               # lens cache (local + optional HF hub), result store, manifests
├── cli.py            # typer CLI: fit | apply | eval | viz | intervene (config-driven)
└── configs/          # example YAML configs
tests/
├── parity/           # numerical parity vs anthropics/jlens (optional extra)
├── conftest.py       # tiny random-init toy model fixture (no downloads)
└── ...
```

### The Adapter contract (crux of modularity)

Because we hook the *real* model to capture residuals and call the *real* final-norm and
unembedding, architecture quirks (Gemma embed-scaling, RoPE, GQA, MoE routing) are handled by
the model itself. The adapter only needs correct module paths + a few flags:

```python
class Adapter:
    n_layers: int; d_model: int; vocab_size: int
    def capture(self, input_ids) -> Capture      # residual streams (retain_grad) per layer
    def to_logits(self, h) -> Tensor             # W_U(final_norm(h)); real modules
    def patch(self, layer, fn) -> ContextManager  # inject/edit residual at a layer, resume fwd
```

`ModelSpec` (the ~15-line config): `layers_path`, `final_norm_path`, `unembed_path`,
`embed_path`, `norm_type`, `tie_embeddings`, optional `quirks`. `autodetect()` fills these from
the HF config + common path patterns; the registry pins known families.

## Capabilities (all in v1)

1. **Lens fit + apply + decode** (core) — estimate `J_ℓ`, apply to any prompt/position, decode
   to ranked vocab per layer; save/load/merge; local cache.
2. **Lens-quality metrics** — top-k accuracy vs model, excess kurtosis, autocorrelation,
   effective linear dimensionality; locates the "workspace" layers.
3. **J-space + interventions** — sparse decomposition of an activation into k lens vectors
   (gradient pursuit / OMP); concept swap/injection via `patch`, measuring output change.
4. **Visualization** — layer × position slice viewer as a self-contained HTML file.

## Interfaces, outputs, reproducibility

- **API-first, thin CLI.** `jlenskit fit|apply|eval|viz|intervene`, each drivable by a YAML
  config for reproducibility. Python API mirrors the CLI.
- **Output store.** Each run writes to a run directory: `manifest.json` (model id+revision,
  lens hash, corpus spec, seed, package + torch/transformers versions, config), lens
  checkpoints (safetensors), results as parquet + JSON, and `viz.html`.
- **Lens cache.** Keyed by `(model, revision, corpus-hash, fit-config-hash)`; local dir by
  default, optional push/pull to the HF Hub (token via `.env`/env).
- **Determinism.** Global seed control; corpus pinned to a named HF dataset + revision + index
  slice; all captured in the manifest.

## Targets & compute

- **Model families (tested in v1):** Qwen (parity target), Llama/Mistral, Gemma, GPT-2/NeoX.
- **Compute:** single modern GPU is the primary target (0.5B–8B comfortably; `device_map` for
  larger). GPT-2-small provides a CPU-runnable path for tests/tutorials. Tests use a tiny
  random-init toy transformer so CI needs no downloads or GPU.

## Dependencies

`torch`, `transformers`, `safetensors`, `numpy`, `pydantic`, `pyarrow`, `typer`,
`huggingface_hub`, `pyyaml`. Dev: `pytest`, `ruff`. Optional extra `[parity]` pulls in
`anthropics/jlens` for parity tests only.

## Testing strategy

- **Toy-model fixture:** a 2-layer, d=32 random-init GPT-2/Llama-like model instantiated
  locally — fast, deterministic, no network.
- **Adapter tests:** each family's spec resolves paths, capture shapes, norm type, tying,
  logit round-trip (`to_logits(final_residual) == model logits`).
- **Core tests:** estimator correctness vs a brute-force autograd Jacobian on the toy model;
  lens apply/decode shapes; save/load/merge round-trips.
- **Metrics/jspace tests:** on toy model — monotonic top-k accuracy trend; OMP reconstruction
  error decreases with k; a swap changes the decoded top token.
- **Parity tests (optional extra):** `J_ℓ` and decoded top-k match `anthropics/jlens` on a
  small real Qwen within tolerance.

## Non-goals (v1)

- Training/fine-tuning models. Encoder or encoder-decoder architectures. A hosted web service
  (viz is a static file). Distributed multi-GPU fitting beyond `merge()` of corpus slices.
