# jlenskit

[![CI](https://github.com/SamC1249/j-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/SamC1249/j-lens/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SamC1249/j-lens/blob/main/examples/quickstart.ipynb)

**A reproducible, modular toolkit for running the Jacobian lens (J-lens) on open-weights language models.**

The J-lens, from Anthropic's [*"Verbalizable Representations Form a Global Workspace in
Language Models"*](https://transformer-circuits.pub/2026/workspace/index.html) (Transformer
Circuits, 2026), reads what a hidden state is *disposed to make the model say* — the "silent"
tokens in a model's internal workspace, even ones it never emits.

`jlenskit` turns Anthropic's [reference code](https://github.com/anthropics/jacobian-lens) into
a researcher-friendly package:

- **Modular** — interpret *any* HuggingFace decoder by writing a ~15-line capability spec, not code.
- **Easy** — one config-driven CLI command per experiment, plus a clean Python API.
- **Accessible outputs** — every run writes structured results (Parquet + JSON), a
  self-contained HTML viewer, and a reproducibility manifest.

## How it works

For a residual-stream activation `h` at layer `ℓ`:

```
lens_ℓ(h) = softmax( W_U · norm( J_ℓ · h ) )      J_ℓ = E_{prompt, t, t'≥t} [ ∂h_final,t' / ∂h_ℓ,t ]
```

`J_ℓ` is the corpus-averaged input→output Jacobian; `norm`/`W_U` are the model's own final-norm
and unembedding. Because the model is autoregressive, the `t'≥t` constraint is free, so fitting
costs ~`d_model` backward passes over the corpus — **independent of sequence length**.

## Install

```bash
pip install -e .            # from a clone
pip install -e ".[dev]"     # + pytest/ruff
pip install -e ".[parity]"  # + anthropics/jlens for parity tests
```

## Quickstart (Python)

```python
import jlenskit
from jlenskit.core import JacobianLens
from jlenskit.data import load_corpus

adapter = jlenskit.load("gpt2")                          # any HF decoder
batches = load_corpus({"source": "builtin", "seq_len": 64, "n_sequences": 64}, adapter.tokenizer)
lens = JacobianLens.fit(adapter, batches, chunk_size=64) # estimate J_ℓ for all layers

res = lens.apply(adapter, "The capital of France is", positions=[-1], top_k=10)
for dl in res.decoded[0]:
    print(dl.layer, dl.tokens[:5])                       # watch silent tokens evolve by layer

# Compare against the classic logit lens on the same activation (the J-lens is a
# principled refinement of it) — same LensResult shape, so viz/metrics just work:
from jlenskit import logit_lens
base = logit_lens(adapter, "The capital of France is", positions=[-1], top_k=10)
```

New to the tool? **[`examples/quickstart.ipynb`](examples/quickstart.ipynb)** walks through
fit → apply → logit-lens comparison → visualize, and runs in a few minutes on CPU
(click the Colab badge above).

## Quickstart (CLI)

Everything is driven by a YAML config (recorded in each run's manifest for reproducibility):

```bash
jlenskit fit       configs/gpt2_demo.yaml   # fit (or load-from-cache) a lens
jlenskit apply     configs/gpt2_demo.yaml   # decode prompts -> results + viz.html
jlenskit eval      configs/gpt2_demo.yaml   # lens-quality metrics over a corpus
jlenskit viz       configs/gpt2_demo.yaml   # render the layer × position viewer
jlenskit intervene configs/qwen_intervene.yaml  # concept inject/swap experiment
```

Outputs land in `output.dir`:

```
runs/gpt2_demo/
├── lens.safetensors      # the fitted lens (+ embedded provenance)
├── manifest.json         # model, revision, seed, corpus, versions, config
├── metrics.parquet/json  # top-k accuracy, kurtosis, autocorrelation, effective dim
└── prompt_000/
    ├── result.parquet/json
    └── viz.html          # self-contained, offline
```

## Capabilities

| Module | What it gives you |
|---|---|
| `jlenskit.core` | Fit / apply / decode the lens; save / load / **merge** (parallel fitting) |
| `jlenskit.metrics` | top-k accuracy, excess kurtosis, autocorrelation, effective dimension — locate the "workspace" layers; **logit-space fidelity** (KL + top-1 vs the model) and **J-space variance-explained** to quantify faithfulness and workspace footprint |
| `jlenskit.jspace` | Sparse decomposition of an activation into lens vectors; causal `inject` / `swap` interventions |
| `jlenskit.viz` | Layer × position slice/grid HTML viewer |
| `jlenskit.store` | Lens cache (local + optional HF Hub), Parquet/JSON result store, run manifests |

## Supported models

Autodetection handles most HF decoders; these families are pinned and tested: **Qwen 2/3,
Llama, Mistral, Gemma 1/2, GPT-2, GPT-NeoX**. Add another by dropping a `ModelSpec` into
`jlenskit/adapters/registry.py`.

## Compute

Primary target is a single modern GPU (0.5B–8B comfortably; `device_map` for larger). GPT-2
runs on CPU for tutorials; the test suite uses a tiny random-init model, so it needs no
downloads or GPU.

## Reproducibility & parity

- Global seed control; the corpus is pinned by spec (source + revision + slice) and recorded
  in the manifest alongside package/torch/transformers versions.
- `tests/parity/` checks decoded top-k against `anthropics/jlens` as a correctness oracle
  (opt-in via the `[parity]` extra).

## Development

```bash
pytest -q          # 40 fast tests, no network, no GPU
ruff check .
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for adding model families and the correctness bar.

## License

Apache-2.0. Companion to the paper; not affiliated with Anthropic.
