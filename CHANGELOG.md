# Changelog

All notable changes to jlenskit are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `jlenskit.baselines.TunedLens` — learned affine read-out (one weight + bias per layer),
  trained on-device with a short cross-entropy pass; implements the same `Lens` protocol as
  the J-lens and logit lens so metrics, viz, and the showdown harness just work.
- `metrics.forward_kl` — KL divergence from the model's own output distribution, per layer,
  averaged over a corpus; captures how faithful each lens is to the model's actual predictive
  distribution (lower = more faithful).
- `metrics.entropy` — mean Shannon entropy (nats) of the lens distribution per layer; a
  companion to `forward_kl` for diagnosing under-sharpened or degenerate lens heads.
- `jlenskit.data.load_probes` — knowledge-probe suite (subject–relation–object triples);
  `showdown` uses these to report *elicitation depth*: the first layer where the correct
  answer appears in each lens's top-k read-out.
- `jlenskit showdown CONFIG` — one-shot CLI command that fits (or loads) all three lenses
  (logit, tuned, J-lens), runs coherence metrics + probe elicitation depth, and writes
  `showdown_metrics.json`, `showdown.md`, and `showdown.html` to the run directory.
  The `layers-to-coherence` statistic (earliest layer where `forward_kl < coherence_tau`)
  provides a single-number summary of how quickly each lens reaches model-level coherence.

### Fixed
- `Adapter.to_logits` now computes the vocabulary projection in float32, so bf16/fp16
  rounding can no longer reorder near-tied top-k tokens in lens read-outs.
- Lens read-outs now apply the model's `final_logit_softcapping` by default, matching
  the model's own logits on architectures like Gemma 2 (previously the model's logits
  were softcapped but the lens's were not, skewing comparisons and kurtosis).
- `JacobianLens.merge` now records both source slices' corpus provenance in
  `meta.extra["merged_from"]` and deep-copies metadata instead of silently claiming the
  left operand's corpus and mutating shared dicts.
- Removed a dead `/ (seq**0.0)` (always 1.0) factor in the `inject` intervention.
- Test tokenizers now use a stable hash instead of the per-process-salted builtin
  `hash()`, removing latent cross-process CI flakiness.

### Changed
- Declared a `hf` optional dependency extra (`pip install 'jlenskit[hf]'`) for the
  `datasets`-backed corpus source.

### Added
- `metrics.logit_fidelity` — per-layer, logit-space faithfulness of the lens to the
  model's actual output (`KL(model || lens)` + exact top-1 agreement), for running on
  held-out text to catch a lens that only works near its fit operating point. Now also
  part of `evaluate` (`fidelity_logit_kl`, `fidelity_top1_agreement`).
- `metrics.jspace_variance_explained` — fraction of residual-stream variance captured by
  the top-k verbalizable J-space concepts per layer (the paper's "global workspace
  footprint", reported ~10% and concentrated mid-network).
- `jlenskit.logit_lens` — the classic logit-lens baseline, exposed as a first-class
  read-out with the same `LensResult` shape as the J-lens for direct comparison
  (the paper frames the J-lens as a principled refinement of the logit lens).
- `examples/quickstart.ipynb` — a runnable, Colab-ready walkthrough (fit → apply →
  compare against the logit lens → visualize) on GPT-2.
- Packaging & project hygiene: `LICENSE` (Apache-2.0), `CITATION.cff`, `py.typed`
  marker (ships type information to downstream users), `CONTRIBUTING.md`, and a
  GitHub Actions CI running ruff + pytest on Python 3.10/3.11/3.12.
- Determinism test pinning that same-seed + same-corpus fits reproduce bit-comparable
  Jacobians.

## [0.1.0] - 2026-07-08

### Added
- Initial release: Jacobian estimator, `JacobianLens` (fit/apply/decode/save/load/merge),
  model adapters with autodetection + registry, lens-quality metrics, J-space
  decomposition and causal inject/swap interventions, HTML viewer, result/manifest
  store, and a config-driven CLI (`fit`/`apply`/`eval`/`viz`/`intervene`).
