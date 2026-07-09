# Changelog

All notable changes to jlenskit are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
