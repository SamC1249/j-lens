# Contributing to jlenskit

Thanks for your interest! jlenskit aims to be a clean, reproducible reference
implementation of the Jacobian lens that researchers can trust and extend.

## Development setup

```bash
git clone https://github.com/SamC1249/j-lens
cd j-lens
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a PR

The CI runs exactly these two commands on Python 3.10–3.12 — run them locally first:

```bash
ruff check .        # lint + import order
pytest -q           # 40 fast tests, no network, no GPU
```

The test suite uses a tiny random-init model (see `tests/conftest.py`), so it
needs no downloads and finishes in a few seconds.

## Adding a new model family

You almost never need new code — just a capability spec. Add a `ModelSpec` entry
to `jlenskit/adapters/registry.py` describing where the layers, final norm,
unembedding, and embedding live, plus any quirks (e.g. logit softcapping). If
autodetection already handles the architecture, add a test model to
`tests/test_adapters.py` and confirm `Adapter.to_logits` round-trips the model's
own logits.

## Correctness bar

Interpretability tooling is only useful if it is exactly right. New estimator or
lens behavior should come with a test that pins it against an
independently-computed reference (see `test_estimator_matches_brute_force` and
`tests/parity/` for the pattern). Prefer a slow-but-obvious oracle in tests over
trusting the optimized path.

## Style

- `ruff` is the source of truth; config lives in `pyproject.toml`.
- Keep model-specific logic confined to `ModelSpec`; the core math talks to
  models only through `Adapter`.
- Public data structures are dataclasses in `core/types.py` — communicate through
  explicit contracts, not ad-hoc dicts.

## License

By contributing you agree that your contributions are licensed under Apache-2.0.
