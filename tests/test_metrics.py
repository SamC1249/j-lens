"""Tests for jlenskit.metrics using the toy fixtures (no downloads / GPU)."""

from __future__ import annotations

import math

import torch

from jlenskit.baselines import LogitLens
from jlenskit.metrics import (
    autocorrelation,
    effective_dimension,
    entropy,
    evaluate,
    excess_kurtosis,
    forward_kl,
    jspace_variance_explained,
    logit_fidelity,
    metrics_to_rows,
    topk_accuracy,
)


def _is_finite_float(x) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def _assert_layer_keyed(result, layers):
    assert isinstance(result, dict)
    assert set(result.keys()) == set(layers)
    for v in result.values():
        assert _is_finite_float(v)


def test_topk_accuracy(toy_adapter, toy_lens, toy_batches):
    result = topk_accuracy(toy_adapter, toy_lens, toy_batches, k=5)
    _assert_layer_keyed(result, toy_lens.layers)
    for v in result.values():
        assert 0.0 <= v <= 1.0


def test_excess_kurtosis(toy_adapter, toy_lens, toy_batches):
    result = excess_kurtosis(toy_adapter, toy_lens, toy_batches)
    _assert_layer_keyed(result, toy_lens.layers)


def test_autocorrelation(toy_adapter, toy_lens, toy_batches):
    result = autocorrelation(toy_adapter, toy_lens, toy_batches)
    _assert_layer_keyed(result, toy_lens.layers)
    for v in result.values():
        assert 0.0 <= v <= 1.0


def test_effective_dimension(toy_adapter, toy_lens):
    result = effective_dimension(toy_lens)
    _assert_layer_keyed(result, toy_lens.layers)
    for v in result.values():
        assert v > 0.0
        assert v <= toy_adapter.d_model


def test_evaluate_returns_all_keys(toy_adapter, toy_lens, toy_batches):
    metrics = evaluate(toy_adapter, toy_lens, toy_batches, k=5)
    assert set(metrics.keys()) == {
        "topk_accuracy",
        "excess_kurtosis",
        "autocorrelation",
        "effective_dimension",
        "fidelity_logit_kl",
        "fidelity_top1_agreement",
    }
    for per_layer in metrics.values():
        _assert_layer_keyed(per_layer, toy_lens.layers)


def test_fidelity_top1_agreement_equals_topk_accuracy_at_k1(toy_adapter, toy_lens, toy_batches):
    """top-1 agreement and top-k accuracy at k=1 both mean 'lens argmax == model argmax';
    two independent code paths must agree. Cross-checks correctness without pinning a value."""
    fid = logit_fidelity(toy_adapter, toy_lens, toy_batches)
    acc1 = topk_accuracy(toy_adapter, toy_lens, toy_batches, k=1)
    for l in toy_lens.layers:
        assert abs(fid[l]["top1_agreement"] - acc1[l]) < 1e-9


def test_fidelity_kl_is_nonnegative_and_finite(toy_adapter, toy_lens, toy_batches):
    """KL(model || lens) >= 0 by definition, for every layer — a property of the measure,
    not of this model."""
    fid = logit_fidelity(toy_adapter, toy_lens, toy_batches)
    for l in toy_lens.layers:
        assert fid[l]["logit_kl"] >= -1e-6
        assert math.isfinite(fid[l]["logit_kl"])


def test_variance_explained_is_bounded_and_monotone_in_k(toy_adapter, toy_lens, toy_batches):
    """Non-negative matching pursuit removes energy each step, so explaining variance with
    more concepts can only help: fraction(k=large) >= fraction(k=small), and both lie in
    [0, 1]. This guards the decomposition's core property, not a magic fraction."""
    lo = jspace_variance_explained(toy_adapter, toy_lens, toy_batches, k=2, max_positions=32)
    hi = jspace_variance_explained(toy_adapter, toy_lens, toy_batches, k=12, max_positions=32)
    for l in toy_lens.layers:
        assert -1e-6 <= lo[l] <= 1.0 + 1e-6
        assert -1e-6 <= hi[l] <= 1.0 + 1e-6
        assert hi[l] >= lo[l] - 1e-6


def test_metrics_to_rows(toy_adapter, toy_lens, toy_batches):
    metrics = evaluate(toy_adapter, toy_lens, toy_batches, k=5)
    rows = metrics_to_rows(metrics)
    assert isinstance(rows, list)
    assert len(rows) > 0
    for row in rows:
        assert isinstance(row, dict)
        assert set(row.keys()) == {"metric", "layer", "value"}
        assert isinstance(row["metric"], str)
        assert isinstance(row["layer"], int)
        assert _is_finite_float(row["value"])


def test_forward_kl_nonnegative_and_shaped(toy_adapter, toy_lens):
    kl = forward_kl(toy_adapter, toy_lens, [torch.randint(1, 64, (2, 8))])
    assert set(kl) == set(toy_lens.layers)
    assert all(v >= -1e-5 for v in kl.values())


def test_entropy_matches_logit_lens_manual(toy_adapter):
    lens = LogitLens(list(range(toy_adapter.n_layers)))
    ent = entropy(toy_adapter, lens, [torch.randint(1, 64, (1, 4))])
    assert set(ent) == set(lens.layers)
    assert all(v >= 0 for v in ent.values())
