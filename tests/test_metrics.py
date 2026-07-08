"""Tests for jlenskit.metrics using the toy fixtures (no downloads / GPU)."""

from __future__ import annotations

import math

from jlenskit.metrics import (
    autocorrelation,
    effective_dimension,
    evaluate,
    excess_kurtosis,
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
    }
    for per_layer in metrics.values():
        _assert_layer_keyed(per_layer, toy_lens.layers)


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
