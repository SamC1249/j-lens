"""Tests for J-space decomposition and causal interventions."""

from __future__ import annotations

import torch

from jlenskit import jspace


def test_decompose_shapes_and_nonneg(toy_adapter, toy_lens):
    h = torch.randn(toy_adapter.d_model)
    dec = jspace.decompose(toy_adapter, toy_lens, h, layer=1, k=8)
    assert dec.layer == 1
    assert len(dec.coords) <= 8
    assert all(c.coefficient >= 0 for c in dec.coords)
    assert 0.0 <= dec.reconstruction_error <= 1.0 + 1e-4
    assert all(isinstance(c.token, str) for c in dec.coords)


def test_reconstruction_improves_with_k(toy_adapter, toy_lens):
    h = torch.randn(toy_adapter.d_model)
    err1 = jspace.decompose(toy_adapter, toy_lens, h, layer=1, k=1).reconstruction_error
    err8 = jspace.decompose(toy_adapter, toy_lens, h, layer=1, k=8).reconstruction_error
    # matching pursuit is monotone non-increasing in residual norm
    assert err8 <= err1 + 1e-5


def test_decompose_prompt_runs(toy_adapter, toy_lens):
    dec = jspace.decompose_prompt(toy_adapter, toy_lens, "hello world there", layer=1, k=4)
    assert len(dec.coords) <= 4


def test_inject_changes_output(toy_adapter, toy_lens):
    res = jspace.inject(toy_adapter, toy_lens, "hello world", layer=1, token=5, strength=8.0)
    assert res.baseline_top and res.intervened_top
    # a strong injection should perturb the next-token distribution
    assert res.baseline_top != res.intervened_top


def test_swap_runs(toy_adapter, toy_lens):
    res = jspace.swap(toy_adapter, toy_lens, "hello world", layer=1, source=3, target=9, strength=2.0)
    assert len(res.baseline_top) == 5
    assert len(res.intervened_top) == 5
    assert "swap" in res.description
