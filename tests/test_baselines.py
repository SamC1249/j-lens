"""Logit-lens baseline: shape parity with the J-lens, and fit/apply determinism."""

from __future__ import annotations

import torch

from jlenskit.baselines import LogitLens, logit_lens
from jlenskit.core.lens import JacobianLens


def test_logit_lens_matches_jlens_shape(toy_adapter, toy_lens):
    """The baseline must produce the same LensResult shape the J-lens does, so the two
    are directly comparable and share the viz path."""
    prompt = "the quick brown fox"
    layers = toy_lens.layers
    j = toy_lens.apply(toy_adapter, prompt, positions=[-1], layers=layers, top_k=5)
    b = logit_lens(toy_adapter, prompt, positions=[-1], layers=layers, top_k=5)

    assert set(b.lens_logits) == set(j.lens_logits)
    assert b.positions == j.positions
    for l in layers:
        assert b.lens_logits[l].shape == j.lens_logits[l].shape
        assert len(b.decoded[0]) == len(j.decoded[0])


def test_logit_lens_is_not_identical_to_jlens(toy_adapter, toy_lens):
    """The transport actually does something: at least one layer's logits differ."""
    prompt = "the quick brown fox"
    layers = toy_lens.layers
    j = toy_lens.apply(toy_adapter, prompt, positions=[-1], layers=layers)
    b = logit_lens(toy_adapter, prompt, positions=[-1], layers=layers)
    differs = any(not torch.allclose(b.lens_logits[l], j.lens_logits[l]) for l in layers)
    assert differs, "logit lens and J-lens gave identical logits at every layer"


def test_logit_lens_deterministic(toy_adapter):
    prompt = "the quick brown fox"
    a = logit_lens(toy_adapter, prompt, positions=[-1])
    b = logit_lens(toy_adapter, prompt, positions=[-1])
    for l in a.lens_logits:
        assert torch.allclose(a.lens_logits[l], b.lens_logits[l])


def test_fit_deterministic_under_seed(toy_adapter, toy_batches):
    """Same seed + same corpus => bit-comparable Jacobians (reproducibility contract)."""
    a = JacobianLens.fit(toy_adapter, toy_batches, layers=[0, 1, 2], chunk_size=8, seed=0)
    b = JacobianLens.fit(toy_adapter, toy_batches, layers=[0, 1, 2], chunk_size=8, seed=0)
    for l in a.layers:
        assert torch.allclose(a.jacobians[l], b.jacobians[l])


def test_logitlens_transport_is_identity(toy_adapter):
    lens = LogitLens(list(range(toy_adapter.n_layers)))
    h = torch.randn(4, toy_adapter.d_model)
    assert torch.equal(lens.transport(1, h), h)
    assert lens.name == "logit"
