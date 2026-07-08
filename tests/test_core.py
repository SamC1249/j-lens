"""Core estimator + lens tests: correctness vs brute force, merge, save/load."""

from __future__ import annotations

import torch

from jlenskit.core.jacobian import compute_layer_jacobians
from jlenskit.core.lens import JacobianLens


def test_estimator_matches_brute_force(toy_adapter):
    ids = torch.randint(1, 64, (2, 6))
    layers = [0, 1]
    J, n_seq, n_pairs = compute_layer_jacobians(toy_adapter, [ids], layers, chunk_size=8)

    cap = toy_adapter.capture(ids, requires_grad=True)
    r0 = cap.residuals[0]
    D = toy_adapter.d_model
    g = cap.final_residual.reshape(-1, D).sum(dim=0)
    brute = torch.zeros(D, D)
    for i in range(D):
        gr = torch.autograd.grad(g[i], r0, retain_graph=True)[0]
        brute[i] = gr.reshape(-1, D).sum(dim=0)
    brute /= n_pairs
    assert torch.allclose(J[0], brute, atol=1e-5)


def test_merge_equivalent_to_full_fit(toy_adapter):
    b1 = [torch.randint(1, 64, (2, 6)) for _ in range(2)]
    b2 = [torch.randint(1, 64, (2, 6)) for _ in range(2)]
    full = JacobianLens.fit(toy_adapter, b1 + b2, layers=[0, 1], chunk_size=8)
    a = JacobianLens.fit(toy_adapter, b1, layers=[0, 1], chunk_size=8)
    b = JacobianLens.fit(toy_adapter, b2, layers=[0, 1], chunk_size=8)
    merged = a.merge(b)
    # sequences are equal length so pair-weighting reproduces the full average
    assert torch.allclose(merged.jacobians[0], full.jacobians[0], atol=1e-5)


def test_save_load_roundtrip(tmp_path, toy_lens):
    p = tmp_path / "lens.safetensors"
    toy_lens.save(p)
    loaded = JacobianLens.load(p)
    assert loaded.layers == toy_lens.layers
    for l in toy_lens.layers:
        assert torch.allclose(loaded.jacobians[l], toy_lens.jacobians[l])
    assert loaded.meta.d_model == toy_lens.meta.d_model
    assert loaded.meta.jlenskit_version == toy_lens.meta.jlenskit_version


def test_apply_decodes_all_layers(toy_adapter, toy_lens):
    res = toy_lens.apply(toy_adapter, "hello there world", positions=[-2, -1], top_k=4)
    assert set(res.lens_logits) == set(toy_lens.layers)
    assert len(res.positions) == 2
    for pos_decoded in res.decoded.values():
        assert len(pos_decoded) == len(toy_lens.layers)
        assert all(len(dl.tokens) == 4 for dl in pos_decoded)
