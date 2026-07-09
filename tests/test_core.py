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


def test_estimator_respects_autoregressive_causality(toy_adapter):
    """The whole efficiency argument rests on d h_final,t' / d h_l,t == 0 for t' < t
    (a source position cannot influence an earlier target). Verify the future-source
    terms really are zero, so summing the final residual over ALL targets t' equals the
    causal t'>=t sum the estimator claims to compute -- not a coincidence of this model."""
    ids = torch.randint(1, 64, (1, 5))
    cap = toy_adapter.capture(ids, requires_grad=True)
    r0 = cap.residuals[0]  # [1, s, D]
    target_t, out_dim = 1, 0
    g = cap.final_residual[0, target_t, out_dim]
    grad = torch.autograd.grad(g, r0, retain_graph=True)[0][0]  # [s, D], wrt each source pos
    # sources strictly AFTER the target (the future) must not influence it
    assert torch.allclose(grad[target_t + 1 :], torch.zeros_like(grad[target_t + 1 :]), atol=1e-6)
    # sources at/before the target generally DO influence it (sanity: the test isn't vacuous)
    assert grad[: target_t + 1].abs().sum() > 0


def test_merge_equivalent_to_full_fit(toy_adapter):
    b1 = [torch.randint(1, 64, (2, 6)) for _ in range(2)]
    b2 = [torch.randint(1, 64, (2, 6)) for _ in range(2)]
    full = JacobianLens.fit(toy_adapter, b1 + b2, layers=[0, 1], chunk_size=8)
    a = JacobianLens.fit(toy_adapter, b1, layers=[0, 1], chunk_size=8)
    b = JacobianLens.fit(toy_adapter, b2, layers=[0, 1], chunk_size=8)
    merged = a.merge(b)
    # sequences are equal length so pair-weighting reproduces the full average
    assert torch.allclose(merged.jacobians[0], full.jacobians[0], atol=1e-5)


def test_merge_equivalent_to_full_fit_unequal_lengths(toy_adapter):
    """Pair-count weighting must reproduce the full-corpus fit even when slices have
    DIFFERENT sequence lengths (the general case). This is the property that makes
    distributed/incremental fitting sound, not an artifact of equal-length batches."""
    b1 = [torch.randint(1, 64, (2, 5))]
    b2 = [torch.randint(1, 64, (3, 9))]
    full = JacobianLens.fit(toy_adapter, b1 + b2, layers=[0, 1], chunk_size=8)
    a = JacobianLens.fit(toy_adapter, b1, layers=[0, 1], chunk_size=8)
    b = JacobianLens.fit(toy_adapter, b2, layers=[0, 1], chunk_size=8)
    merged = a.merge(b)
    for layer in (0, 1):
        assert torch.allclose(merged.jacobians[layer], full.jacobians[layer], atol=1e-5)


def test_merge_records_both_slice_provenance(toy_adapter):
    """A merged lens must record BOTH corpus slices, not silently claim one."""
    b1 = [torch.randint(1, 64, (2, 6)) for _ in range(2)]
    b2 = [torch.randint(1, 64, (2, 6)) for _ in range(2)]
    a = JacobianLens.fit(toy_adapter, b1, layers=[0, 1], chunk_size=8, corpus_spec={"tag": "a"})
    b = JacobianLens.fit(toy_adapter, b2, layers=[0, 1], chunk_size=8, corpus_spec={"tag": "b"})
    merged = a.merge(b)

    provenance = merged.meta.extra["merged_from"]
    assert [p["corpus_spec"]["tag"] for p in provenance] == ["a", "b"]
    # merging must not mutate the operands' metadata
    assert a.meta.extra.get("merged_from") is None
    assert merged.meta.n_prompts == a.meta.n_prompts + b.meta.n_prompts


def test_decoded_topk_invariant_to_jacobian_rescaling(toy_adapter, toy_lens):
    """The code claims J_l's overall scale is irrelevant to decoded tokens because the
    model's final norm is rescaling-invariant. Test that claim directly: scaling every
    J_l by a positive constant leaves the decoded top-k identical. (Holds exactly for
    RMSNorm models like the toy Llama; would NOT for LayerNorm-with-bias -- a real
    distinction, so this guards the documented invariant rather than a magic value.)"""
    prompt = "the quick brown fox jumps"
    layers = toy_lens.layers
    base = toy_lens.apply(toy_adapter, prompt, positions=[-1], layers=layers, top_k=5)
    scaled = JacobianLens({l: toy_lens.jacobians[l] * 7.5 for l in layers}, toy_lens.meta)
    got = scaled.apply(toy_adapter, prompt, positions=[-1], layers=layers, top_k=5)
    for i in range(len(layers)):
        assert base.decoded[0][i].token_ids == got.decoded[0][i].token_ids


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
