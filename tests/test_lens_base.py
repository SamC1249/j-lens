import torch

from jlenskit.core.lens_base import apply_lens


def test_jacobian_transport_matches_manual(toy_lens):
    h = torch.randn(toy_lens.meta.d_model)
    manual = h @ toy_lens.jacobians[1].t()
    assert torch.allclose(toy_lens.transport(1, h), manual, atol=1e-6)
    assert toy_lens.name == "jacobian"


def test_apply_lens_matches_jacobianlens_apply(toy_adapter, toy_lens):
    a = toy_lens.apply(toy_adapter, "one two three", positions=[-1], top_k=3)
    b = apply_lens(toy_lens, toy_adapter, "one two three", positions=[-1], top_k=3)
    for l in a.lens_logits:
        assert torch.allclose(a.lens_logits[l], b.lens_logits[l], atol=1e-6)
    assert a.decoded[0][-1].token_ids == b.decoded[0][-1].token_ids
