import torch

from jlenskit.baselines import LogitLens, TunedLens
from jlenskit.core.lens_base import apply_lens
from jlenskit.core.types import LensMeta


def _meta(adapter):
    return LensMeta(model_id="toy", model_revision=None, d_model=adapter.d_model,
                    vocab_size=adapter.vocab_size, n_layers=adapter.n_layers,
                    layers_fit=[0, 1, 2], n_prompts=0, n_positions=0)


def test_zero_init_tuned_equals_logit(toy_adapter):
    layers = [0, 1, 2]
    tuned = TunedLens.zeros(layers, toy_adapter.d_model, _meta(toy_adapter))
    logit = LogitLens(layers)
    rt = apply_lens(tuned, toy_adapter, "one two three", top_k=4)
    rl = apply_lens(logit, toy_adapter, "one two three", top_k=4)
    for l in layers:
        assert torch.allclose(rt.lens_logits[l], rl.lens_logits[l], atol=1e-6)


def test_tuned_save_load_roundtrip(tmp_path, toy_adapter):
    tuned = TunedLens.zeros([0, 1, 2], toy_adapter.d_model, _meta(toy_adapter))
    tuned.A[1] = torch.randn_like(tuned.A[1])
    tuned.b[1] = torch.randn_like(tuned.b[1])
    p = tuned.save(tmp_path / "tuned.safetensors")
    back = TunedLens.load(p)
    assert torch.allclose(back.A[1], tuned.A[1])
    assert torch.allclose(back.b[1], tuned.b[1])
    assert back.meta.model_id == "toy"
