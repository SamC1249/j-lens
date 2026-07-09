"""Adapter tests across architectures: autodetect, norm type, tying, logit round-trip."""

from __future__ import annotations

import hashlib

import torch

from jlenskit.adapters.base import Adapter
from jlenskit.adapters.spec import autodetect


def _stable_id(word: str) -> int:
    """Deterministic across processes (builtin hash() is salted per-process)."""
    return (int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:4], "big") % 63) + 1


class StubTokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [_stable_id(w) for w in (text.split() or [text])]
        return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}

    def decode(self, ids):
        return f"<{int(ids[0])}>"


def _gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(n_embd=32, n_layer=2, n_head=4, vocab_size=64, n_positions=64)
    return GPT2LMHeadModel(cfg).to(torch.float32).eval()


def test_autodetect_llama(toy_adapter):
    spec = toy_adapter.spec
    assert spec.layers_path == "model.layers"
    assert spec.final_norm_path == "model.norm"
    assert spec.norm_type == "rmsnorm"


def test_autodetect_gpt2_layernorm_and_tying():
    model = _gpt2()
    spec = autodetect(model)
    assert spec.layers_path == "transformer.h"
    assert spec.final_norm_path == "transformer.ln_f"
    assert spec.norm_type == "layernorm"
    assert spec.tie_embeddings is True  # gpt2 ties wte and lm_head


def test_to_logits_roundtrip_llama(toy_adapter):
    ids = torch.randint(1, 64, (2, 6))
    cap = toy_adapter.capture(ids, requires_grad=False)
    recon = toy_adapter.to_logits(cap.final_residual)
    assert torch.allclose(recon, cap.model_logits, atol=1e-4)


def test_to_logits_roundtrip_gpt2():
    model = _gpt2()
    adapter = Adapter(model, StubTokenizer(), spec=autodetect(model))
    ids = torch.randint(1, 64, (2, 6))
    cap = adapter.capture(ids, requires_grad=False)
    recon = adapter.to_logits(cap.final_residual)
    assert torch.allclose(recon, cap.model_logits, atol=1e-4)


def test_to_logits_is_float32(toy_adapter):
    """The vocabulary projection is computed in float32 regardless of model dtype so
    bf16 rounding can't reorder near-tied top-k tokens."""
    ids = torch.randint(1, 64, (1, 4))
    cap = toy_adapter.capture(ids, requires_grad=False)
    assert toy_adapter.to_logits(cap.final_residual).dtype == torch.float32


def test_to_logits_applies_softcap(toy_adapter):
    """Lens read-outs must match the model's own softcapped logits (e.g. Gemma 2)."""
    toy_adapter.spec.quirks["final_logit_softcapping"] = 5.0
    try:
        ids = torch.randint(1, 64, (1, 4))
        h = toy_adapter.capture(ids, requires_grad=False).final_residual
        raw = toy_adapter.to_logits(h, apply_softcap=False)
        capped = toy_adapter.to_logits(h, apply_softcap=True)  # default
        assert torch.allclose(capped, torch.tanh(raw / 5.0) * 5.0)
        assert not torch.allclose(raw, capped)
    finally:
        toy_adapter.spec.quirks.pop("final_logit_softcapping", None)


def test_capture_residual_shapes(toy_adapter):
    ids = torch.randint(1, 64, (2, 7))
    cap = toy_adapter.capture(ids, requires_grad=False)
    # residual points 0..n_layers
    assert set(cap.residuals) == set(range(toy_adapter.n_layers + 1))
    for h in cap.residuals.values():
        assert h.shape == (2, 7, toy_adapter.d_model)


def test_patch_changes_output(toy_adapter):
    ids = torch.randint(1, 64, (1, 6))
    base = toy_adapter.capture(ids, requires_grad=False).model_logits.clone()
    with toy_adapter.patch(1, lambda h: h + 5.0):
        out = toy_adapter.model(input_ids=ids.to(toy_adapter.device), use_cache=False).logits
    assert not torch.allclose(base, out)
