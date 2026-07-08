"""Shared test fixtures: a tiny random-init model so tests need no downloads or GPU."""

from __future__ import annotations

import pytest
import torch


class StubTokenizer:
    """A minimal deterministic tokenizer over a small vocab (whitespace + hashing)."""

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size

    def _tok(self, text: str) -> list[int]:
        words = text.split() or [text]
        return [(abs(hash(w)) % (self.vocab_size - 1)) + 1 for w in words]

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = self._tok(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(self, ids):
        return f"<{int(ids[0])}>"


@pytest.fixture(scope="session")
def toy_adapter():
    from transformers import LlamaConfig, LlamaForCausalLM

    from jlenskit.adapters.base import Adapter
    from jlenskit.adapters.spec import autodetect

    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=64,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).to(torch.float32).eval()
    return Adapter(model, StubTokenizer(64), spec=autodetect(model))


@pytest.fixture(scope="session")
def toy_batches():
    g = torch.Generator().manual_seed(1)
    return [torch.randint(1, 64, (2, 8), generator=g) for _ in range(4)]


@pytest.fixture(scope="session")
def toy_lens(toy_adapter, toy_batches):
    from jlenskit.core.lens import JacobianLens

    return JacobianLens.fit(toy_adapter, toy_batches, layers=[0, 1, 2], chunk_size=8, seed=0)
