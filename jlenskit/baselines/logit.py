# jlenskit/baselines/logit.py
"""Logit lens: decode a residual with no transport (identity)."""

from __future__ import annotations

import torch

from ..adapters.base import Adapter
from ..core.lens_base import apply_lens
from ..core.types import LensResult


class LogitLens:
    name = "logit"

    def __init__(self, layers: list[int]):
        self.layers = list(layers)

    def transport(self, layer: int, h: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return h


@torch.no_grad()
def logit_lens(adapter: Adapter, prompt, positions=None, layers=None, top_k=10, decode=True) -> LensResult:
    lens = LogitLens(layers or list(range(adapter.n_layers)))
    return apply_lens(lens, adapter, prompt, positions=positions, layers=lens.layers, top_k=top_k, decode=decode)
