"""Common lens interface: every lens is transport -> to_logits -> decode.

logit lens (identity transport), tuned lens (learned affine), and the Jacobian
lens all implement ``transport`` and reuse the apply/decode plumbing here so the
decode loop and LensResult shape live in exactly one place.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from ..adapters.base import Adapter
from .types import DecodedLayer, LensResult


@runtime_checkable
class Lens(Protocol):
    name: str
    layers: list[int]

    def transport(self, layer: int, h: torch.Tensor) -> torch.Tensor: ...


@torch.no_grad()
def apply_lens(
    lens: Lens,
    adapter: Adapter,
    prompt: str,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    top_k: int = 10,
    decode: bool = True,
) -> LensResult:
    input_ids = adapter.encode(prompt)
    seq = input_ids.shape[1]
    if positions is None:
        positions = [-1]
    pos = [p % seq for p in positions]
    layers = layers or list(lens.layers)

    cap = adapter.capture(input_ids, requires_grad=False)
    lens_logits: dict[int, torch.Tensor] = {}
    for l in layers:
        h = cap.residuals[l][0, pos].to(torch.float32)  # [n_pos, D]
        transported = lens.transport(l, h).to(adapter.dtype)
        lens_logits[l] = adapter.to_logits(transported).to(torch.float32)

    result = LensResult(
        prompt=prompt,
        positions=pos,
        lens_logits=lens_logits,
        model_logits=cap.model_logits[0, pos].to(torch.float32),
    )
    if decode:
        result.decoded = decode_result(adapter, result, top_k)
    return result


def decode_result(adapter: Adapter, result: LensResult, top_k: int) -> dict[int, list[DecodedLayer]]:
    decoded: dict[int, list[DecodedLayer]] = {}
    for pi, _p in enumerate(result.positions):
        per_layer = []
        for l in sorted(result.lens_logits):
            probs = torch.softmax(result.lens_logits[l][pi], dim=-1)
            top = probs.topk(top_k)
            ids = top.indices.tolist()
            per_layer.append(
                DecodedLayer(layer=l, token_ids=ids, tokens=adapter.decode_tokens(ids), scores=top.values.tolist())
            )
        decoded[pi] = per_layer
    return decoded
