"""Baseline read-outs to compare against the Jacobian lens.

The J-lens is, in the paper's framing, a *principled refinement of the logit lens*:
the logit lens assumes a hidden state already lives in the final-layer coordinate
system and decodes it directly, whereas the J-lens first transports it there via the
per-layer Jacobian ``J_l``. Being able to run both on the same activation — with the
same norm, unembedding, tokenizer, and output shape — is the most common sanity check
a researcher will want, so it ships as a first-class function.

    logit_lens_l(h) = softmax( W_U . norm(h) )          # no transport
    jacobian_lens_l(h) = softmax( W_U . norm(J_l . h) )  # J-lens (see core.lens)
"""

from __future__ import annotations

import torch

from .adapters.base import Adapter
from .core.types import DecodedLayer, LensResult


@torch.no_grad()
def logit_lens(
    adapter: Adapter,
    prompt: str,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    top_k: int = 10,
    decode: bool = True,
) -> LensResult:
    """Apply the classic logit lens: decode each layer's residual with no transport.

    Mirrors :meth:`JacobianLens.apply` exactly (same positions convention, same
    ``LensResult`` shape) so results are directly comparable and reusable by
    :mod:`jlenskit.viz`. ``layers`` defaults to all residual points ``0..n_layers-1``.
    """
    input_ids = adapter.encode(prompt)
    seq = input_ids.shape[1]
    if positions is None:
        positions = [-1]
    pos = [p % seq for p in positions]
    if layers is None:
        layers = list(range(adapter.n_layers))

    cap = adapter.capture(input_ids, requires_grad=False)

    lens_logits: dict[int, torch.Tensor] = {}
    for l in layers:
        h = cap.residuals[l][0, pos].to(adapter.dtype)  # [n_pos, D], no Jacobian applied
        lens_logits[l] = adapter.to_logits(h).to(torch.float32)

    model_logits = cap.model_logits[0, pos].to(torch.float32)
    result = LensResult(
        prompt=prompt,
        positions=pos,
        lens_logits=lens_logits,
        model_logits=model_logits,
    )
    if decode:
        result.decoded = _decode(adapter, result, top_k)
    return result


def _decode(adapter: Adapter, result: LensResult, top_k: int) -> dict[int, list[DecodedLayer]]:
    """Softmax + top-k per layer/position (shared shape with ``JacobianLens._decode``)."""
    decoded: dict[int, list[DecodedLayer]] = {}
    for pi, _p in enumerate(result.positions):
        per_layer = []
        for l in sorted(result.lens_logits):
            probs = torch.softmax(result.lens_logits[l][pi], dim=-1)
            top = probs.topk(top_k)
            ids = top.indices.tolist()
            per_layer.append(
                DecodedLayer(
                    layer=l,
                    token_ids=ids,
                    tokens=adapter.decode_tokens(ids),
                    scores=top.values.tolist(),
                )
            )
        decoded[pi] = per_layer
    return decoded
