"""J-space decomposition and causal interventions.

J-space is the set of activations expressible as a sparse non-negative combination of
per-token *lens vectors*. The lens vector for token ``v`` at layer ``l`` is the direction in
the layer-``l`` residual stream that raises the lens read-out logit for ``v``:

    u_v = J_l^T (gamma ⊙ W_U[v])          (gamma = final-norm weight; linearized norm)

We never materialize the full [D, vocab] dictionary: correlations of a residual ``r`` with
every lens vector are  c_v = W_U[v] · (gamma ⊙ (J_l r)), a single unembed matvec.

- ``decompose`` finds the k concepts active in an activation via non-negative matching
  pursuit — the activation's "J-space coordinates".
- ``inject`` / ``swap`` edit the residual mid-forward (via Adapter.patch) and let the rest of
  the network react, so you can test whether a concept is causally used downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..adapters.base import Adapter
from ..core.lens import JacobianLens


@dataclass
class JSpaceCoord:
    token_id: int
    token: str
    coefficient: float  # non-negative weight along the unit lens vector


@dataclass
class Decomposition:
    layer: int
    coords: list[JSpaceCoord]
    reconstruction_error: float  # ||h - reconstruction|| / ||h||


@dataclass
class InterventionResult:
    layer: int
    position: int
    baseline_top: list[tuple[str, float]]
    intervened_top: list[tuple[str, float]]
    description: str


def _pieces(adapter: Adapter, lens: JacobianLens, layer: int):
    device = adapter.device
    J = lens.jacobians[layer].to(device=device, dtype=torch.float32)  # [D, D]
    W_U = adapter.unembed.weight.detach().to(device=device, dtype=torch.float32)  # [vocab, D]
    gamma = getattr(adapter.final_norm, "weight", None)
    gamma = gamma.detach().to(device=device, dtype=torch.float32) if gamma is not None else None
    return J, W_U, gamma


def _u_vector(J, W_U, gamma, v: int) -> torch.Tensor:
    w = W_U[v]
    if gamma is not None:
        w = gamma * w
    return J.t() @ w  # [D]


def _correlations(J, W_U, gamma, r) -> torch.Tensor:
    z = J @ r
    if gamma is not None:
        z = gamma * z
    return W_U @ z  # [vocab]


def _u_norms(J, W_U, gamma) -> torch.Tensor:
    Wg = W_U * gamma if gamma is not None else W_U  # [vocab, D]
    M = J @ J.t()  # [D, D]
    A = M @ Wg.t()  # [D, vocab]
    norms2 = (Wg.t() * A).sum(dim=0)  # [vocab]
    return norms2.clamp_min(1e-12).sqrt()


def decompose(
    adapter: Adapter,
    lens: JacobianLens,
    h: torch.Tensor,
    layer: int,
    k: int = 16,
    normalize: bool = True,
) -> Decomposition:
    """Decompose activation ``h`` (a [D] residual at ``layer``) into k lens vectors."""
    device = adapter.device
    h = h.detach().to(device=device, dtype=torch.float32).reshape(-1)
    h_norm = h.norm().clamp_min(1e-12)
    J, W_U, gamma = _pieces(adapter, lens, layer)
    norms = _u_norms(J, W_U, gamma) if normalize else torch.ones(W_U.shape[0], device=device)

    r = h.clone()
    coords: list[JSpaceCoord] = []
    for _ in range(k):
        c = _correlations(J, W_U, gamma, r) / norms  # correlation with unit lens vectors
        v = int(c.argmax())
        if float(c[v]) <= 0:
            break
        u = _u_vector(J, W_U, gamma, v)
        un = u / norms[v]
        alpha = float(r @ un)  # >= 0 because c[v] > 0
        if alpha <= 0:
            break
        r = r - alpha * un
        coords.append(JSpaceCoord(token_id=v, token=adapter.decode_tokens([v])[0], coefficient=alpha))

    err = float((r.norm() / h_norm).item())
    return Decomposition(layer=layer, coords=coords, reconstruction_error=err)


def decompose_prompt(
    adapter: Adapter,
    lens: JacobianLens,
    prompt: str,
    layer: int,
    position: int = -1,
    k: int = 16,
) -> Decomposition:
    """Convenience: decompose the residual at ``position`` of ``prompt``."""
    ids = adapter.encode(prompt)
    seq = ids.shape[1]
    pos = position % seq
    cap = adapter.capture(ids, requires_grad=False)
    h = cap.residuals[layer][0, pos]
    return decompose(adapter, lens, h, layer, k=k)


def _top_tokens(adapter: Adapter, logits: torch.Tensor, k: int) -> list[tuple[str, float]]:
    probs = torch.softmax(logits.float(), dim=-1)
    top = probs.topk(k)
    toks = adapter.decode_tokens(top.indices.tolist())
    return list(zip(toks, [round(float(v), 4) for v in top.values.tolist()]))


@torch.no_grad()
def inject(
    adapter: Adapter,
    lens: JacobianLens,
    prompt: str,
    layer: int,
    token: int | str,
    strength: float = 6.0,
    position: int = -1,
    top_k: int = 5,
) -> InterventionResult:
    """Add a token's lens vector into the residual at ``layer``/``position`` and re-run.

    ``strength`` is measured in units of the residual norm at that position, so it is
    comparable across models/layers.
    """
    token_id = token if isinstance(token, int) else int(adapter.encode(token, add_special_tokens=False)[0, -1])
    ids = adapter.encode(prompt)
    seq = ids.shape[1]
    pos = position % seq

    J, W_U, gamma = _pieces(adapter, lens, layer)
    u = _u_vector(J, W_U, gamma, token_id)
    un = u / u.norm().clamp_min(1e-12)

    base_logits = adapter.model(input_ids=ids.to(adapter.device), use_cache=False).logits[0, pos]

    def fn(hs):
        hs = hs.clone()
        scale = hs[0, pos].norm()
        hs[0, pos] = hs[0, pos] + (strength * scale / (seq**0.0)) * un.to(hs.dtype)
        return hs

    with adapter.patch(layer, fn):
        new_logits = adapter.model(input_ids=ids.to(adapter.device), use_cache=False).logits[0, pos]

    return InterventionResult(
        layer=layer,
        position=pos,
        baseline_top=_top_tokens(adapter, base_logits, top_k),
        intervened_top=_top_tokens(adapter, new_logits, top_k),
        description=f"inject '{adapter.decode_tokens([token_id])[0]}' @ layer {layer} strength {strength}",
    )


@torch.no_grad()
def swap(
    adapter: Adapter,
    lens: JacobianLens,
    prompt: str,
    layer: int,
    source: int | str,
    target: int | str,
    strength: float = 1.0,
    position: int = -1,
    top_k: int = 5,
) -> InterventionResult:
    """Swap a concept: remove the source lens direction, add the target one, then re-run."""
    def _tid(t):
        return t if isinstance(t, int) else int(adapter.encode(t, add_special_tokens=False)[0, -1])

    src, tgt = _tid(source), _tid(target)
    ids = adapter.encode(prompt)
    seq = ids.shape[1]
    pos = position % seq

    J, W_U, gamma = _pieces(adapter, lens, layer)
    us = _u_vector(J, W_U, gamma, src)
    ut = _u_vector(J, W_U, gamma, tgt)
    us_n = us / us.norm().clamp_min(1e-12)
    ut_n = ut / ut.norm().clamp_min(1e-12)

    base_logits = adapter.model(input_ids=ids.to(adapter.device), use_cache=False).logits[0, pos]

    def fn(hs):
        hs = hs.clone()
        h = hs[0, pos]
        proj = (h @ us_n.to(hs.dtype))
        mag = proj.clamp_min(hs[0, pos].norm() * 0.1)  # fall back to a fraction of the norm
        hs[0, pos] = h - proj * us_n.to(hs.dtype) + (strength * mag) * ut_n.to(hs.dtype)
        return hs

    with adapter.patch(layer, fn):
        new_logits = adapter.model(input_ids=ids.to(adapter.device), use_cache=False).logits[0, pos]

    s_tok = adapter.decode_tokens([src])[0]
    t_tok = adapter.decode_tokens([tgt])[0]
    return InterventionResult(
        layer=layer,
        position=pos,
        baseline_top=_top_tokens(adapter, base_logits, top_k),
        intervened_top=_top_tokens(adapter, new_logits, top_k),
        description=f"swap '{s_tok}' -> '{t_tok}' @ layer {layer} strength {strength}",
    )
