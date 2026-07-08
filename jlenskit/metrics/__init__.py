"""Diagnostic metrics for a fitted Jacobian lens.

These metrics quantify *where* and *how well* the lens reads the residual stream,
reproducing the diagnostics used to locate the model's "global workspace":

- ``topk_accuracy``   — agreement between the lens read-out and the model's own
  next-token prediction. Climbs monotonically in later layers as the residual
  stream commits to an output.
- ``excess_kurtosis`` — peakedness of the lens logit distribution over the vocab.
  Peaks in the mid "workspace" layers, where a few tokens dominate.
- ``autocorrelation`` — how often the lens top-1 token stays fixed across adjacent
  positions. Also peaks in workspace layers, reflecting a stable broadcast state.
- ``effective_dimension`` — participation ratio of the Jacobian's singular values.
  Rises at workspace onset as the transport map spreads energy across more modes.

All metric functions aggregate over *every* position in *every* batch and return a
``dict`` keyed by lens layer index. Batched compute runs under ``torch.no_grad()``
and processes one batch at a time to keep memory bounded.
"""

from __future__ import annotations

import torch

from ..adapters.base import Adapter
from ..core.lens import JacobianLens

__all__ = [
    "topk_accuracy",
    "excess_kurtosis",
    "autocorrelation",
    "effective_dimension",
    "evaluate",
    "metrics_to_rows",
]


@torch.no_grad()
def _lens_logits_for_batch(
    adapter: Adapter,
    lens: JacobianLens,
    input_ids: torch.Tensor,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Compute lens logits ``[b, s, vocab]`` for each layer of one batch.

    Captures residuals without grad, transports each layer's residual through its
    Jacobian, then maps to logits via the model's real final-norm + unembedding.
    """
    cap = adapter.capture(input_ids, requires_grad=False)
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        residual = cap.residuals[layer].float()  # [b, s, D]
        J = lens.jacobians[layer].to(residual.device, torch.float32)  # [D, D]
        transported = residual @ J.t()  # [b, s, D], final ~= J h
        logits = adapter.to_logits(transported.to(adapter.dtype)).float()
        out[layer] = logits  # [b, s, vocab]
    return out


@torch.no_grad()
def topk_accuracy(
    adapter: Adapter,
    lens: JacobianLens,
    batches,
    k: int = 5,
) -> dict[int, float]:
    """Fraction of positions where the model's top-1 token is in the lens top-k.

    At each position we take the model's own argmax next-token prediction and check
    whether it appears among the lens's top-``k`` token ids at that layer/position.
    High, rising values in later layers mean the lens is reading a prediction that
    matches the model's committed output.
    """
    layers = lens.layers
    hits = {l: 0 for l in layers}
    total = 0
    for input_ids in batches:
        cap = adapter.capture(input_ids, requires_grad=False)
        model_top1 = cap.model_logits.float().argmax(dim=-1)  # [b, s]
        n_pos = model_top1.numel()
        total += n_pos
        lens_logits = _lens_logits_for_batch(adapter, lens, input_ids, layers)
        for l in layers:
            topk_ids = lens_logits[l].topk(k, dim=-1).indices  # [b, s, k]
            match = (topk_ids == model_top1.unsqueeze(-1)).any(dim=-1)  # [b, s]
            hits[l] += int(match.sum().item())
    return {l: (hits[l] / total if total else 0.0) for l in layers}


@torch.no_grad()
def excess_kurtosis(
    adapter: Adapter,
    lens: JacobianLens,
    batches,
) -> dict[int, float]:
    """Mean excess kurtosis of the lens logit vector over the vocabulary.

    For each position we compute ``E[(x-mu)^4]/var^2 - 3`` over the vocab dimension,
    then average across all positions. Large positive values mean a few tokens
    dominate the read-out; this peaks in the mid "workspace" layers.
    """
    layers = lens.layers
    sums = {l: 0.0 for l in layers}
    total = 0
    for input_ids in batches:
        lens_logits = _lens_logits_for_batch(adapter, lens, input_ids, layers)
        for l in layers:
            x = lens_logits[l]  # [b, s, vocab]
            flat = x.reshape(-1, x.shape[-1])  # [P, vocab]
            mu = flat.mean(dim=-1, keepdim=True)
            centered = flat - mu
            var = centered.pow(2).mean(dim=-1)
            m4 = centered.pow(4).mean(dim=-1)
            kurt = m4 / var.pow(2) - 3.0  # [P]
            kurt = torch.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)
            sums[l] += float(kurt.sum().item())
            n_pos = flat.shape[0]
        total += n_pos
    return {l: (sums[l] / total if total else 0.0) for l in layers}


@torch.no_grad()
def autocorrelation(
    adapter: Adapter,
    lens: JacobianLens,
    batches,
) -> dict[int, float]:
    """Fraction of adjacent position pairs sharing the same lens top-1 token.

    For each sequence, compares the lens top-1 token id at position ``t`` and
    ``t+1``; averages over all adjacent pairs across all sequences/batches. High
    values mean a stable broadcast token, peaking in workspace layers.
    """
    layers = lens.layers
    same = {l: 0 for l in layers}
    total_pairs = 0
    for input_ids in batches:
        lens_logits = _lens_logits_for_batch(adapter, lens, input_ids, layers)
        for l in layers:
            top1 = lens_logits[l].argmax(dim=-1)  # [b, s]
            if top1.shape[1] < 2:
                continue
            eq = top1[:, 1:] == top1[:, :-1]  # [b, s-1]
            same[l] += int(eq.sum().item())
        # pair count is identical across layers for a given batch
        b, s = input_ids.shape[0], input_ids.shape[1]
        if s >= 2:
            total_pairs += b * (s - 1)
    return {l: (same[l] / total_pairs if total_pairs else 0.0) for l in layers}


def effective_dimension(lens: JacobianLens) -> dict[int, float]:
    """Participation ratio of each Jacobian's singular values.

    ``PR = (sum s_i)^2 / sum(s_i^2)``, bounded in ``(0, d_model]``. It measures how
    many singular directions the transport map effectively uses; it rises at
    workspace onset as the map engages more modes.
    """
    out: dict[int, float] = {}
    for l in lens.layers:
        J = lens.jacobians[l].to(torch.float32)
        s = torch.linalg.svdvals(J)
        num = s.sum().pow(2)
        den = s.pow(2).sum()
        pr = (num / den) if den > 0 else torch.tensor(0.0)
        out[l] = float(pr.item())
    return out


def evaluate(
    adapter: Adapter,
    lens: JacobianLens,
    batches,
    k: int = 5,
) -> dict[str, dict[int, float]]:
    """Run all lens metrics and return them keyed by metric name.

    Returns ``{"topk_accuracy", "excess_kurtosis", "autocorrelation",
    "effective_dimension"}``, each mapping layer index -> float.
    """
    batch_list = list(batches)
    return {
        "topk_accuracy": topk_accuracy(adapter, lens, batch_list, k=k),
        "excess_kurtosis": excess_kurtosis(adapter, lens, batch_list),
        "autocorrelation": autocorrelation(adapter, lens, batch_list),
        "effective_dimension": effective_dimension(lens),
    }


def metrics_to_rows(metrics: dict) -> list[dict]:
    """Flatten an ``evaluate`` result into tidy rows for tabular export.

    Each row is ``{"metric": name, "layer": l, "value": v}``.
    """
    rows: list[dict] = []
    for name, per_layer in metrics.items():
        for layer, value in per_layer.items():
            rows.append({"metric": name, "layer": int(layer), "value": float(value)})
    return rows
