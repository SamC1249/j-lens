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
from ..jspace import decompose

__all__ = [
    "topk_accuracy",
    "excess_kurtosis",
    "autocorrelation",
    "effective_dimension",
    "logit_fidelity",
    "jspace_variance_explained",
    "forward_kl",
    "entropy",
    "evaluate",
    "metrics_to_rows",
]


@torch.no_grad()
def _lens_logits_for_batch(
    adapter: Adapter,
    lens,
    input_ids: torch.Tensor,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Compute lens logits ``[b, s, vocab]`` for each layer of one batch.

    Captures residuals without grad, transports each layer's residual through the
    lens's ``transport`` method, then maps to logits via the model's real final-norm
    + unembedding. Works for any ``Lens`` implementation.
    """
    cap = adapter.capture(input_ids, requires_grad=False)
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        residual = cap.residuals[layer].float()  # [b, s, D]
        transported = lens.transport(layer, residual)  # any Lens
        out[layer] = adapter.to_logits(transported.to(adapter.dtype)).float()
    return out


@torch.no_grad()
def topk_accuracy(
    adapter: Adapter,
    lens,
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


@torch.no_grad()
def logit_fidelity(
    adapter: Adapter,
    lens: JacobianLens,
    batches,
    layers: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Per-layer faithfulness of the linear lens to the model's ACTUAL output.

    The lens approximates the model's output by transporting a layer-``l`` residual
    through ``J_l`` and decoding it; this measures how well that read-out matches the
    model's true next-token distribution at each position:

    - ``logit_kl``        — mean ``KL(model || lens)`` over the vocabulary (nats). 0 means
      the lens reproduces the model's output distribution exactly; large values mean the
      lens is telling a different story than the model.
    - ``top1_agreement``  — fraction of positions where the lens's argmax token equals the
      model's argmax token (exact match).

    Both live in logit space on purpose: the decoded read-out is invariant to the overall
    scale of ``J_l`` (the final norm removes it) and the lens drops the affine offset, so a
    raw residual-space ``||J_l h - h_final||`` would look large even for a faithful lens.

    Run this on HELD-OUT text (not the fit corpus) to catch a lens that is only faithful
    near the operating point it was fit on. Returns ``{layer: {metric: value}}``.
    """
    layers = layers or lens.layers
    kl_sum = {l: 0.0 for l in layers}
    hits = {l: 0 for l in layers}
    total = 0
    for input_ids in batches:
        cap = adapter.capture(input_ids, requires_grad=False)
        model_logp = torch.log_softmax(cap.model_logits.float(), dim=-1)  # [b, s, vocab]
        model_p = model_logp.exp()
        model_top1 = model_logp.argmax(dim=-1)  # [b, s]
        total += model_top1.numel()
        for l in layers:
            residual = cap.residuals[l].float()
            J = lens.jacobians[l].to(residual.device, torch.float32)
            transported = residual @ J.t()
            lens_logp = torch.log_softmax(
                adapter.to_logits(transported.to(adapter.dtype)).float(), dim=-1
            )
            kl = (model_p * (model_logp - lens_logp)).sum(dim=-1)  # [b, s]
            kl_sum[l] += float(kl.sum().item())
            hits[l] += int((lens_logp.argmax(dim=-1) == model_top1).sum().item())
    return {
        l: {
            "logit_kl": (kl_sum[l] / total if total else 0.0),
            "top1_agreement": (hits[l] / total if total else 0.0),
        }
        for l in layers
    }


@torch.no_grad()
def jspace_variance_explained(
    adapter: Adapter,
    lens: JacobianLens,
    batches,
    layers: list[int] | None = None,
    k: int = 16,
    max_positions: int = 256,
) -> dict[int, float]:
    """Fraction of residual-stream variance captured by the top-``k`` J-space concepts.

    For each sampled residual ``h`` at a layer, :func:`jlenskit.jspace.decompose`
    reconstructs it from ``k`` non-negative *verbalizable* lens vectors; the fraction of
    energy explained is ``1 - sum||h - recon||^2 / sum||h||^2`` aggregated over positions.
    This is the toolkit's operationalization of the paper's "global workspace footprint"
    — the fraction of activation variance that is verbalizable — which the paper reports
    is modest (~10%) and concentrated in the middle block of the network.

    ``max_positions`` caps the number of positions decomposed (matching pursuit runs per
    position, so this dominates cost). Returns ``{layer: explained_fraction}`` in [0, 1].
    """
    layers = layers or lens.layers
    res_energy = {l: 0.0 for l in layers}
    tot_energy = {l: 0.0 for l in layers}
    seen = 0
    for input_ids in batches:
        if seen >= max_positions:
            break
        cap = adapter.capture(input_ids, requires_grad=False)
        take = min(
            cap.residuals[layers[0]].reshape(-1, adapter.d_model).shape[0],
            max_positions - seen,
        )
        for l in layers:
            H = cap.residuals[l].reshape(-1, adapter.d_model)[:take].float()  # [take, D]
            for j in range(take):
                h = H[j]
                h_energy = float(h @ h)  # ||h||^2
                if h_energy <= 0:
                    continue
                d = decompose(adapter, lens, h, l, k=k)
                res_energy[l] += (d.reconstruction_error**2) * h_energy
                tot_energy[l] += h_energy
        seen += take
    return {
        l: (1.0 - res_energy[l] / tot_energy[l]) if tot_energy[l] > 0 else 0.0 for l in layers
    }


@torch.no_grad()
def forward_kl(adapter, lens, batches, layers=None):
    """Mean KL(model_final || lens_l) per layer (nats). Lower = more coherent."""
    layers = layers or list(lens.layers)
    kl_sum = {l: 0.0 for l in layers}
    total = 0
    for input_ids in batches:
        cap = adapter.capture(input_ids, requires_grad=False)
        model_logp = torch.log_softmax(cap.model_logits.float(), dim=-1)
        model_p = model_logp.exp()
        total += model_p.reshape(-1, adapter.vocab_size).shape[0]
        for l in layers:
            h = cap.residuals[l].float()
            lens_logp = torch.log_softmax(
                adapter.to_logits(lens.transport(l, h).to(adapter.dtype)).float(), dim=-1
            )
            kl = (model_p * (model_logp - lens_logp)).sum(dim=-1)  # [b, s]
            kl_sum[l] += float(kl.sum().item())
    return {l: (kl_sum[l] / total if total else 0.0) for l in layers}


@torch.no_grad()
def entropy(adapter, lens, batches, layers=None):
    """Mean Shannon entropy (nats) of the lens distribution per layer."""
    layers = layers or list(lens.layers)
    ent_sum = {l: 0.0 for l in layers}
    total = 0
    for input_ids in batches:
        logits = _lens_logits_for_batch(adapter, lens, input_ids, layers)
        for l in layers:
            logp = torch.log_softmax(logits[l], dim=-1)
            ent = -(logp.exp() * logp).sum(dim=-1)  # [b, s]
            ent_sum[l] += float(ent.sum().item())
            n = ent.numel()
        total += n
    return {l: (ent_sum[l] / total if total else 0.0) for l in layers}


def evaluate(
    adapter: Adapter,
    lens: JacobianLens,
    batches,
    k: int = 5,
) -> dict[str, dict[int, float]]:
    """Run all lens metrics and return them keyed by metric name.

    Returns ``topk_accuracy``, ``excess_kurtosis``, ``autocorrelation``,
    ``effective_dimension``, and the logit-space fidelity of the lens to the model
    (``fidelity_logit_kl`` and ``fidelity_top1_agreement``), each mapping layer -> float.
    (``jspace_variance_explained`` is not included by default — it is much costlier and is
    most meaningful on held-out text; call it directly.)
    """
    batch_list = list(batches)
    fid = logit_fidelity(adapter, lens, batch_list)
    return {
        "topk_accuracy": topk_accuracy(adapter, lens, batch_list, k=k),
        "excess_kurtosis": excess_kurtosis(adapter, lens, batch_list),
        "autocorrelation": autocorrelation(adapter, lens, batch_list),
        "effective_dimension": effective_dimension(lens),
        "fidelity_logit_kl": {l: fid[l]["logit_kl"] for l in fid},
        "fidelity_top1_agreement": {l: fid[l]["top1_agreement"] for l in fid},
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
