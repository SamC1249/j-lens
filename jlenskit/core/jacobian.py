"""Averaged input->output Jacobian estimator.

Implements  J_l = E_{prompt, t, t'>=t} [ d h_final,t' / d h_l,t ].

Key efficiency insight: the model is autoregressive, so d h_final,t' / d h_l,t = 0 for
t' < t; the t'>=t constraint is therefore free. Define g[i] = sum over batch and t' of
h_final[b, t', i]. Then d g[i] / d h_l = sum over t'>=t of the Jacobian, for every source
position t at once. A single (batched) backward per output dim i thus yields row i of J_l
for *every* layer l simultaneously. Fitting costs ~d_model backward passes over the corpus,
independent of sequence length.

The overall scale of J_l is irrelevant to decoded tokens because the lens applies the
model's final norm (RMSNorm/LayerNorm), which is invariant to positive rescaling; we still
divide by the number of (t, t') pairs to stay faithful to the "average over target
positions" definition.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from ..adapters.base import Adapter


def compute_layer_jacobians(
    adapter: Adapter,
    batches: Iterable[torch.Tensor],
    layers: list[int],
    chunk_size: int = 32,
    progress: bool = False,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Estimate ``J_l`` for each layer in ``layers`` by accumulating over ``batches``.

    ``batches`` yields ``input_ids`` tensors of shape ``[batch, seq]`` (fixed ``seq``,
    no padding). Returns ``(jacobians, n_sequences, n_pairs)`` where ``jacobians[l]`` is a
    ``float32 [d_model, d_model]`` matrix on CPU.
    """
    D = adapter.d_model
    device = adapter.device
    J = {l: torch.zeros(D, D, dtype=torch.float32, device=device) for l in layers}

    n_sequences = 0
    n_pairs = 0
    eye = torch.eye(D, device=device)

    for step, input_ids in enumerate(batches):
        input_ids = input_ids.to(device)
        b, s = input_ids.shape
        n_sequences += b
        n_pairs += b * s * (s + 1) // 2  # valid (t, t'>=t) pairs per sequence

        cap = adapter.capture(input_ids, requires_grad=True)
        residual_list = [cap.residuals[l] for l in layers]
        # g[i] = sum over batch and target positions of the final residual's i-th dim.
        g = cap.final_residual.reshape(-1, D).sum(dim=0)  # [D]

        for start in range(0, D, chunk_size):
            end = min(start + chunk_size, D)
            grad_outputs = eye[start:end]  # [chunk, D] one-hot rows
            grads = torch.autograd.grad(
                outputs=g,
                inputs=residual_list,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                retain_graph=True,
            )
            for l, gr in zip(layers, grads):
                # gr: [chunk, b, s, D] -> sum over batch and source positions -> [chunk, D]
                J[l][start:end] += gr.reshape(end - start, -1, D).sum(dim=1).float()

        # free the graph for this batch
        del cap, residual_list, g, grads
        if progress:
            print(f"[jacobian] batch {step + 1}: seqs={n_sequences} pairs={n_pairs}", flush=True)

    for l in layers:
        J[l] = (J[l] / max(n_pairs, 1)).cpu()

    return J, n_sequences, n_pairs
