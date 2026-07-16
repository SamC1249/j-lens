"""Tuned lens (Belrose et al. 2023): a per-layer affine translator, trained by
distilling to the model's own final output. At zero init it equals the logit lens."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..core.types import LensMeta


class TunedLens:
    name = "tuned"

    def __init__(self, A: dict[int, torch.Tensor], b: dict[int, torch.Tensor], meta: LensMeta):
        self.A = A
        self.b = b
        self.meta = meta

    @property
    def layers(self) -> list[int]:
        return sorted(self.A)

    @classmethod
    def zeros(cls, layers, d_model: int, meta: LensMeta) -> TunedLens:
        A = {l: torch.zeros(d_model, d_model) for l in layers}
        b = {l: torch.zeros(d_model) for l in layers}
        return cls(A, b, meta)

    @classmethod
    def fit(cls, adapter, batches, layers=None, n_steps=250, lr=1e-3, seed=None,
            corpus_spec=None, progress=False, minibatch=512):
        if seed is not None:
            torch.manual_seed(seed)
        if layers is None:
            layers = list(range(adapter.n_layers))
        d = adapter.d_model
        device = adapter.device

        # Cache residuals per layer + the model's final log-probs (shared across layers,
        # since the target is the model's own output). No grad flows through the model.
        res_by_layer = {l: [] for l in layers}
        target_chunks = []
        with torch.no_grad():
            for input_ids in batches:
                cap = adapter.capture(input_ids, requires_grad=False)
                target_chunks.append(
                    torch.log_softmax(cap.model_logits.float(), dim=-1).reshape(-1, adapter.vocab_size)
                )
                for l in layers:
                    res_by_layer[l].append(cap.residuals[l].float().reshape(-1, d))
        target = torch.cat(target_chunks, dim=0).to(device)            # [P, vocab]
        target_p = target.exp()
        H = {l: torch.cat(res_by_layer[l], dim=0).to(device) for l in layers}  # [P, D]
        P = target.shape[0]

        # Train each layer's affine translator independently (objective is separable per
        # layer), minibatching positions so the retained vocab-space graph stays bounded.
        A, b = {}, {}
        for l in layers:
            A_l = torch.zeros(d, d, device=device, requires_grad=True)
            b_l = torch.zeros(d, device=device, requires_grad=True)
            opt = torch.optim.Adam([A_l, b_l], lr=lr)
            gen = torch.Generator()
            if seed is not None:
                gen.manual_seed(seed + l)
            for step in range(n_steps):
                if minibatch and P > minibatch:
                    idx = torch.randint(0, P, (minibatch,), generator=gen).to(device)
                    h, tp, tlp = H[l][idx], target_p[idx], target[idx]
                else:
                    h, tp, tlp = H[l], target_p, target
                opt.zero_grad()
                transported = h + h @ A_l.t() + b_l
                lens_logp = torch.log_softmax(
                    adapter.to_logits(transported.to(adapter.dtype)).float(), dim=-1
                )
                loss = (tp * (tlp - lens_logp)).sum(dim=-1).mean()  # forward KL(model || lens)
                loss.backward()
                opt.step()
                if progress and step % 50 == 0:
                    print(f"[tuned] layer {l} step {step} loss {loss.item():.4f}")
            A[l] = A_l.detach().cpu()
            b[l] = b_l.detach().cpu()

        meta = LensMeta(
            model_id=getattr(adapter.model.config, "_name_or_path", "unknown"),
            model_revision=None, d_model=d, vocab_size=adapter.vocab_size,
            n_layers=adapter.n_layers, layers_fit=sorted(layers),
            n_prompts=len(target_chunks), n_positions=P,
            corpus_spec=corpus_spec or {},
            fit_config={"n_steps": n_steps, "lr": lr, "minibatch": minibatch},
            seed=seed, jlenskit_version="0.1.0", extra={"lens_type": "tuned"},
        )
        return cls(A, b, meta)

    def transport(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        A = self.A[layer].to(h.device, torch.float32)
        b = self.b[layer].to(h.device, torch.float32)
        h = h.to(torch.float32)
        return h + h @ A.t() + b

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for l in self.layers:
            tensors[f"A_{l}"] = self.A[l].contiguous().cpu()
            tensors[f"b_{l}"] = self.b[l].contiguous().cpu()
        meta = {"jlenskit_meta": json.dumps(self.meta.__dict__), "lens_type": "tuned"}
        save_file(tensors, str(path), metadata=meta)
        return path

    @classmethod
    def load(cls, path, device: str = "cpu") -> TunedLens:
        path = Path(path)
        tensors = load_file(str(path), device=device)
        A = {int(k[2:]): v for k, v in tensors.items() if k.startswith("A_")}
        b = {int(k[2:]): v for k, v in tensors.items() if k.startswith("b_")}
        import safetensors

        with safetensors.safe_open(str(path), framework="pt") as f:
            raw = f.metadata() or {}
        meta_dict = json.loads(raw.get("jlenskit_meta", "{}"))
        meta = LensMeta(**{k: meta_dict.get(k) for k in LensMeta.__dataclass_fields__})
        return cls(A, b, meta)
