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
            corpus_spec=None, progress=False):
        if seed is not None:
            torch.manual_seed(seed)
        if layers is None:
            layers = list(range(adapter.n_layers))
        d = adapter.d_model
        device = adapter.device

        # Cache residuals + model target log-probs once (no grad through the model).
        cached = []  # list of (residuals: {l: [P, D]}, target_logp: [P, vocab])
        with torch.no_grad():
            for input_ids in batches:
                cap = adapter.capture(input_ids, requires_grad=False)
                target = torch.log_softmax(cap.model_logits.float(), dim=-1).reshape(-1, adapter.vocab_size)
                res = {l: cap.residuals[l].float().reshape(-1, d) for l in layers}
                cached.append((res, target))

        A = {l: torch.zeros(d, d, device=device, requires_grad=True) for l in layers}
        b = {l: torch.zeros(d, device=device, requires_grad=True) for l in layers}
        params = [A[l] for l in layers] + [b[l] for l in layers]
        opt = torch.optim.Adam(params, lr=lr)

        for step in range(n_steps):
            opt.zero_grad()
            loss = torch.zeros((), device=device)
            for res, target in cached:
                target_p = target.exp()
                for l in layers:
                    h = res[l].to(device)
                    transported = h + h @ A[l].t() + b[l]
                    lens_logp = torch.log_softmax(
                        adapter.to_logits(transported.to(adapter.dtype)).float(), dim=-1
                    )
                    # forward KL(model || lens), summed over vocab, mean over positions
                    loss = loss + (target_p * (target - lens_logp)).sum(dim=-1).mean()
            loss.backward()
            opt.step()
            if progress and step % 50 == 0:
                print(f"[tuned] step {step} loss {float(loss):.4f}")

        meta = LensMeta(
            model_id=getattr(adapter.model.config, "_name_or_path", "unknown"),
            model_revision=None, d_model=d, vocab_size=adapter.vocab_size,
            n_layers=adapter.n_layers, layers_fit=sorted(layers),
            n_prompts=len(cached), n_positions=sum(t.shape[0] for _, t in cached),
            corpus_spec=corpus_spec or {}, fit_config={"n_steps": n_steps, "lr": lr},
            seed=seed, jlenskit_version="0.1.0", extra={"lens_type": "tuned"},
        )
        return cls({l: A[l].detach().cpu() for l in layers},
                   {l: b[l].detach().cpu() for l in layers}, meta)

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
