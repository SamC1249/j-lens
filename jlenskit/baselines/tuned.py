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
