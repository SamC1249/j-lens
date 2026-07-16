"""JacobianLens: fit, apply, decode, persist, and merge lenses."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..adapters.base import Adapter
from .jacobian import compute_layer_jacobians
from .lens_base import apply_lens, decode_result
from .types import LensMeta, LensResult

_VERSION = "0.1.0"


class JacobianLens:
    """A fitted Jacobian lens: one transport matrix ``J_l`` per layer."""

    name = "jacobian"

    def __init__(self, jacobians: dict[int, torch.Tensor], meta: LensMeta):
        self.jacobians = jacobians
        self.meta = meta

    @property
    def layers(self) -> list[int]:
        return sorted(self.jacobians)

    def transport(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        J = self.jacobians[layer].to(h.device, torch.float32)
        return h.to(torch.float32) @ J.t()

    # -- fitting --------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        adapter: Adapter,
        batches,
        layers: list[int] | None = None,
        chunk_size: int = 32,
        seed: int | None = None,
        corpus_spec: dict | None = None,
        fit_config: dict | None = None,
        progress: bool = False,
    ) -> JacobianLens:
        """Fit a lens by estimating the averaged Jacobian over ``batches``.

        ``batches`` is an iterable of ``input_ids`` tensors (fixed-length, no padding).
        It may be a list (so it can be consumed once). ``layers`` defaults to all
        residual points except the trivial final one.
        """
        if seed is not None:
            torch.manual_seed(seed)
        if layers is None:
            layers = list(range(adapter.n_layers))  # 0..n_layers-1 (skip trivial identity at n_layers)

        batch_list = list(batches)
        jac, n_seq, n_pairs = compute_layer_jacobians(
            adapter, batch_list, layers, chunk_size=chunk_size, progress=progress
        )

        meta = LensMeta(
            model_id=getattr(adapter.model.config, "_name_or_path", "unknown"),
            model_revision=None,
            d_model=adapter.d_model,
            vocab_size=adapter.vocab_size,
            n_layers=adapter.n_layers,
            layers_fit=sorted(jac),
            n_prompts=n_seq,
            n_positions=n_pairs,
            corpus_spec=corpus_spec or {},
            fit_config={"chunk_size": chunk_size, **(fit_config or {})},
            seed=seed,
            jlenskit_version=_VERSION,
        )
        return cls(jac, meta)

    def merge(self, other: JacobianLens) -> JacobianLens:
        """Combine two lenses fit on disjoint corpus slices (weighted by pair count).

        Enables parallel/distributed fitting: fit on slices, then merge.
        """
        if set(self.jacobians) != set(other.jacobians):
            raise ValueError("Cannot merge lenses fit on different layer sets.")
        wa, wb = self.meta.n_positions, other.meta.n_positions
        total = max(wa + wb, 1)
        merged = {
            l: (self.jacobians[l] * wa + other.jacobians[l] * wb) / total for l in self.jacobians
        }
        # Deep-copy so we never mutate the operands' shared dicts, and record BOTH
        # slices' provenance — otherwise the merged lens silently claims it was fit on
        # only the left operand's corpus, breaking the manifest's reproducibility.
        meta = copy.deepcopy(self.meta)
        meta.n_prompts = self.meta.n_prompts + other.meta.n_prompts
        meta.n_positions = wa + wb
        meta.extra = {
            **(self.meta.extra or {}),
            "merged_from": [
                {"corpus_spec": self.meta.corpus_spec, "n_positions": wa, "n_prompts": self.meta.n_prompts},
                {"corpus_spec": other.meta.corpus_spec, "n_positions": wb, "n_prompts": other.meta.n_prompts},
            ],
        }
        return JacobianLens(merged, meta)

    # -- application ----------------------------------------------------------
    @torch.no_grad()
    def apply(
        self,
        adapter: Adapter,
        prompt: str,
        positions: list[int] | None = None,
        layers: list[int] | None = None,
        top_k: int = 10,
        decode: bool = True,
    ) -> LensResult:
        """Apply the lens to ``prompt`` at the given ``positions`` (defaults to last token)."""
        return apply_lens(self, adapter, prompt, positions=positions,
                          layers=layers or self.layers, top_k=top_k, decode=decode)

    def _decode(self, adapter: Adapter, result: LensResult, top_k: int) -> dict:
        return decode_result(adapter, result, top_k)

    # -- persistence ----------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Save the lens as a single ``.safetensors`` file with embedded metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {f"layer_{l}": self.jacobians[l].contiguous().cpu() for l in self.jacobians}
        metadata = {"jlenskit_meta": json.dumps(self.meta.__dict__)}
        save_file(tensors, str(path), metadata=metadata)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> JacobianLens:
        path = Path(path)
        tensors = load_file(str(path), device=device)
        jac = {int(k.split("_")[1]): v for k, v in tensors.items()}
        # safetensors metadata is read separately from the header
        import safetensors

        with safetensors.safe_open(str(path), framework="pt") as f:
            raw = f.metadata() or {}
        meta_dict = json.loads(raw.get("jlenskit_meta", "{}"))
        meta = LensMeta(**{k: meta_dict.get(k) for k in LensMeta.__dataclass_fields__})
        return cls(jac, meta)
