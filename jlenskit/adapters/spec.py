"""ModelSpec: the small capability config that makes a model interpretable.

Adding support for a new architecture is a *data* problem: fill in (or let
``autodetect`` infer) the handful of module paths and flags below. Because we hook
the real model to capture residuals and call the real final-norm + unembedding,
architecture quirks (RoPE, GQA, MoE routing, Gemma embed-scaling) are handled by
the model itself and need no special code here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

# Common attribute paths, tried in order during autodetection.
_LAYERS_PATHS = ["model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"]
_FINAL_NORM_PATHS = ["model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm", "model.decoder.final_layer_norm"]
_UNEMBED_PATHS = ["lm_head", "embed_out"]
_EMBED_PATHS = ["model.embed_tokens", "transformer.wte", "gpt_neox.embed_in", "model.decoder.embed_tokens"]


@dataclass
class ModelSpec:
    """Describes how to reach the interpretability-relevant parts of a model."""

    family: str
    layers_path: str
    final_norm_path: str
    unembed_path: str
    embed_path: str
    norm_type: str = "rmsnorm"  # "rmsnorm" | "layernorm"; informational
    tie_embeddings: bool = False
    quirks: dict[str, object] = field(default_factory=dict)


def _get_by_path(module: nn.Module, path: str):
    obj = module
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _first_present(model: nn.Module, paths: list[str]) -> str | None:
    for p in paths:
        if _get_by_path(model, p) is not None:
            return p
    return None


def autodetect(model: nn.Module, family: str = "auto") -> ModelSpec:
    """Infer a ModelSpec from a loaded HF model by probing common attribute paths.

    Raises a clear error if a required component cannot be found, so unsupported
    architectures fail loudly rather than silently misbehaving.
    """
    layers_path = _first_present(model, _LAYERS_PATHS)
    final_norm_path = _first_present(model, _FINAL_NORM_PATHS)
    unembed_path = _first_present(model, _UNEMBED_PATHS)
    embed_path = _first_present(model, _EMBED_PATHS)

    missing = [
        name
        for name, val in [
            ("decoder layers", layers_path),
            ("final norm", final_norm_path),
            ("unembedding", unembed_path),
            ("token embedding", embed_path),
        ]
        if val is None
    ]
    if missing:
        raise ValueError(
            f"Could not autodetect {', '.join(missing)} for model "
            f"{type(model).__name__}. Register an explicit ModelSpec in "
            f"jlenskit.adapters.registry for this architecture."
        )

    final_norm = _get_by_path(model, final_norm_path)
    norm_type = "layernorm" if isinstance(final_norm, nn.LayerNorm) else "rmsnorm"

    cfg = getattr(model, "config", None)
    tie = bool(getattr(cfg, "tie_word_embeddings", False)) if cfg is not None else False

    quirks: dict[str, object] = {}
    # Gemma scales embeddings by sqrt(d_model) and soft-caps final logits. The
    # scaling is applied inside the model's own forward (which we hook), so it needs
    # no handling here. We record the soft-cap so callers may optionally apply it.
    if cfg is not None and getattr(cfg, "final_logit_softcapping", None):
        quirks["final_logit_softcapping"] = float(cfg.final_logit_softcapping)

    if family == "auto":
        family = getattr(cfg, "model_type", "unknown") if cfg is not None else "unknown"

    return ModelSpec(
        family=family,
        layers_path=layers_path,
        final_norm_path=final_norm_path,
        unembed_path=unembed_path,
        embed_path=embed_path,
        norm_type=norm_type,
        tie_embeddings=tie,
        quirks=quirks,
    )
