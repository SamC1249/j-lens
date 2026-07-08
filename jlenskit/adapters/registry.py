"""Known-family ModelSpecs.

Autodetection handles most HuggingFace decoders, but pinning known families gives
stable, tested behaviour and documents exactly what each architecture needs. Adding a
family is a ~15-line entry here, not new code.
"""

from __future__ import annotations

from .spec import ModelSpec

# Keyed by HF ``config.model_type``.
REGISTRY: dict[str, ModelSpec] = {
    "qwen2": ModelSpec(
        family="qwen2",
        layers_path="model.layers",
        final_norm_path="model.norm",
        unembed_path="lm_head",
        embed_path="model.embed_tokens",
        norm_type="rmsnorm",
    ),
    "qwen3": ModelSpec(
        family="qwen3",
        layers_path="model.layers",
        final_norm_path="model.norm",
        unembed_path="lm_head",
        embed_path="model.embed_tokens",
        norm_type="rmsnorm",
    ),
    "llama": ModelSpec(
        family="llama",
        layers_path="model.layers",
        final_norm_path="model.norm",
        unembed_path="lm_head",
        embed_path="model.embed_tokens",
        norm_type="rmsnorm",
    ),
    "mistral": ModelSpec(
        family="mistral",
        layers_path="model.layers",
        final_norm_path="model.norm",
        unembed_path="lm_head",
        embed_path="model.embed_tokens",
        norm_type="rmsnorm",
    ),
    "gemma": ModelSpec(
        family="gemma",
        layers_path="model.layers",
        final_norm_path="model.norm",
        unembed_path="lm_head",
        embed_path="model.embed_tokens",
        norm_type="rmsnorm",
    ),
    "gemma2": ModelSpec(
        family="gemma2",
        layers_path="model.layers",
        final_norm_path="model.norm",
        unembed_path="lm_head",
        embed_path="model.embed_tokens",
        norm_type="rmsnorm",
    ),
    "gpt2": ModelSpec(
        family="gpt2",
        layers_path="transformer.h",
        final_norm_path="transformer.ln_f",
        unembed_path="lm_head",
        embed_path="transformer.wte",
        norm_type="layernorm",
    ),
    "gpt_neox": ModelSpec(
        family="gpt_neox",
        layers_path="gpt_neox.layers",
        final_norm_path="gpt_neox.final_layer_norm",
        unembed_path="embed_out",
        embed_path="gpt_neox.embed_in",
        norm_type="layernorm",
    ),
}


def spec_for(model) -> ModelSpec | None:
    """Return the registered spec for a loaded model, or None to fall back to autodetect."""
    cfg = getattr(model, "config", None)
    model_type = getattr(cfg, "model_type", None) if cfg is not None else None
    return REGISTRY.get(model_type)
