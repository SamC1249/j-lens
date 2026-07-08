"""Loading helpers: turn a HuggingFace model id into a ready-to-use Adapter."""

from __future__ import annotations

import os

import torch

from .adapters.base import Adapter
from .adapters.registry import spec_for
from .adapters.spec import autodetect


def load(
    model_id: str,
    revision: str | None = None,
    dtype: str | torch.dtype = "auto",
    device: str | None = None,
    trust_remote_code: bool = False,
) -> Adapter:
    """Load an HF causal-LM + tokenizer and wrap them in an Adapter.

    The ModelSpec is taken from the registry when the family is known, otherwise
    autodetected. ``dtype='auto'`` uses the checkpoint's dtype.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
    _dtype_map = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    elif dtype == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = _dtype_map.get(dtype, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, token=token, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch_dtype,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    spec = spec_for(model) or autodetect(model)
    adapter = Adapter(model, tokenizer, spec=spec)
    if revision is not None:
        adapter.spec.quirks.setdefault("_revision", revision)
    return adapter


def adapter_from(model, tokenizer, spec=None) -> Adapter:
    """Wrap an already-loaded model+tokenizer (e.g. in a notebook) in an Adapter."""
    spec = spec or spec_for(model) or autodetect(model)
    return Adapter(model, tokenizer, spec=spec)
