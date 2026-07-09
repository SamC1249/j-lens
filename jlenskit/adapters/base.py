"""Adapter: the single interface the Jacobian math uses to talk to a model.

The estimator and lens never touch HuggingFace internals directly. They only:
  - ``capture`` residual streams at every layer (as graph nodes, grad retained),
  - turn a residual into logits via the model's *real* final-norm + unembedding,
  - ``patch`` a residual mid-forward for causal interventions.

Everything model-specific is confined to the ModelSpec paths, so a new architecture
is a config entry, not new code.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn

from .spec import ModelSpec, _get_by_path, autodetect


@dataclass
class Capture:
    """Residual streams from one forward pass.

    ``residuals[i]`` is the residual stream entering layer ``i`` for ``i in 0..n_layers-1``;
    ``residuals[n_layers]`` is the final pre-norm residual (output of the last layer).
    Each is shape ``[batch, seq, d_model]``. ``model_logits`` is the model's own output.
    """

    residuals: dict[int, torch.Tensor]
    final_residual: torch.Tensor
    model_logits: torch.Tensor
    input_ids: torch.Tensor


def _extract_hidden(x):
    """Decoder layers return a Tensor or a tuple whose first element is hidden_states."""
    if isinstance(x, tuple):
        return x[0]
    return x


class Adapter:
    def __init__(self, model: nn.Module, tokenizer, spec: ModelSpec | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec or autodetect(model)
        self.model.eval()

        self.layers: nn.ModuleList = _get_by_path(model, self.spec.layers_path)
        self.final_norm: nn.Module = _get_by_path(model, self.spec.final_norm_path)
        self.unembed: nn.Module = _get_by_path(model, self.spec.unembed_path)
        self.embed: nn.Module = _get_by_path(model, self.spec.embed_path)

        self.n_layers = len(self.layers)
        cfg = model.config
        self.d_model = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
        self.vocab_size = int(cfg.vocab_size)

    # -- basics ---------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"]
        return ids.to(self.device)

    def decode_tokens(self, token_ids) -> list[str]:
        return [self.tokenizer.decode([int(t)]) for t in token_ids]

    # -- residual capture -----------------------------------------------------
    def capture(self, input_ids: torch.Tensor, requires_grad: bool = True) -> Capture:
        """Run a forward pass, returning the residual stream at every layer.

        When ``requires_grad`` is True the residuals are kept in the autograd graph so
        the estimator can differentiate the final residual with respect to them.
        """
        input_ids = input_ids.to(self.device)
        residuals: dict[int, torch.Tensor] = {}
        handles = []

        def make_pre_hook(layer_idx):
            def pre_hook(module, args, kwargs):
                hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                residuals[layer_idx] = hs
                return None

            return pre_hook

        def final_hook(module, args, kwargs, output):
            residuals[self.n_layers] = _extract_hidden(output)
            return None

        for i, layer in enumerate(self.layers):
            handles.append(layer.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True))
        handles.append(self.layers[-1].register_forward_hook(final_hook, with_kwargs=True))

        grad_ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()
        try:
            with grad_ctx:
                out = self.model(input_ids=input_ids, use_cache=False)
                logits = out.logits
        finally:
            for h in handles:
                h.remove()

        return Capture(
            residuals=residuals,
            final_residual=residuals[self.n_layers],
            model_logits=logits,
            input_ids=input_ids,
        )

    # -- residual -> logits ---------------------------------------------------
    def to_logits(self, h: torch.Tensor, apply_softcap: bool = True) -> torch.Tensor:
        """Map a (pre-final-norm) residual to vocabulary logits with the real modules.

        The vocabulary projection is computed in float32 so bf16/fp16 rounding cannot
        reorder near-tied top-k tokens (the model's own final norm still runs in its
        native dtype, so the read-out stays faithful to what the model computes).

        ``apply_softcap`` defaults to True so lens read-outs match the model's own
        (softcapped) logits on architectures like Gemma 2; pass False for the raw
        pre-softcap logits.
        """
        normed = self.final_norm(h)
        weight = self.unembed.weight
        bias = getattr(self.unembed, "bias", None)
        logits = nn.functional.linear(
            normed.to(torch.float32),
            weight.to(torch.float32),
            bias.to(torch.float32) if bias is not None else None,
        )
        cap = self.spec.quirks.get("final_logit_softcapping")
        if apply_softcap and cap:
            logits = torch.tanh(logits / cap) * cap
        return logits

    # -- interventions --------------------------------------------------------
    @contextlib.contextmanager
    def patch(self, layer: int, fn):
        """Context manager: replace the residual entering ``layer`` with ``fn(residual)``.

        Used by causal interventions (concept swaps/injections): run forward, edit the
        residual at one layer, let the rest of the network react.
        """
        target = self.layers[layer]

        def pre_hook(module, args, kwargs):
            hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            new_hs = fn(hs)
            if len(args) > 0:
                return (new_hs, *args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = new_hs
            return args, kwargs

        handle = target.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()
