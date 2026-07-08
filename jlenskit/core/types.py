"""Typed data structures passed between jlenskit modules.

Keeping these in one place means every module communicates through explicit,
inspectable contracts rather than ad-hoc tuples/dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DecodedLayer:
    """The lens read-out at a single layer for a single position."""

    layer: int
    token_ids: list[int]
    tokens: list[str]
    scores: list[float]  # softmax probabilities aligned with token_ids/tokens


@dataclass
class LensResult:
    """Result of applying a lens to one prompt at one or more positions.

    ``lens_logits`` / ``model_logits`` are keyed by layer index. ``decoded`` is a
    convenience view: position -> list[DecodedLayer].
    """

    prompt: str
    positions: list[int]
    lens_logits: dict[int, torch.Tensor]  # layer -> [n_positions, vocab]
    model_logits: torch.Tensor  # [n_positions, vocab]
    decoded: dict[int, list[DecodedLayer]] = field(default_factory=dict)

    def top_tokens(self, layer: int, position_index: int = 0, k: int = 5) -> list[str]:
        probs = torch.softmax(self.lens_logits[layer][position_index], dim=-1)
        idx = probs.topk(k).indices.tolist()
        return idx


@dataclass
class LensMeta:
    """Provenance for a fitted lens (goes into the manifest and lens checkpoint)."""

    model_id: str
    model_revision: str | None
    d_model: int
    vocab_size: int
    n_layers: int
    layers_fit: list[int]
    n_prompts: int
    n_positions: int
    corpus_spec: dict[str, Any] = field(default_factory=dict)
    fit_config: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    jlenskit_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunManifest:
    """Everything needed to reproduce a run. Serialized to manifest.json."""

    command: str
    model_id: str
    model_revision: str | None
    seed: int | None
    config: dict[str, Any]
    lens_meta: dict[str, Any]
    environment: dict[str, Any]  # python/torch/transformers/jlenskit versions, device
    outputs: dict[str, Any] = field(default_factory=dict)
