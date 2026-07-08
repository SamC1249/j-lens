"""Reproducibility manifest: capture environment + provenance for a run."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

from ..adapters.base import Adapter
from ..core.lens import JacobianLens
from ..core.types import RunManifest


def capture_environment() -> dict:
    """Snapshot the runtime environment relevant to reproducing a run."""
    import transformers

    import jlenskit

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "jlenskit": jlenskit.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def build_manifest(
    command: str,
    adapter: Adapter,
    lens: JacobianLens,
    config: dict,
    seed: int | None,
    outputs: dict | None = None,
) -> RunManifest:
    """Assemble a :class:`RunManifest` from an adapter, a fitted lens, and run config."""
    model_id = getattr(adapter.model.config, "_name_or_path", "unknown")
    return RunManifest(
        command=command,
        model_id=model_id,
        model_revision=lens.meta.model_revision,
        seed=seed,
        config=config,
        lens_meta=dict(lens.meta.__dict__),
        environment=capture_environment(),
        outputs=outputs or {},
    )


def write_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Serialize a manifest to ``path`` as indented JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.__dict__, indent=2, default=str))
    return path
