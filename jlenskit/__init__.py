"""jlenskit — a reproducible, modular toolkit for the Jacobian lens (J-lens).

Reference: "Verbalizable Representations Form a Global Workspace in Language Models"
(Anthropic, Transformer Circuits, 2026). https://transformer-circuits.pub/2026/workspace/
"""

from __future__ import annotations

from .adapters import Adapter, ModelSpec, autodetect
from .baselines import logit_lens
from .core import JacobianLens, LensResult
from .models import adapter_from, load

__version__ = "0.1.0"

__all__ = [
    "load",
    "adapter_from",
    "Adapter",
    "ModelSpec",
    "autodetect",
    "JacobianLens",
    "LensResult",
    "logit_lens",
    "__version__",
]
