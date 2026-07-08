"""Stable hashing of JSON-serializable objects for cache keys and manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(obj: Any) -> str:
    """Deterministic short hash of a JSON-serializable object.

    Serializes with sorted keys (so dict ordering never affects the result) and
    ``default=str`` (so non-JSON values degrade gracefully), then returns the
    first 16 hex chars of the SHA-256 digest.
    """
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
