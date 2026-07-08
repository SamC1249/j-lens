"""LensCache: content-addressed local cache of fitted lenses, with optional HF hub sync."""

from __future__ import annotations

import os
from pathlib import Path

from ..core.lens import JacobianLens
from .hashing import stable_hash


class LensCache:
    """A content-addressed cache of fitted lenses on the local filesystem.

    The cache key is a stable hash of (model_id, revision, corpus_spec, fit_config),
    so an identical fitting request always maps to the same file and can be reused
    instead of re-fit. Files are stored as ``{key}.safetensors``.
    """

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = os.environ.get("JLENSKIT_CACHE") or (
                Path.home() / ".cache" / "jlenskit" / "lenses"
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- keying ---------------------------------------------------------------
    def key(
        self,
        model_id: str,
        revision: str | None,
        corpus_spec: dict,
        fit_config: dict,
    ) -> str:
        """Stable cache key (filename stem) for a fitting request."""
        return stable_hash(
            {
                "model_id": model_id,
                "revision": revision,
                "corpus_spec": corpus_spec,
                "fit_config": fit_config,
            }
        )

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.safetensors"

    # -- get / put ------------------------------------------------------------
    def get(self, key: str) -> JacobianLens | None:
        """Load a cached lens by key, or ``None`` if it is not cached."""
        path = self.path_for(key)
        if not path.exists():
            return None
        return JacobianLens.load(path)

    def put(self, key: str, lens: JacobianLens) -> Path:
        """Persist ``lens`` under ``key``; returns the file path."""
        return lens.save(self.path_for(key))

    def get_or_none(
        self,
        model_id: str,
        revision: str | None,
        corpus_spec: dict,
        fit_config: dict,
    ) -> JacobianLens | None:
        """Convenience: compute the key and return the cached lens, or ``None``."""
        return self.get(self.key(model_id, revision, corpus_spec, fit_config))

    # -- HF hub (optional) ----------------------------------------------------
    @staticmethod
    def _hf_token() -> str | None:
        return os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")

    def push_to_hub(self, key: str, repo_id: str) -> str:
        """Upload the cached lens file for ``key`` to a HuggingFace hub repo."""
        from huggingface_hub import HfApi

        path = self.path_for(key)
        if not path.exists():
            raise FileNotFoundError(f"No cached lens for key {key!r} at {path}")
        api = HfApi(token=self._hf_token())
        return api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            token=self._hf_token(),
        )

    def pull_from_hub(self, repo_id: str, filename: str, key: str | None = None) -> Path:
        """Download ``filename`` from ``repo_id`` into the cache (as ``{key}.safetensors``)."""
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id, filename=filename, token=self._hf_token()
        )
        stem = key if key is not None else Path(filename).stem
        dest = self.path_for(stem)
        dest.write_bytes(Path(downloaded).read_bytes())
        return dest
