"""Corpus loading for lens fitting.

Fitting averages the Jacobian over a corpus of fixed-length, unpadded token sequences.
For reproducibility every corpus is described by a spec dict that is recorded in the run
manifest. Three sources are supported:

  - ``builtin``: a small bundled English corpus (offline, deterministic; good for demos/tests)
  - ``file``:    a local UTF-8 text file, one document per line
  - ``hf``:      a HuggingFace dataset slice (requires the optional ``datasets`` package)

``random`` sequences are provided for tests.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import torch


def pack_texts(texts: list[str], tokenizer, seq_len: int, n_sequences: int) -> list[torch.Tensor]:
    """Tokenize documents and pack them into fixed-length, unpadded sequences.

    Packing (rather than padding) keeps the "no padding" assumption the estimator relies on
    for its (t, t'>=t) pair counting.
    """
    buffer: list[int] = []
    seqs: list[torch.Tensor] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        while len(buffer) >= seq_len:
            seqs.append(torch.tensor(buffer[:seq_len]).unsqueeze(0))
            buffer = buffer[seq_len:]
            if len(seqs) >= n_sequences:
                return seqs
    return seqs


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_corpus(spec: dict, tokenizer) -> list[torch.Tensor]:
    """Build a list of ``input_ids`` tensors from a corpus spec.

    Common keys: ``source``, ``seq_len`` (default 64), ``n_sequences`` (default 128),
    ``seed``. Source-specific: ``path`` (file); ``name``/``revision``/``split``/
    ``text_column`` (hf).
    """
    source = spec.get("source", "builtin")
    seq_len = int(spec.get("seq_len", 64))
    n_sequences = int(spec.get("n_sequences", 128))

    if source == "builtin":
        text = files("jlenskit.data").joinpath("sample_corpus.txt").read_text(encoding="utf-8")
        texts = [ln for ln in text.splitlines() if ln.strip()]
        # repeat to reach the requested count if necessary (deterministic)
        return pack_texts(texts * (n_sequences + 1), tokenizer, seq_len, n_sequences)

    if source == "file":
        texts = _read_lines(Path(spec["path"]))
        return pack_texts(texts, tokenizer, seq_len, n_sequences)

    if source == "hf":
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "source='hf' requires the optional 'datasets' package: pip install 'jlenskit[hf]'"
            ) from e
        ds = load_dataset(
            spec["name"], revision=spec.get("revision"), split=spec.get("split", "train"),
            streaming=True,
        )
        col = spec.get("text_column", "text")
        texts = []
        for i, row in enumerate(ds):
            if i >= n_sequences * 4:  # gather enough to pack
                break
            texts.append(row[col])
        return pack_texts(texts, tokenizer, seq_len, n_sequences)

    raise ValueError(f"Unknown corpus source: {source!r}")


def random_corpus(vocab_size: int, seq_len: int, n_sequences: int, seed: int = 0) -> list[torch.Tensor]:
    """Random token sequences for tests (no tokenizer/model download needed)."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, vocab_size, (1, seq_len), generator=g) for _ in range(n_sequences)]
