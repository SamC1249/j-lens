"""Knowledge-probe suite: determinate-answer prompts + elicitation-depth metric."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch

from ..core.lens_base import apply_lens

_DIR = Path(__file__).parent / "probes"


@dataclass
class Probe:
    prompt: str
    answers: list[str]
    category: str


def load_probes(name: str = "core_v1") -> list[Probe]:
    path = _DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Probe(prompt=d["prompt"], answers=list(d["answers"]), category=d["category"]) for d in data]


@torch.no_grad()
def elicitation_depth(adapter, lens, probes: list[Probe], top_k: int = 5) -> dict:
    per_probe = []
    by_cat: dict[str, list[int]] = {}
    for p in probes:
        res = apply_lens(lens, adapter, p.prompt, positions=[-1], top_k=top_k)
        wanted = {a.strip().lower() for a in p.answers}
        depth = None
        for dl in res.decoded[0]:  # ordered by layer
            toks = {t.strip().lower() for t in dl.tokens}
            if toks & wanted:          # exact-match intersection
                depth = dl.layer
                break
        per_probe.append({"prompt": p.prompt, "category": p.category, "depth": depth})
        if depth is not None:
            by_cat.setdefault(p.category, []).append(depth)
    median_by_category = {c: (median(v) if v else None) for c, v in by_cat.items()}
    return {"per_probe": per_probe, "median_by_category": median_by_category}
