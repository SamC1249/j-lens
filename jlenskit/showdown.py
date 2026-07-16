"""Showdown harness: compare logit / tuned / Jacobian lenses on coherence + elicitation."""

from __future__ import annotations

import json
from pathlib import Path

from .baselines import LogitLens
from .data import load_probes
from .data.probes import elicitation_depth
from .metrics import entropy, forward_kl, topk_accuracy


def layers_to_coherence(kl: dict[int, float], tau: float) -> int | None:
    if not kl:
        return None
    layers = sorted(kl)
    threshold = tau * kl[layers[0]]
    for i, l in enumerate(layers):
        if all(kl[m] <= threshold for m in layers[i:]):
            return l
    return None


def run_showdown(cfg, adapter, jlens, batches, tuned=None) -> dict:
    batches = list(batches)
    lens_objs = {}
    if "logit" in cfg.lenses:
        lens_objs["logit"] = LogitLens(jlens.layers)
    if "tuned" in cfg.lenses and tuned is not None:
        lens_objs["tuned"] = tuned
    if "jacobian" in cfg.lenses:
        lens_objs["jacobian"] = jlens

    out = {"lenses": {}, "elicitation": None}
    for name, lens in lens_objs.items():
        kl = forward_kl(adapter, lens, batches)
        out["lenses"][name] = {
            "forward_kl": kl,
            "entropy": entropy(adapter, lens, batches),
            "topk_accuracy": topk_accuracy(adapter, lens, batches, k=cfg.top_k),
            "layers_to_coherence": layers_to_coherence(kl, cfg.coherence_tau),
        }

    if cfg.probes:
        probes = load_probes(cfg.probes)
        out["elicitation"] = {
            name: elicitation_depth(adapter, lens, probes, top_k=cfg.top_k)
            for name, lens in lens_objs.items()
        }
    return out


def _mean_early(kl: dict[int, float], upto: int = 15) -> float:
    vals = [v for l, v in kl.items() if l <= upto]
    return sum(vals) / len(vals) if vals else float("nan")


def write_showdown_outputs(results: dict, out_dir) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "showdown_metrics.json"
    jpath.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = ["# Lens showdown", "", "| lens | layers-to-coherence | mean fwd-KL (L0-15) | final top-k acc |",
             "|------|--------------------:|--------------------:|----------------:|"]
    for name, m in results["lenses"].items():
        kl = m["forward_kl"]
        last = max(kl) if kl else None
        acc = m["topk_accuracy"].get(last)
        l_star = m["layers_to_coherence"]
        lines.append(f"| {name} | {l_star if l_star is not None else 'n/a'} | "
                     f"{_mean_early(kl):.3f} | {acc:.3f} |")
    if results.get("elicitation"):
        lines += ["", "## Median elicitation depth by category", "",
                  "| lens | " + " | ".join(sorted({c for e in results['elicitation'].values()
                                                    for c in e['median_by_category']})) + " |"]
        cats = sorted({c for e in results["elicitation"].values() for c in e["median_by_category"]})
        lines.append("|------|" + "|".join(["---:"] * len(cats)) + "|")
        for name, e in results["elicitation"].items():
            cells = [str(e["median_by_category"].get(c, "n/a")) for c in cats]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    mdpath = out_dir / "showdown.md"
    mdpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"showdown_metrics": str(jpath), "showdown_md": str(mdpath)}
