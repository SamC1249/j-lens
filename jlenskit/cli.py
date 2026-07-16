"""jlenskit command-line interface.

    jlenskit fit CONFIG        # fit (or load-from-cache) a lens
    jlenskit apply CONFIG      # apply the lens to prompts -> results (+ optional viz)
    jlenskit eval CONFIG       # lens-quality metrics over a corpus
    jlenskit viz CONFIG        # render the layer x position slice viewer
    jlenskit intervene CONFIG  # concept inject/swap experiment

Every command is driven by one YAML config and writes a reproducibility manifest to the
run's output directory.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import jspace
from .baselines import TunedLens
from .config import Config, ShowdownCfg, load_config
from .core.lens import JacobianLens
from .data import load_corpus
from .metrics import evaluate, metrics_to_rows
from .models import load
from .showdown import run_showdown, write_showdown_outputs
from .store import (
    LensCache,
    build_manifest,
    save_metrics,
    save_result,
    write_manifest,
)
from .viz import render, render_showdown

app = typer.Typer(add_completion=False, help="Run the Jacobian lens (J-lens) on models.")


def _adapter(cfg: Config):
    m = cfg.model
    typer.echo(f"[jlenskit] loading model {m.id} ...")
    return load(m.id, revision=m.revision, dtype=m.dtype, device=m.device, trust_remote_code=m.trust_remote_code)


def _resolve_lens(cfg: Config, adapter) -> JacobianLens:
    """Load an explicit lens, hit the cache, or fit a fresh one."""
    if cfg.lens.path:
        typer.echo(f"[jlenskit] loading lens from {cfg.lens.path}")
        return JacobianLens.load(cfg.lens.path)

    cache = LensCache()
    key = cache.key(cfg.model.id, cfg.model.revision, cfg.corpus.to_spec(), cfg.fit.model_dump())
    if cfg.lens.use_cache:
        cached = cache.get(key)
        if cached is not None:
            typer.echo(f"[jlenskit] using cached lens {key}")
            return cached

    typer.echo("[jlenskit] fitting lens (building corpus) ...")
    batches = load_corpus(cfg.corpus.to_spec(), adapter.tokenizer)
    typer.echo(f"[jlenskit] corpus: {len(batches)} sequences of length {cfg.corpus.seq_len}")
    lens = JacobianLens.fit(
        adapter,
        batches,
        layers=cfg.fit.layers,
        chunk_size=cfg.fit.chunk_size,
        seed=cfg.seed,
        corpus_spec=cfg.corpus.to_spec(),
        fit_config=cfg.fit.model_dump(),
        progress=True,
    )
    if cfg.lens.use_cache:
        cache.put(key, lens)
        typer.echo(f"[jlenskit] cached lens {key}")
    return lens


def _finish(cfg: Config, adapter, lens, command: str, outputs: dict):
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lens_path = out_dir / "lens.safetensors"
    if not lens_path.exists():
        lens.save(lens_path)
        outputs["lens"] = str(lens_path)
    manifest = build_manifest(command, adapter, lens, cfg.model_dump(), cfg.seed, outputs=outputs)
    mpath = write_manifest(manifest, out_dir / "manifest.json")
    typer.echo(f"[jlenskit] wrote manifest -> {mpath}")


@app.command()
def fit(config: str):
    """Fit (or load-from-cache) a Jacobian lens and persist it with a manifest."""
    cfg = load_config(config)
    adapter = _adapter(cfg)
    lens = _resolve_lens(cfg, adapter)
    typer.echo(f"[jlenskit] lens ready: layers {lens.layers} (d_model={lens.meta.d_model})")
    _finish(cfg, adapter, lens, "fit", {})


@app.command()
def apply(config: str):
    """Apply the lens to the configured prompts and save decoded results (+ optional viz)."""
    cfg = load_config(config)
    adapter = _adapter(cfg)
    lens = _resolve_lens(cfg, adapter)
    out_dir = Path(cfg.output.dir)
    outputs: dict = {}
    for i, prompt in enumerate(cfg.apply.prompts):
        res = lens.apply(adapter, prompt, positions=cfg.apply.positions, layers=cfg.apply.layers, top_k=cfg.apply.top_k)
        sub = out_dir / f"prompt_{i:03d}"
        outputs[f"prompt_{i}"] = save_result(res, sub)
        if cfg.apply.viz:
            outputs[f"viz_{i}"] = str(render(res, sub / "viz.html", kind=cfg.apply.viz_kind))
        top = res.decoded[0][-1]
        typer.echo(f"[jlenskit] '{prompt[:40]}' final-layer top: {top.tokens[:5]}")
    _finish(cfg, adapter, lens, "apply", outputs)


@app.command()
def eval(config: str):
    """Compute lens-quality metrics over a corpus and save them."""
    cfg = load_config(config)
    adapter = _adapter(cfg)
    lens = _resolve_lens(cfg, adapter)
    batches = load_corpus(cfg.corpus.to_spec(), adapter.tokenizer)
    typer.echo(f"[jlenskit] evaluating over {len(batches)} sequences ...")
    metrics = evaluate(adapter, lens, batches, k=cfg.eval.top_k)
    out_dir = Path(cfg.output.dir)
    paths = save_metrics(metrics, out_dir)
    for name, per_layer in metrics.items():
        best = max(per_layer.items(), key=lambda kv: kv[1])
        typer.echo(f"[jlenskit] {name}: peak layer {best[0]} = {best[1]:.4f}")
    _finish(cfg, adapter, lens, "eval", {"metrics": paths, "rows": len(metrics_to_rows(metrics))})


@app.command()
def viz(config: str):
    """Render the layer x position slice viewer for the first configured prompt."""
    cfg = load_config(config)
    adapter = _adapter(cfg)
    lens = _resolve_lens(cfg, adapter)
    if not cfg.apply.prompts:
        raise typer.BadParameter("viz needs at least one prompt under apply.prompts")
    res = lens.apply(adapter, cfg.apply.prompts[0], positions=cfg.apply.positions, layers=cfg.apply.layers, top_k=cfg.apply.top_k)
    out_dir = Path(cfg.output.dir)
    path = render(res, out_dir / "viz.html", kind=cfg.apply.viz_kind)
    typer.echo(f"[jlenskit] wrote viz -> {path}")
    _finish(cfg, adapter, lens, "viz", {"viz": str(path)})


@app.command()
def intervene(config: str):
    """Run a concept inject/swap experiment and print the effect on the next token."""
    cfg = load_config(config)
    if cfg.intervene is None:
        raise typer.BadParameter("provide an 'intervene:' section in the config")
    ic = cfg.intervene
    adapter = _adapter(cfg)
    lens = _resolve_lens(cfg, adapter)
    if ic.type == "inject":
        res = jspace.inject(adapter, lens, ic.prompt, ic.layer, ic.token, strength=ic.strength, position=ic.position, top_k=ic.top_k)
    elif ic.type == "swap":
        res = jspace.swap(adapter, lens, ic.prompt, ic.layer, ic.source, ic.target, strength=ic.strength, position=ic.position, top_k=ic.top_k)
    else:
        raise typer.BadParameter(f"unknown intervene.type {ic.type!r}")
    typer.echo(f"[jlenskit] {res.description}")
    typer.echo(f"  baseline : {res.baseline_top}")
    typer.echo(f"  intervened: {res.intervened_top}")
    _finish(cfg, adapter, lens, "intervene", {"description": res.description,
                                              "baseline_top": res.baseline_top,
                                              "intervened_top": res.intervened_top})


@app.command()
def showdown(config: str):
    """Compare logit / tuned / Jacobian lenses on coherence + knowledge elicitation."""
    cfg = load_config(config)
    sc = cfg.showdown or ShowdownCfg()
    adapter = _adapter(cfg)
    jlens = _resolve_lens(cfg, adapter)
    batches = load_corpus(cfg.corpus.to_spec(), adapter.tokenizer)

    # Resolve tuned lens via file-based caching in the output dir.
    # LensCache.get() hardcodes JacobianLens.load(), so it cannot round-trip a
    # TunedLens checkpoint.  We therefore skip LensCache for TunedLens and use a
    # plain safetensors file (<output_dir>/tuned_lens.safetensors) instead:
    # load from file if present, otherwise train and save.
    tuned = None
    if "tuned" in sc.lenses:
        out_dir = Path(cfg.output.dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tuned_path = out_dir / "tuned_lens.safetensors"
        if tuned_path.exists():
            typer.echo(f"[jlenskit] loading cached tuned lens from {tuned_path}")
            tuned = TunedLens.load(tuned_path)
        else:
            typer.echo(f"[jlenskit] training tuned lens ({sc.tuned_steps} steps) ...")
            tuned = TunedLens.fit(
                adapter,
                batches,
                layers=jlens.layers,
                n_steps=sc.tuned_steps,
                lr=sc.tuned_lr,
                seed=cfg.seed,
                corpus_spec=cfg.corpus.to_spec(),
                progress=True,
            )
            tuned.save(tuned_path)

    results = run_showdown(sc, adapter, jlens, batches, tuned=tuned)
    out_dir = Path(cfg.output.dir)
    outputs = write_showdown_outputs(results, out_dir)
    outputs["showdown_html"] = str(render_showdown(results, out_dir / "showdown.html"))
    for name, m in results["lenses"].items():
        typer.echo(f"[jlenskit] {name}: layers-to-coherence = {m['layers_to_coherence']}")
    _finish(cfg, adapter, jlens, "showdown", outputs)


if __name__ == "__main__":
    app()
