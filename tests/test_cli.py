"""End-to-end CLI test, offline: monkeypatch model loading to the toy adapter.

Exercises the full config -> fit -> apply/eval/viz/intervene pipeline including manifest
and result writing, without any downloads.
"""

from __future__ import annotations

import json

import yaml

from jlenskit import cli


def _write_cfg(tmp_path, extra=None):
    cfg = {
        "model": {"id": "dummy", "dtype": "float32", "device": "cpu"},
        "corpus": {"source": "builtin", "seq_len": 8, "n_sequences": 4, "seed": 0},
        "fit": {"layers": [0, 1, 2], "chunk_size": 8},
        "lens": {"use_cache": False},
        "apply": {"prompts": ["hello world there", "the sky is"], "positions": [-1],
                  "top_k": 5, "viz": True, "viz_kind": "grid"},
        "eval": {"top_k": 3},
        "seed": 0,
        "output": {"dir": str(tmp_path / "run")},
    }
    if extra:
        cfg.update(extra)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_cli_fit_apply_eval(tmp_path, monkeypatch, toy_adapter):
    monkeypatch.setattr(cli, "load", lambda *a, **k: toy_adapter)
    cfg_path = _write_cfg(tmp_path)

    cli.fit(str(cfg_path))
    run = tmp_path / "run"
    assert (run / "lens.safetensors").exists()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["command"] == "fit"
    assert "torch" in manifest["environment"]

    cli.apply(str(cfg_path))
    assert (run / "prompt_000" / "result.json").exists()
    assert (run / "prompt_000" / "viz.html").exists()

    cli.eval(str(cfg_path))
    assert (run / "metrics.json").exists() or (run / "metrics.parquet").exists()


def test_cli_intervene(tmp_path, monkeypatch, toy_adapter):
    monkeypatch.setattr(cli, "load", lambda *a, **k: toy_adapter)
    extra = {"intervene": {"type": "inject", "prompt": "hello world", "layer": 1,
                           "token": 5, "strength": 8.0, "top_k": 5}}
    cfg_path = _write_cfg(tmp_path, extra)
    cli.intervene(str(cfg_path))
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["command"] == "intervene"
