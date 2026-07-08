"""Tests for the jlenskit.store module: cache, results/metrics, manifest."""

from __future__ import annotations

import json

import pyarrow.parquet as pq
import torch

from jlenskit.store import (
    LensCache,
    build_manifest,
    save_metrics,
    save_result,
    stable_hash,
    write_manifest,
)


def test_stable_hash_deterministic():
    a = stable_hash({"x": 1, "y": [1, 2]})
    b = stable_hash({"y": [1, 2], "x": 1})  # key order should not matter
    assert a == b
    assert a != stable_hash({"x": 2, "y": [1, 2]})


def test_cache_put_get_roundtrip(tmp_path, toy_lens):
    cache = LensCache(root=tmp_path)
    key = cache.key("toy/model", "main", {"corpus": "a"}, {"chunk_size": 8})

    assert cache.get(key) is None  # cold cache

    path = cache.put(key, toy_lens)
    assert path.exists()
    assert path == cache.path_for(key)

    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.layers == toy_lens.layers
    for layer in toy_lens.layers:
        assert torch.allclose(loaded.jacobians[layer], toy_lens.jacobians[layer])

    # get_or_none convenience returns the same lens
    via_convenience = cache.get_or_none(
        "toy/model", "main", {"corpus": "a"}, {"chunk_size": 8}
    )
    assert via_convenience is not None
    assert via_convenience.layers == toy_lens.layers


def test_cache_key_determinism_and_sensitivity(tmp_path):
    cache = LensCache(root=tmp_path)
    k1 = cache.key("m", "r", {"corpus": "a"}, {"c": 1})
    k2 = cache.key("m", "r", {"corpus": "a"}, {"c": 1})
    assert k1 == k2  # deterministic

    k3 = cache.key("m", "r", {"corpus": "b"}, {"c": 1})
    assert k1 != k3  # changes with corpus_spec


def test_save_result_roundtrip(tmp_path, toy_adapter, toy_lens):
    result = toy_lens.apply(
        toy_adapter, "hello world there", positions=[-1], top_k=3
    )
    paths = save_result(result, tmp_path)

    assert paths["parquet"].exists()
    assert paths["json"].exists()

    table = pq.read_table(str(paths["parquet"]))
    assert set(table.column_names) == {
        "position",
        "layer",
        "rank",
        "token_id",
        "token",
        "score",
    }
    assert table.num_rows > 0

    with open(paths["json"]) as f:
        data = json.load(f)
    assert data["prompt"] == "hello world there"
    assert data["positions"] == list(result.positions)
    assert "decoded" in data


def test_save_metrics_roundtrip(tmp_path):
    metrics = {
        "kl_divergence": {0: 0.5, 1: 0.3, 2: 0.1},
        "top1_agreement": {0: 0.4, 1: 0.6, 2: 0.9},
    }
    paths = save_metrics(metrics, tmp_path)

    assert paths["parquet"].exists()
    assert paths["json"].exists()

    table = pq.read_table(str(paths["parquet"]))
    assert set(table.column_names) == {"metric", "layer", "value"}
    assert table.num_rows == 6

    with open(paths["json"]) as f:
        data = json.load(f)
    assert data["kl_divergence"]["0"] == 0.5


def test_manifest_build_and_write(tmp_path, toy_adapter, toy_lens):
    manifest = build_manifest(
        command="jlenskit fit ...",
        adapter=toy_adapter,
        lens=toy_lens,
        config={"chunk_size": 8},
        seed=0,
        outputs={"lens": "path/to/lens.safetensors"},
    )
    path = write_manifest(manifest, tmp_path / "sub" / "manifest.json")
    assert path.exists()

    with open(path) as f:
        data = json.load(f)
    for key in (
        "command",
        "model_id",
        "model_revision",
        "seed",
        "config",
        "lens_meta",
        "environment",
        "outputs",
    ):
        assert key in data

    assert "torch" in data["environment"]
    assert "transformers" in data["environment"]
    assert data["environment"]["torch"]
    assert data["environment"]["transformers"]
