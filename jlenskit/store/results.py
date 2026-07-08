"""Serialize lens read-outs and metrics to Parquet + JSON on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..core.types import LensResult


def save_result(result: LensResult, out_dir: str | Path) -> dict:
    """Write a lens read-out to ``result.parquet`` and ``result.json``.

    The Parquet table is fully flattened: one row per (position, layer, rank),
    with columns ``position, layer, rank, token_id, token, score``. The JSON view
    holds the prompt, positions, and the decoded read-out in nested form.
    Returns a dict of the written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positions: list[int] = []
    layers: list[int] = []
    ranks: list[int] = []
    token_ids: list[int] = []
    tokens: list[str] = []
    scores: list[float] = []

    decoded_json: dict[int, list[dict]] = {}
    for pos_index, per_layer in result.decoded.items():
        pos_value = result.positions[pos_index]
        decoded_json[pos_index] = []
        for dl in per_layer:
            decoded_json[pos_index].append(
                {
                    "layer": dl.layer,
                    "token_ids": dl.token_ids,
                    "tokens": dl.tokens,
                    "scores": dl.scores,
                }
            )
            for rank, (tid, tok, sc) in enumerate(
                zip(dl.token_ids, dl.tokens, dl.scores)
            ):
                positions.append(pos_value)
                layers.append(dl.layer)
                ranks.append(rank)
                token_ids.append(int(tid))
                tokens.append(str(tok))
                scores.append(float(sc))

    table = pa.table(
        {
            "position": pa.array(positions, type=pa.int64()),
            "layer": pa.array(layers, type=pa.int64()),
            "rank": pa.array(ranks, type=pa.int64()),
            "token_id": pa.array(token_ids, type=pa.int64()),
            "token": pa.array(tokens, type=pa.string()),
            "score": pa.array(scores, type=pa.float64()),
        }
    )
    parquet_path = out_dir / "result.parquet"
    pq.write_table(table, str(parquet_path))

    json_path = out_dir / "result.json"
    json_view = {
        "prompt": result.prompt,
        "positions": list(result.positions),
        "decoded": decoded_json,
    }
    json_path.write_text(json.dumps(json_view, indent=2, default=str))

    return {"parquet": parquet_path, "json": json_path}


def save_metrics(metrics: dict[str, dict[int, float]], out_dir: str | Path) -> dict:
    """Write per-layer metrics to ``metrics.parquet`` and ``metrics.json``.

    ``metrics`` maps ``metric_name -> {layer: value}``. Parquet columns are
    ``metric, layer, value``. Returns a dict of the written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    layers: list[int] = []
    values: list[float] = []
    for name, per_layer in metrics.items():
        for layer, value in per_layer.items():
            names.append(str(name))
            layers.append(int(layer))
            values.append(float(value))

    table = pa.table(
        {
            "metric": pa.array(names, type=pa.string()),
            "layer": pa.array(layers, type=pa.int64()),
            "value": pa.array(values, type=pa.float64()),
        }
    )
    parquet_path = out_dir / "metrics.parquet"
    pq.write_table(table, str(parquet_path))

    json_path = out_dir / "metrics.json"
    json_view = {
        name: {str(layer): float(value) for layer, value in per_layer.items()}
        for name, per_layer in metrics.items()
    }
    json_path.write_text(json.dumps(json_view, indent=2, default=str))

    return {"parquet": parquet_path, "json": json_path}
