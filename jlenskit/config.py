"""Typed, validated run configuration (YAML-driven, for reproducibility).

Every CLI command reads the relevant sections of one config file. The exact config is
recorded in each run's manifest so a run can be reproduced from its output directory.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelCfg(BaseModel):
    id: str
    revision: str | None = None
    dtype: str = "auto"
    device: str | None = None
    trust_remote_code: bool = False


class CorpusCfg(BaseModel):
    source: str = "builtin"  # builtin | file | hf
    seq_len: int = 64
    n_sequences: int = 128
    seed: int = 0
    path: str | None = None
    name: str | None = None
    split: str = "train"
    text_column: str = "text"
    revision: str | None = None

    def to_spec(self) -> dict:
        return self.model_dump()


class FitCfg(BaseModel):
    layers: list[int] | None = None
    chunk_size: int = 32


class LensCfg(BaseModel):
    path: str | None = None  # explicit lens .safetensors to load instead of fitting
    use_cache: bool = True


class ApplyCfg(BaseModel):
    prompts: list[str] = Field(default_factory=list)
    positions: list[int] = Field(default_factory=lambda: [-1])
    top_k: int = 10
    layers: list[int] | None = None
    viz: bool = True
    viz_kind: str = "grid"


class EvalCfg(BaseModel):
    top_k: int = 5
    # eval uses its own corpus draw; reuse CorpusCfg fields via the top-level corpus by default


class InterveneCfg(BaseModel):
    type: str = "inject"  # inject | swap
    prompt: str = ""
    layer: int = 0
    position: int = -1
    token: str | int | None = None  # for inject
    source: str | int | None = None  # for swap
    target: str | int | None = None  # for swap
    strength: float = 6.0
    top_k: int = 5


class OutputCfg(BaseModel):
    dir: str = "runs/run"


class Config(BaseModel):
    model: ModelCfg
    corpus: CorpusCfg = Field(default_factory=CorpusCfg)
    fit: FitCfg = Field(default_factory=FitCfg)
    lens: LensCfg = Field(default_factory=LensCfg)
    apply: ApplyCfg = Field(default_factory=ApplyCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    intervene: InterveneCfg | None = None
    seed: int | None = 0
    output: OutputCfg = Field(default_factory=OutputCfg)


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config(**data)
