# Lens Showdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `jlenskit showdown` capability that compares the logit lens, a new in-package tuned lens, and the J-lens on coherence + knowledge-elicitation metrics across model families.

**Architecture:** All three lenses become the same operation — `to_logits(transport(layer, h))` — behind a common `Lens` protocol. The tuned lens is a per-layer affine translator trained by distilling to the model's final output. New coherence metrics (`forward_kl`, `entropy`) and a knowledge-probe suite drive a `showdown` harness that emits a comparison table, JSON, and an HTML plot.

**Tech Stack:** Python 3.12 (repo venv `.venv`), PyTorch, transformers, safetensors, pydantic, typer, pytest, ruff.

## Global Constraints

- Run everything with the project venv: `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`.
- All new tests must run on the toy Llama fixture (`tests/conftest.py`): no downloads, no GPU.
- New lenses must produce the existing `LensResult` shape (`jlenskit/core/types.py`) so viz/metrics work unchanged.
- Preserve public API: `from jlenskit import logit_lens` and `JacobianLens.apply(...)` keep their current signatures and behavior.
- `jlenskit` version stays `0.1.0` unless the cleanup task bumps it.
- Do NOT `git push`. Commits are local; the human authorizes any push.

---

### Task 1: `Lens` protocol + `transport` + shared apply/decode

**Files:**
- Create: `jlenskit/core/lens_base.py`
- Modify: `jlenskit/core/lens.py` (add `name`, `transport`; route `apply`/`_decode` through the shared helpers)
- Modify: `jlenskit/core/__init__.py` (export `Lens`, `apply_lens`)
- Test: `tests/test_lens_base.py`

**Interfaces:**
- Produces:
  - `class Lens(Protocol)` with attrs `name: str`, `layers: list[int]`, method `transport(self, layer: int, h: torch.Tensor) -> torch.Tensor` (h shape `[..., D]` → `[..., D]`).
  - `apply_lens(lens: Lens, adapter: Adapter, prompt: str, positions: list[int] | None = None, layers: list[int] | None = None, top_k: int = 10, decode: bool = True) -> LensResult`
  - `decode_result(adapter: Adapter, result: LensResult, top_k: int) -> dict[int, list[DecodedLayer]]`
  - `JacobianLens.transport(self, layer, h)` and `JacobianLens.name = "jacobian"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lens_base.py
import torch
from jlenskit.core.lens_base import apply_lens


def test_jacobian_transport_matches_manual(toy_lens):
    h = torch.randn(toy_lens.meta.d_model)
    manual = h @ toy_lens.jacobians[1].t()
    assert torch.allclose(toy_lens.transport(1, h), manual, atol=1e-6)
    assert toy_lens.name == "jacobian"


def test_apply_lens_matches_jacobianlens_apply(toy_adapter, toy_lens):
    a = toy_lens.apply(toy_adapter, "one two three", positions=[-1], top_k=3)
    b = apply_lens(toy_lens, toy_adapter, "one two three", positions=[-1], top_k=3)
    for l in a.lens_logits:
        assert torch.allclose(a.lens_logits[l], b.lens_logits[l], atol=1e-6)
    assert a.decoded[0][-1].token_ids == b.decoded[0][-1].token_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_lens_base.py -v`
Expected: FAIL (`cannot import name 'apply_lens'`).

- [ ] **Step 3: Write the shared module**

```python
# jlenskit/core/lens_base.py
"""Common lens interface: every lens is transport -> to_logits -> decode.

logit lens (identity transport), tuned lens (learned affine), and the Jacobian
lens all implement ``transport`` and reuse the apply/decode plumbing here so the
decode loop and LensResult shape live in exactly one place.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from ..adapters.base import Adapter
from .types import DecodedLayer, LensResult


@runtime_checkable
class Lens(Protocol):
    name: str
    layers: list[int]

    def transport(self, layer: int, h: torch.Tensor) -> torch.Tensor: ...


@torch.no_grad()
def apply_lens(
    lens: Lens,
    adapter: Adapter,
    prompt: str,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    top_k: int = 10,
    decode: bool = True,
) -> LensResult:
    input_ids = adapter.encode(prompt)
    seq = input_ids.shape[1]
    if positions is None:
        positions = [-1]
    pos = [p % seq for p in positions]
    layers = layers or list(lens.layers)

    cap = adapter.capture(input_ids, requires_grad=False)
    lens_logits: dict[int, torch.Tensor] = {}
    for l in layers:
        h = cap.residuals[l][0, pos].to(torch.float32)  # [n_pos, D]
        transported = lens.transport(l, h).to(adapter.dtype)
        lens_logits[l] = adapter.to_logits(transported).to(torch.float32)

    result = LensResult(
        prompt=prompt,
        positions=pos,
        lens_logits=lens_logits,
        model_logits=cap.model_logits[0, pos].to(torch.float32),
    )
    if decode:
        result.decoded = decode_result(adapter, result, top_k)
    return result


def decode_result(adapter: Adapter, result: LensResult, top_k: int) -> dict[int, list[DecodedLayer]]:
    decoded: dict[int, list[DecodedLayer]] = {}
    for pi, _p in enumerate(result.positions):
        per_layer = []
        for l in sorted(result.lens_logits):
            probs = torch.softmax(result.lens_logits[l][pi], dim=-1)
            top = probs.topk(top_k)
            ids = top.indices.tolist()
            per_layer.append(
                DecodedLayer(layer=l, token_ids=ids, tokens=adapter.decode_tokens(ids), scores=top.values.tolist())
            )
        decoded[pi] = per_layer
    return decoded
```

- [ ] **Step 4: Wire `JacobianLens` to the shared helpers**

In `jlenskit/core/lens.py`, add the import and `name`, add `transport`, and replace the bodies of `apply` and `_decode` to delegate (keep the method signatures unchanged):

```python
# add near the top imports
from .lens_base import apply_lens, decode_result

class JacobianLens:
    name = "jacobian"
    # ... existing __init__, layers, fit, merge unchanged ...

    def transport(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        J = self.jacobians[layer].to(h.device, torch.float32)
        return h.to(torch.float32) @ J.t()

    @torch.no_grad()
    def apply(self, adapter, prompt, positions=None, layers=None, top_k=10, decode=True):
        return apply_lens(self, adapter, prompt, positions=positions,
                          layers=layers or self.layers, top_k=top_k, decode=decode)

    def _decode(self, adapter, result, top_k):
        return decode_result(adapter, result, top_k)
```

Add to `jlenskit/core/__init__.py` exports: `from .lens_base import Lens, apply_lens` and include `"Lens"`, `"apply_lens"` in `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_lens_base.py tests/test_core.py -v`
Expected: PASS (new tests pass; existing core tests still pass).

- [ ] **Step 6: Commit**

```bash
git add jlenskit/core/lens_base.py jlenskit/core/lens.py jlenskit/core/__init__.py tests/test_lens_base.py
git commit -m "refactor: add Lens protocol + shared apply/decode; JacobianLens.transport"
```

---

### Task 2: `LogitLens` class (baselines → package)

**Files:**
- Create: `jlenskit/baselines/__init__.py`, `jlenskit/baselines/logit.py`
- Delete: `jlenskit/baselines.py`
- Test: `tests/test_baselines.py` (extend existing)

**Interfaces:**
- Consumes: `apply_lens` (Task 1).
- Produces:
  - `class LogitLens` with `name = "logit"`, `__init__(self, layers: list[int])`, `transport(self, layer, h) -> h`.
  - `logit_lens(adapter, prompt, positions=None, layers=None, top_k=10, decode=True) -> LensResult` (unchanged signature; delegates to `apply_lens(LogitLens(range(n_layers)), ...)`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_baselines.py
import torch
from jlenskit.baselines import LogitLens


def test_logitlens_transport_is_identity(toy_adapter):
    lens = LogitLens(list(range(toy_adapter.n_layers)))
    h = torch.randn(4, toy_adapter.d_model)
    assert torch.equal(lens.transport(1, h), h)
    assert lens.name == "logit"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baselines.py::test_logitlens_transport_is_identity -v`
Expected: FAIL (`cannot import name 'LogitLens'`).

- [ ] **Step 3: Convert `baselines.py` to a package**

Delete `jlenskit/baselines.py`. Create `jlenskit/baselines/logit.py`:

```python
# jlenskit/baselines/logit.py
"""Logit lens: decode a residual with no transport (identity)."""

from __future__ import annotations

import torch

from ..adapters.base import Adapter
from ..core.lens_base import apply_lens
from ..core.types import LensResult


class LogitLens:
    name = "logit"

    def __init__(self, layers: list[int]):
        self.layers = list(layers)

    def transport(self, layer: int, h: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return h


@torch.no_grad()
def logit_lens(adapter: Adapter, prompt, positions=None, layers=None, top_k=10, decode=True) -> LensResult:
    lens = LogitLens(layers or list(range(adapter.n_layers)))
    return apply_lens(lens, adapter, prompt, positions=positions, layers=lens.layers, top_k=top_k, decode=decode)
```

Create `jlenskit/baselines/__init__.py`:

```python
# jlenskit/baselines/__init__.py
"""Baseline read-outs to compare against the Jacobian lens (logit lens, tuned lens)."""

from __future__ import annotations

from .logit import LogitLens, logit_lens

__all__ = ["LogitLens", "logit_lens"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_baselines.py -v`
Expected: PASS (new test + existing baseline tests, since `logit_lens` behavior is preserved).

- [ ] **Step 5: Commit**

```bash
git add jlenskit/baselines tests/test_baselines.py
git rm jlenskit/baselines.py
git commit -m "refactor: baselines package; add LogitLens class"
```

---

### Task 3: `TunedLens` structure + save/load (no training)

**Files:**
- Create: `jlenskit/baselines/tuned.py`
- Modify: `jlenskit/baselines/__init__.py` (export `TunedLens`)
- Test: `tests/test_tuned.py`

**Interfaces:**
- Consumes: `apply_lens` (Task 1), `LogitLens` (Task 2), `LensMeta` (`jlenskit/core/types.py`).
- Produces:
  - `class TunedLens` with `name = "tuned"`, attrs `A: dict[int, torch.Tensor]` (`[D, D]`), `b: dict[int, torch.Tensor]` (`[D]`), `meta: LensMeta`, `layers` property.
  - `transport(self, layer, h) = h + h @ A[layer].t() + b[layer]`.
  - classmethod `zeros(layers: list[int], d_model: int, meta: LensMeta) -> TunedLens` (A=0, b=0 → equals logit lens).
  - `save(self, path) -> Path`, classmethod `load(cls, path, device="cpu") -> TunedLens` (safetensors, keys `A_{l}` / `b_{l}`, `LensMeta` in metadata under `jlenskit_meta`, plus `lens_type="tuned"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tuned.py
import torch
from jlenskit.baselines import LogitLens, TunedLens
from jlenskit.core.lens_base import apply_lens
from jlenskit.core.types import LensMeta


def _meta(adapter):
    return LensMeta(model_id="toy", model_revision=None, d_model=adapter.d_model,
                    vocab_size=adapter.vocab_size, n_layers=adapter.n_layers,
                    layers_fit=[0, 1, 2], n_prompts=0, n_positions=0)


def test_zero_init_tuned_equals_logit(toy_adapter):
    layers = [0, 1, 2]
    tuned = TunedLens.zeros(layers, toy_adapter.d_model, _meta(toy_adapter))
    logit = LogitLens(layers)
    rt = apply_lens(tuned, toy_adapter, "one two three", top_k=4)
    rl = apply_lens(logit, toy_adapter, "one two three", top_k=4)
    for l in layers:
        assert torch.allclose(rt.lens_logits[l], rl.lens_logits[l], atol=1e-6)


def test_tuned_save_load_roundtrip(tmp_path, toy_adapter):
    tuned = TunedLens.zeros([0, 1, 2], toy_adapter.d_model, _meta(toy_adapter))
    tuned.A[1] = torch.randn_like(tuned.A[1])
    tuned.b[1] = torch.randn_like(tuned.b[1])
    p = tuned.save(tmp_path / "tuned.safetensors")
    back = TunedLens.load(p)
    assert torch.allclose(back.A[1], tuned.A[1])
    assert torch.allclose(back.b[1], tuned.b[1])
    assert back.meta.model_id == "toy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tuned.py -v`
Expected: FAIL (`cannot import name 'TunedLens'`).

- [ ] **Step 3: Implement `TunedLens` (structure + persistence)**

```python
# jlenskit/baselines/tuned.py
"""Tuned lens (Belrose et al. 2023): a per-layer affine translator, trained by
distilling to the model's own final output. At zero init it equals the logit lens."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..core.types import LensMeta


class TunedLens:
    name = "tuned"

    def __init__(self, A: dict[int, torch.Tensor], b: dict[int, torch.Tensor], meta: LensMeta):
        self.A = A
        self.b = b
        self.meta = meta

    @property
    def layers(self) -> list[int]:
        return sorted(self.A)

    @classmethod
    def zeros(cls, layers, d_model: int, meta: LensMeta) -> "TunedLens":
        A = {l: torch.zeros(d_model, d_model) for l in layers}
        b = {l: torch.zeros(d_model) for l in layers}
        return cls(A, b, meta)

    def transport(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        A = self.A[layer].to(h.device, torch.float32)
        b = self.b[layer].to(h.device, torch.float32)
        h = h.to(torch.float32)
        return h + h @ A.t() + b

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for l in self.layers:
            tensors[f"A_{l}"] = self.A[l].contiguous().cpu()
            tensors[f"b_{l}"] = self.b[l].contiguous().cpu()
        meta = {"jlenskit_meta": json.dumps(self.meta.__dict__), "lens_type": "tuned"}
        save_file(tensors, str(path), metadata=meta)
        return path

    @classmethod
    def load(cls, path, device: str = "cpu") -> "TunedLens":
        path = Path(path)
        tensors = load_file(str(path), device=device)
        A = {int(k[2:]): v for k, v in tensors.items() if k.startswith("A_")}
        b = {int(k[2:]): v for k, v in tensors.items() if k.startswith("b_")}
        import safetensors

        with safetensors.safe_open(str(path), framework="pt") as f:
            raw = f.metadata() or {}
        meta_dict = json.loads(raw.get("jlenskit_meta", "{}"))
        meta = LensMeta(**{k: meta_dict.get(k) for k in LensMeta.__dataclass_fields__})
        return cls(A, b, meta)
```

Add to `jlenskit/baselines/__init__.py`: `from .tuned import TunedLens` and add `"TunedLens"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tuned.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jlenskit/baselines/tuned.py jlenskit/baselines/__init__.py tests/test_tuned.py
git commit -m "feat: TunedLens structure + safetensors persistence"
```

---

### Task 4: `TunedLens.fit` (training)

**Files:**
- Modify: `jlenskit/baselines/tuned.py` (add `fit`)
- Test: `tests/test_tuned.py` (add training test)

**Interfaces:**
- Consumes: `Adapter.capture`, `Adapter.to_logits`.
- Produces: classmethod `TunedLens.fit(adapter, batches, layers: list[int] | None = None, n_steps: int = 250, lr: float = 1e-3, seed: int | None = None, corpus_spec: dict | None = None, progress: bool = False) -> TunedLens`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_tuned.py
from jlenskit.baselines import LogitLens
from jlenskit.metrics import forward_kl  # defined in Task 5; this test runs after Task 5


def test_tuned_fit_reduces_forward_kl(toy_adapter, toy_batches):
    layers = [0, 1, 2]
    tuned = TunedLens.fit(toy_adapter, toy_batches, layers=layers, n_steps=100, lr=1e-2, seed=0)
    logit = LogitLens(layers)
    kl_t = forward_kl(toy_adapter, tuned, toy_batches)
    kl_l = forward_kl(toy_adapter, logit, toy_batches)
    mean_t = sum(kl_t.values()) / len(kl_t)
    mean_l = sum(kl_l.values()) / len(kl_l)
    assert mean_t <= mean_l + 1e-4  # training can only help on the fit corpus
```

> Note: this test depends on `forward_kl` from Task 5. If executing strictly in order, mark it xfail until Task 5 lands, or implement Task 5 before running it. The subagent executing Task 5 must re-run this test.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tuned.py::test_tuned_fit_reduces_forward_kl -v`
Expected: FAIL (`TunedLens has no attribute 'fit'`, or import error for `forward_kl`).

- [ ] **Step 3: Implement `fit`**

```python
# add to jlenskit/baselines/tuned.py (method on TunedLens)
    @classmethod
    def fit(cls, adapter, batches, layers=None, n_steps=250, lr=1e-3, seed=None,
            corpus_spec=None, progress=False):
        if seed is not None:
            torch.manual_seed(seed)
        if layers is None:
            layers = list(range(adapter.n_layers))
        d = adapter.d_model
        device = adapter.device

        # Cache residuals + model target log-probs once (no grad through the model).
        cached = []  # list of (residuals: {l: [P, D]}, target_logp: [P, vocab])
        with torch.no_grad():
            for input_ids in batches:
                cap = adapter.capture(input_ids, requires_grad=False)
                target = torch.log_softmax(cap.model_logits.float(), dim=-1).reshape(-1, adapter.vocab_size)
                res = {l: cap.residuals[l].float().reshape(-1, d) for l in layers}
                cached.append((res, target))

        A = {l: torch.zeros(d, d, device=device, requires_grad=True) for l in layers}
        b = {l: torch.zeros(d, device=device, requires_grad=True) for l in layers}
        params = [A[l] for l in layers] + [b[l] for l in layers]
        opt = torch.optim.Adam(params, lr=lr)

        for step in range(n_steps):
            opt.zero_grad()
            loss = torch.zeros((), device=device)
            for res, target in cached:
                target_p = target.exp()
                for l in layers:
                    h = res[l].to(device)
                    transported = h + h @ A[l].t() + b[l]
                    lens_logp = torch.log_softmax(
                        adapter.to_logits(transported.to(adapter.dtype)).float(), dim=-1
                    )
                    # forward KL(model || lens), summed over vocab, mean over positions
                    loss = loss + (target_p * (target - lens_logp)).sum(dim=-1).mean()
            loss.backward()
            opt.step()
            if progress and step % 50 == 0:
                print(f"[tuned] step {step} loss {float(loss):.4f}")

        meta = LensMeta(
            model_id=getattr(adapter.model.config, "_name_or_path", "unknown"),
            model_revision=None, d_model=d, vocab_size=adapter.vocab_size,
            n_layers=adapter.n_layers, layers_fit=sorted(layers),
            n_prompts=len(cached), n_positions=sum(t.shape[0] for _, t in cached),
            corpus_spec=corpus_spec or {}, fit_config={"n_steps": n_steps, "lr": lr},
            seed=seed, jlenskit_version="0.1.0", extra={"lens_type": "tuned"},
        )
        return cls({l: A[l].detach().cpu() for l in layers},
                   {l: b[l].detach().cpu() for l in layers}, meta)
```

- [ ] **Step 4: Run tests to verify they pass** (after Task 5's `forward_kl` exists)

Run: `.venv/bin/python -m pytest tests/test_tuned.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jlenskit/baselines/tuned.py tests/test_tuned.py
git commit -m "feat: TunedLens.fit (distill to model final via forward KL)"
```

---

### Task 5: coherence metrics (`forward_kl`, `entropy`) generalized to any Lens

**Files:**
- Modify: `jlenskit/metrics/__init__.py` (generalize `_lens_logits_for_batch`, `topk_accuracy`; add `forward_kl`, `entropy`)
- Test: `tests/test_metrics.py` (extend)

**Interfaces:**
- Consumes: `Lens` protocol (`transport`), `Adapter`.
- Produces:
  - `forward_kl(adapter, lens, batches, layers: list[int] | None = None) -> dict[int, float]` — mean `KL(model_final ‖ lens_ℓ)` in nats.
  - `entropy(adapter, lens, batches, layers: list[int] | None = None) -> dict[int, float]` — mean Shannon entropy (nats) of the lens distribution.
  - `topk_accuracy(adapter, lens, batches, k=5)` unchanged signature, now accepts any `Lens`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_metrics.py
import torch
from jlenskit.baselines import LogitLens
from jlenskit.metrics import entropy, forward_kl


def test_forward_kl_nonnegative_and_shaped(toy_adapter, toy_lens):
    kl = forward_kl(toy_adapter, toy_lens, [torch.randint(1, 64, (2, 8))])
    assert set(kl) == set(toy_lens.layers)
    assert all(v >= -1e-5 for v in kl.values())


def test_entropy_matches_logit_lens_manual(toy_adapter):
    lens = LogitLens(list(range(toy_adapter.n_layers)))
    ent = entropy(toy_adapter, lens, [torch.randint(1, 64, (1, 4))])
    assert set(ent) == set(lens.layers)
    assert all(v >= 0 for v in ent.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py::test_forward_kl_nonnegative_and_shaped -v`
Expected: FAIL (`cannot import name 'forward_kl'`).

- [ ] **Step 3: Generalize the batch helper and add metrics**

In `jlenskit/metrics/__init__.py`, change `_lens_logits_for_batch` to use `lens.transport` instead of `lens.jacobians`:

```python
@torch.no_grad()
def _lens_logits_for_batch(adapter, lens, input_ids, layers):
    cap = adapter.capture(input_ids, requires_grad=False)
    out = {}
    for layer in layers:
        residual = cap.residuals[layer].float()  # [b, s, D]
        transported = lens.transport(layer, residual)  # any Lens
        out[layer] = adapter.to_logits(transported.to(adapter.dtype)).float()
    return out
```

Change the `topk_accuracy` type hint from `lens: JacobianLens` to `lens` (accept any Lens) — behavior is unchanged since it already only calls `_lens_logits_for_batch`. Append:

```python
@torch.no_grad()
def forward_kl(adapter, lens, batches, layers=None):
    """Mean KL(model_final || lens_l) per layer (nats). Lower = more coherent."""
    layers = layers or list(lens.layers)
    kl_sum = {l: 0.0 for l in layers}
    total = 0
    for input_ids in batches:
        cap = adapter.capture(input_ids, requires_grad=False)
        model_logp = torch.log_softmax(cap.model_logits.float(), dim=-1)
        model_p = model_logp.exp()
        total += model_p.reshape(-1, adapter.vocab_size).shape[0]
        for l in layers:
            h = cap.residuals[l].float()
            lens_logp = torch.log_softmax(
                adapter.to_logits(lens.transport(l, h).to(adapter.dtype)).float(), dim=-1
            )
            kl = (model_p * (model_logp - lens_logp)).sum(dim=-1)  # [b, s]
            kl_sum[l] += float(kl.sum().item())
    return {l: (kl_sum[l] / total if total else 0.0) for l in layers}


@torch.no_grad()
def entropy(adapter, lens, batches, layers=None):
    """Mean Shannon entropy (nats) of the lens distribution per layer."""
    layers = layers or list(lens.layers)
    ent_sum = {l: 0.0 for l in layers}
    total = 0
    for input_ids in batches:
        logits = _lens_logits_for_batch(adapter, lens, input_ids, layers)
        for l in layers:
            logp = torch.log_softmax(logits[l], dim=-1)
            ent = -(logp.exp() * logp).sum(dim=-1)  # [b, s]
            ent_sum[l] += float(ent.sum().item())
            n = ent.numel()
        total += n
    return {l: (ent_sum[l] / total if total else 0.0) for l in layers}
```

Add `"forward_kl"` and `"entropy"` to `__all__`.

- [ ] **Step 4: Run tests (including Task 4's) to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_metrics.py tests/test_tuned.py -v`
Expected: PASS (this is where `test_tuned_fit_reduces_forward_kl` finally runs green).

- [ ] **Step 5: Commit**

```bash
git add jlenskit/metrics/__init__.py tests/test_metrics.py
git commit -m "feat: forward_kl + entropy coherence metrics for any Lens"
```

---

### Task 6: knowledge-probe suite + loader + elicitation depth

**Files:**
- Create: `jlenskit/data/probes/core_v1.json`
- Create: `jlenskit/data/probes.py`
- Modify: `jlenskit/data/__init__.py` (export `load_probes`, `Probe`)
- Test: `tests/test_probes.py`

**Interfaces:**
- Produces:
  - `@dataclass class Probe: prompt: str; answers: list[str]; category: str`
  - `load_probes(name: str = "core_v1") -> list[Probe]` (reads `jlenskit/data/probes/<name>.json`).
  - `elicitation_depth(adapter, lens, probes: list[Probe], top_k: int = 5) -> dict` returning `{"per_probe": [{prompt, category, depth: int|None}], "median_by_category": {category: float|None}}`. `depth` = first layer whose top-k decoded tokens (stripped, case-insensitive) contain any answer surface form.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probes.py
from jlenskit.data import Probe, load_probes
from jlenskit.data.probes import elicitation_depth


def test_load_probes_parses_shipped_suite():
    probes = load_probes("core_v1")
    assert len(probes) >= 20
    assert all(isinstance(p, Probe) and p.prompt and p.answers and p.category for p in probes)


def test_elicitation_depth_structure(toy_adapter, toy_lens):
    probes = load_probes("core_v1")[:3]
    out = elicitation_depth(toy_adapter, toy_lens, probes, top_k=5)
    assert len(out["per_probe"]) == 3
    assert "median_by_category" in out
    for row in out["per_probe"]:
        assert row["depth"] is None or isinstance(row["depth"], int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_probes.py -v`
Expected: FAIL (module/import missing).

- [ ] **Step 3: Ship the probe dataset**

Create `jlenskit/data/probes/core_v1.json` with at least 20 entries across `factual`, `two_hop`, `antonym`, `language`. Full content:

```json
[
  {"prompt": "The capital of France is", "answers": ["Paris"], "category": "factual"},
  {"prompt": "The capital of Japan is", "answers": ["Tokyo"], "category": "factual"},
  {"prompt": "The capital of Italy is", "answers": ["Rome"], "category": "factual"},
  {"prompt": "The capital of Germany is", "answers": ["Berlin"], "category": "factual"},
  {"prompt": "The chemical symbol for gold is", "answers": ["Au"], "category": "factual"},
  {"prompt": "The largest planet in the solar system is", "answers": ["Jupiter"], "category": "factual"},
  {"prompt": "The country whose capital city is Tokyo uses a currency called the", "answers": ["yen"], "category": "two_hop"},
  {"prompt": "The currency used in the country whose capital is Paris is the", "answers": ["euro"], "category": "two_hop"},
  {"prompt": "The language spoken in the country whose capital is Madrid is", "answers": ["Spanish"], "category": "two_hop"},
  {"prompt": "The continent containing the country whose capital is Cairo is", "answers": ["Africa"], "category": "two_hop"},
  {"prompt": "The ocean on the west coast of the country whose capital is Washington is the", "answers": ["Pacific"], "category": "two_hop"},
  {"prompt": "The opposite of hot is", "answers": ["cold"], "category": "antonym"},
  {"prompt": "The opposite of up is", "answers": ["down"], "category": "antonym"},
  {"prompt": "The opposite of big is", "answers": ["small", "little"], "category": "antonym"},
  {"prompt": "The opposite of fast is", "answers": ["slow"], "category": "antonym"},
  {"prompt": "The opposite of happy is", "answers": ["sad", "unhappy"], "category": "antonym"},
  {"prompt": "The opposite of open is", "answers": ["closed", "shut"], "category": "antonym"},
  {"prompt": "She was born in Paris and grew up speaking the language called", "answers": ["French"], "category": "language"},
  {"prompt": "He was born in Berlin and grew up speaking the language called", "answers": ["German"], "category": "language"},
  {"prompt": "The people of Brazil mostly speak the language called", "answers": ["Portuguese"], "category": "language"},
  {"prompt": "The people of Mexico mostly speak the language called", "answers": ["Spanish"], "category": "language"},
  {"prompt": "Water is made of hydrogen and", "answers": ["oxygen"], "category": "factual"}
]
```

- [ ] **Step 4: Implement loader + elicitation depth**

```python
# jlenskit/data/probes.py
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
        wanted = [a.strip().lower() for a in p.answers]
        depth = None
        for dl in res.decoded[0]:  # ordered by layer
            toks = [t.strip().lower() for t in dl.tokens]
            if any(any(w in t or t in w for t in toks) for w in wanted):
                depth = dl.layer
                break
        per_probe.append({"prompt": p.prompt, "category": p.category, "depth": depth})
        if depth is not None:
            by_cat.setdefault(p.category, []).append(depth)
    median_by_category = {c: (median(v) if v else None) for c, v in by_cat.items()}
    return {"per_probe": per_probe, "median_by_category": median_by_category}
```

Add to `jlenskit/data/__init__.py`: `from .probes import Probe, load_probes` and add both to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_probes.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add jlenskit/data/probes jlenskit/data/probes.py jlenskit/data/__init__.py tests/test_probes.py
git commit -m "feat: knowledge-probe suite + elicitation-depth metric"
```

---

### Task 7: `ShowdownCfg` + `run_showdown` orchestration

**Files:**
- Modify: `jlenskit/config.py` (add `ShowdownCfg`, `Config.showdown`)
- Create: `jlenskit/showdown.py`
- Test: `tests/test_showdown.py`

**Interfaces:**
- Consumes: `LogitLens`, `TunedLens` (Tasks 2–4), `forward_kl`/`entropy`/`topk_accuracy` (Task 5), `load_probes`/`elicitation_depth` (Task 6), a fitted `JacobianLens`.
- Produces:
  - `class ShowdownCfg(BaseModel)`: `lenses: list[str] = ["logit","tuned","jacobian"]`, `probes: str | None = "core_v1"`, `coherence_tau: float = 0.5`, `tuned_steps: int = 250`, `tuned_lr: float = 1e-3`, `top_k: int = 5`.
  - `run_showdown(cfg, adapter, jlens, batches, tuned=None) -> dict` returning `{"lenses": {name: {"forward_kl": {...}, "entropy": {...}, "topk_accuracy": {...}, "layers_to_coherence": int|None}}, "elicitation": {name: {...}} | None}`.
  - `layers_to_coherence(kl: dict[int, float], tau: float) -> int | None`.
  - `write_showdown_outputs(results: dict, out_dir) -> dict[str, str]` writing `showdown_metrics.json` + `showdown.md`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_showdown.py
from jlenskit.baselines import TunedLens
from jlenskit.config import ShowdownCfg
from jlenskit.showdown import layers_to_coherence, run_showdown, write_showdown_outputs


def test_layers_to_coherence():
    kl = {0: 10.0, 1: 8.0, 2: 3.0, 3: 2.0}
    # tau=0.5 -> threshold = 5.0; first layer with kl<=5 that stays below = 2
    assert layers_to_coherence(kl, 0.5) == 2
    assert layers_to_coherence({0: 10.0, 1: 9.0}, 0.5) is None


def test_run_showdown_and_write(tmp_path, toy_adapter, toy_lens, toy_batches):
    cfg = ShowdownCfg(lenses=["logit", "tuned", "jacobian"], probes="core_v1",
                      tuned_steps=20, tuned_lr=1e-2)
    tuned = TunedLens.fit(toy_adapter, toy_batches, layers=toy_lens.layers, n_steps=20, lr=1e-2, seed=0)
    results = run_showdown(cfg, toy_adapter, toy_lens, toy_batches, tuned=tuned)
    for name in ["logit", "tuned", "jacobian"]:
        assert "forward_kl" in results["lenses"][name]
        assert "layers_to_coherence" in results["lenses"][name]
    paths = write_showdown_outputs(results, tmp_path)
    assert (tmp_path / "showdown_metrics.json").exists()
    assert (tmp_path / "showdown.md").exists()
    assert "logit" in (tmp_path / "showdown.md").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_showdown.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Add `ShowdownCfg` to config**

In `jlenskit/config.py`, add the class and field:

```python
class ShowdownCfg(BaseModel):
    lenses: list[str] = Field(default_factory=lambda: ["logit", "tuned", "jacobian"])
    probes: str | None = "core_v1"
    coherence_tau: float = 0.5
    tuned_steps: int = 250
    tuned_lr: float = 1e-3
    top_k: int = 5
```

Add to `Config`: `showdown: ShowdownCfg | None = None`.

- [ ] **Step 4: Implement the harness**

```python
# jlenskit/showdown.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_showdown.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add jlenskit/config.py jlenskit/showdown.py tests/test_showdown.py
git commit -m "feat: run_showdown harness + ShowdownCfg + markdown/JSON outputs"
```

---

### Task 8: showdown HTML viz (forward-KL vs layer)

**Files:**
- Create: `jlenskit/viz/showdown.py`
- Modify: `jlenskit/viz/__init__.py` (export `render_showdown`)
- Test: `tests/test_showdown.py` (add viz test)

**Interfaces:**
- Consumes: the `results` dict from `run_showdown` (Task 7).
- Produces: `render_showdown(results: dict, path) -> Path` writing a self-contained HTML file with an inline SVG line chart (forward-KL vs layer, one polyline per lens) — no external JS/CDN.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_showdown.py
from jlenskit.viz import render_showdown


def test_render_showdown_writes_html(tmp_path):
    results = {"lenses": {
        "logit": {"forward_kl": {0: 9.0, 1: 7.0, 2: 2.0}, "entropy": {}, "topk_accuracy": {2: 0.5}, "layers_to_coherence": 2},
        "jacobian": {"forward_kl": {0: 4.0, 1: 2.0, 2: 1.0}, "entropy": {}, "topk_accuracy": {2: 0.6}, "layers_to_coherence": 1},
    }, "elicitation": None}
    p = render_showdown(results, tmp_path / "showdown.html")
    html = p.read_text()
    assert "<svg" in html and "logit" in html and "jacobian" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_showdown.py::test_render_showdown_writes_html -v`
Expected: FAIL (`cannot import name 'render_showdown'`).

- [ ] **Step 3: Implement the SVG renderer**

```python
# jlenskit/viz/showdown.py
"""Self-contained SVG line chart: forward-KL vs layer, one line per lens."""

from __future__ import annotations

from pathlib import Path

_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def render_showdown(results: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H, pad = 720, 380, 48
    series = {name: m["forward_kl"] for name, m in results["lenses"].items() if m["forward_kl"]}
    all_layers = sorted({l for kl in series.values() for l in kl})
    all_vals = [v for kl in series.values() for v in kl.values()]
    if not all_layers or not all_vals:
        path.write_text("<html><body><p>no data</p></body></html>", encoding="utf-8")
        return path
    lmin, lmax = all_layers[0], all_layers[-1]
    vmax = max(all_vals) or 1.0

    def x(l):
        return pad + (W - 2 * pad) * (l - lmin) / max(lmax - lmin, 1)

    def y(v):
        return H - pad - (H - 2 * pad) * (v / vmax)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W/2}" y="20" text-anchor="middle" font-family="sans-serif" font-size="14">'
             'forward KL(model &#8214; lens) vs layer (lower = more coherent)</text>',
             f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#999"/>',
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" stroke="#999"/>']
    for i, (name, kl) in enumerate(series.items()):
        color = _COLORS[i % len(_COLORS)]
        pts = " ".join(f"{x(l):.1f},{y(kl[l]):.1f}" for l in sorted(kl))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
        parts.append(f'<text x="{W-pad-90}" y="{pad+16+18*i}" fill="{color}" '
                     f'font-family="sans-serif" font-size="13">{name}</text>')
    parts.append("</svg>")
    path.write_text("<html><body>" + "".join(parts) + "</body></html>", encoding="utf-8")
    return path
```

Add to `jlenskit/viz/__init__.py`: `from .showdown import render_showdown` and add `"render_showdown"` to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_showdown.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jlenskit/viz/showdown.py jlenskit/viz/__init__.py tests/test_showdown.py
git commit -m "feat: showdown HTML viz (forward-KL vs layer)"
```

---

### Task 9: CLI `showdown` command + configs

**Files:**
- Modify: `jlenskit/cli.py` (add `showdown` command; helper to resolve/cache the tuned lens)
- Modify: `configs/qwen05b_mps.yaml`, `configs/gpt2_demo.yaml` (add `showdown:` section)
- Test: `tests/test_cli.py` (add a registration/parse test)

**Interfaces:**
- Consumes: `_adapter`, `_resolve_lens`, `_finish` (existing in `cli.py`); `TunedLens.fit`, `run_showdown`, `write_showdown_outputs`, `render_showdown`, `LensCache`.
- Produces: `jlenskit showdown CONFIG` writing `showdown_metrics.json`, `showdown.md`, `showdown.html`, and a manifest.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_cli.py
from typer.testing import CliRunner
from jlenskit.cli import app


def test_showdown_command_registered():
    runner = CliRunner()
    result = runner.invoke(app, ["showdown", "--help"])
    assert result.exit_code == 0
    assert "showdown" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli.py::test_showdown_command_registered -v`
Expected: FAIL (no such command).

- [ ] **Step 3: Add the CLI command**

In `jlenskit/cli.py`, add imports and the command:

```python
from .baselines import TunedLens
from .config import ShowdownCfg
from .showdown import run_showdown, write_showdown_outputs
from .viz import render_showdown  # render already imported; add render_showdown


@app.command()
def showdown(config: str):
    """Compare logit / tuned / Jacobian lenses on coherence + knowledge elicitation."""
    cfg = load_config(config)
    sc = cfg.showdown or ShowdownCfg()
    adapter = _adapter(cfg)
    jlens = _resolve_lens(cfg, adapter)
    batches = load_corpus(cfg.corpus.to_spec(), adapter.tokenizer)

    tuned = None
    if "tuned" in sc.lenses:
        cache = LensCache()
        key = cache.key(cfg.model.id, cfg.model.revision, cfg.corpus.to_spec(),
                        {"tuned": True, "steps": sc.tuned_steps, "lr": sc.tuned_lr, "layers": jlens.layers})
        tuned_path = Path(cfg.output.dir) / "tuned_lens.safetensors"
        cached = cache.get(key) if cfg.lens.use_cache else None
        if isinstance(cached, TunedLens):
            typer.echo(f"[jlenskit] using cached tuned lens {key}")
            tuned = cached
        else:
            typer.echo(f"[jlenskit] training tuned lens ({sc.tuned_steps} steps) ...")
            tuned = TunedLens.fit(adapter, batches, layers=jlens.layers,
                                  n_steps=sc.tuned_steps, lr=sc.tuned_lr, seed=cfg.seed,
                                  corpus_spec=cfg.corpus.to_spec(), progress=True)
            tuned.save(tuned_path)

    results = run_showdown(sc, adapter, jlens, batches, tuned=tuned)
    out_dir = Path(cfg.output.dir)
    outputs = write_showdown_outputs(results, out_dir)
    outputs["showdown_html"] = str(render_showdown(results, out_dir / "showdown.html"))
    for name, m in results["lenses"].items():
        typer.echo(f"[jlenskit] {name}: layers-to-coherence = {m['layers_to_coherence']}")
    _finish(cfg, adapter, jlens, "showdown", outputs)
```

> Note: `LensCache.get` may return a `JacobianLens`; guard with `isinstance(cached, TunedLens)`. If the cache backend cannot store a `TunedLens`, skip caching and always retrain (acceptable — training is cheap). Verify `LensCache` behavior when implementing; if it only handles `JacobianLens`, drop the cache lookup for tuned and rely on `tuned_lens.safetensors` + a `TunedLens.load` on re-run instead.

- [ ] **Step 4: Extend the two configs**

Append to `configs/qwen05b_mps.yaml`:

```yaml
showdown:
  lenses: [logit, tuned, jacobian]
  probes: core_v1
  coherence_tau: 0.5
  tuned_steps: 250
  tuned_lr: 0.001
  top_k: 5
```

Append the identical block to `configs/gpt2_demo.yaml`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add jlenskit/cli.py configs/qwen05b_mps.yaml configs/gpt2_demo.yaml tests/test_cli.py
git commit -m "feat: jlenskit showdown CLI command + configs"
```

---

### Task 10: cleanup, docs, full verification, real run

**Files:**
- Delete: `scratch_silent.py`, `scratch_curve.py`, `runs_rerun.log`, `runs_eval2.log`
- Modify: `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Remove scratch artifacts**

```bash
git rm -f scratch_silent.py scratch_curve.py 2>/dev/null; rm -f runs_rerun.log runs_eval2.log
```

- [ ] **Step 2: Update docs**

Add a "Lens showdown" subsection to `README.md` (after the logit-lens paragraph) describing `jlenskit showdown CONFIG`, the three lenses table, and the `showdown.md`/`.html` outputs. Add a `CHANGELOG.md` entry under a new dated heading: tuned-lens baseline, `forward_kl`/`entropy` coherence metrics, knowledge-probe suite, `showdown` command.

- [ ] **Step 3: Full test + lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/ruff check jlenskit tests`
Expected: all tests pass, ruff clean.

- [ ] **Step 4: Real run on both families**

```bash
source .env; export HUGGINGFACE_ACCESS_TOKEN
.venv/bin/jlenskit showdown configs/qwen05b_mps.yaml
.venv/bin/jlenskit showdown configs/gpt2_demo.yaml
```
Expected: each writes `runs/<dir>/showdown.md`, `showdown.html`, `showdown_metrics.json`. Inspect `showdown.md`: the Qwen J-lens `layers-to-coherence` should be materially shallower than the logit lens, with the tuned lens comparable to the J-lens.

- [ ] **Step 5: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: document lens showdown; remove scratch prototypes"
```

---

## Self-review notes

- **Spec coverage:** Lens protocol (T1), tuned lens struct+train (T3,T4), coherence metrics + layers-to-coherence (T5,T7), probe suite + elicitation depth (T6), showdown harness + outputs (T7), viz (T8), CLI + two configs (T9), scratch cleanup + real run (T10). All spec sections mapped.
- **Cross-task dependency to watch:** Task 4's test imports `forward_kl` from Task 5. Execute Task 5 before green-running Task 4 (noted in Task 4 Step 1/4). The subagent for Task 5 must re-run `tests/test_tuned.py`.
- **`LensCache` for tuned lens:** Task 9 flags the isinstance guard / fallback — verify the cache API at implementation time rather than assuming it stores non-Jacobian lenses.
- **Toy-fixture reality:** `StubTokenizer.decode` returns `<id>` strings, so probe elicitation on the toy model will mostly return `depth=None`; the probe tests assert *structure*, not hits. Real answer-hit behavior is validated by the Task 10 real run.
