"""Self-contained HTML viewers for :class:`~jlenskit.core.types.LensResult`.

The Jacobian lens shows, per layer, the tokens a hidden state is "disposed to
say". These helpers render that read-out as an offline, single-file HTML
document (all CSS inlined, no external/CDN resources) so a researcher can watch
"silent" tokens evolve across layers.

Two views:

* :func:`slice_html` -- one position, rows = layers, each row shows the full
  top-k list as colored chips (darker = higher probability).
* :func:`grid_html` -- all positions as columns, all layers as rows; each cell
  is the top-1 token colored by probability with the full top-k on hover.

Everything is pure-Python string templating (no jinja, no external deps).
"""

from __future__ import annotations

import html
from pathlib import Path

from ..core.types import DecodedLayer, LensResult

__all__ = ["slice_html", "grid_html", "save_html", "render"]


# -- helpers ------------------------------------------------------------------
def _require_decoded(result: LensResult) -> None:
    if not result.decoded:
        raise ValueError(
            "LensResult.decoded is empty; call JacobianLens.apply(..., decode=True) "
            "before rendering (viz needs the decoded top-k tokens)."
        )


def _esc(token: str) -> str:
    """HTML-escape a token and make whitespace visible.

    Tokens can contain ``<``, ``>``, ``&``, quotes, spaces and newlines. We
    escape everything first (quote=True so ``"`` is safe inside attributes),
    then render spaces as a middle dot and newlines as a small marker so the
    reader can see them.
    """
    safe = html.escape(token, quote=True)
    safe = safe.replace("\n", "↵")  # downwards arrow with corner (visible newline)
    safe = safe.replace("\t", "→")  # rightwards arrow (visible tab)
    safe = safe.replace(" ", "·")  # middle dot (visible space)
    return safe


def _bg(score: float) -> str:
    """Blue background whose opacity tracks the probability (0..1)."""
    s = max(0.0, min(1.0, float(score)))
    # opacity floors at ~0.08 so even tiny scores are faintly tinted.
    alpha = 0.08 + 0.92 * s
    return f"rgba(30, 90, 200, {alpha:.3f})"


def _fg(score: float) -> str:
    """White text once the blue is dark enough, otherwise near-black."""
    return "#f7f9ff" if float(score) >= 0.5 else "#101418"


def _title_attr(dl: DecodedLayer) -> str:
    """Full top-k list for a hover tooltip (title attribute, so escaped)."""
    parts = [f"{html.escape(t, quote=True)} {sc:.4f}" for t, sc in zip(dl.tokens, dl.scores)]
    return " | ".join(parts)


_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  margin: 0; padding: 24px; background: #fbfcfe; color: #101418;
}
header { margin-bottom: 20px; }
header h1 { font-size: 18px; margin: 0 0 8px; }
.meta { font-size: 13px; color: #556; }
.meta code {
  background: #eef1f7; padding: 2px 6px; border-radius: 4px;
  white-space: pre-wrap; word-break: break-all;
}
table { border-collapse: collapse; font-size: 13px; }
th, td {
  border: 1px solid #d7dbe4; padding: 4px 8px; text-align: center;
  white-space: nowrap;
}
th { background: #eef1f7; position: sticky; top: 0; font-weight: 600; }
.rowhdr { background: #f4f6fa; font-weight: 600; text-align: right; }
.chip {
  display: inline-block; margin: 2px; padding: 3px 8px; border-radius: 5px;
  font-size: 12px; white-space: pre; cursor: default;
}
.chip .sc { opacity: 0.75; font-size: 10px; margin-left: 4px; }
.cell { min-width: 64px; cursor: default; }
.cell .sc { display: block; font-size: 10px; opacity: 0.8; }
.legend { margin-top: 16px; font-size: 12px; color: #556; }
.legend .bar {
  display: inline-block; width: 180px; height: 12px; vertical-align: middle;
  border: 1px solid #ccd; border-radius: 3px;
  background: linear-gradient(to right, rgba(30,90,200,0.08), rgba(30,90,200,1));
}
"""


def _doc(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_STYLE}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def _header(title: str, result: LensResult, extra: str = "") -> str:
    return (
        "<header>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f'<div class="meta">prompt: <code>{_esc(result.prompt)}</code></div>\n'
        f'<div class="meta">positions: <code>{html.escape(str(result.positions))}</code>{extra}</div>\n'
        "</header>\n"
    )


# -- slice (single position) --------------------------------------------------
def slice_html(result: LensResult, position_index: int = 0, title: str | None = None) -> str:
    """Render a single position: rows = layers, top-k tokens as colored chips."""
    _require_decoded(result)
    if position_index not in result.decoded:
        raise ValueError(
            f"position_index {position_index} not in decoded positions "
            f"{sorted(result.decoded)}."
        )

    per_layer = result.decoded[position_index]
    title = title or f"J-lens slice — position {position_index}"

    rows = []
    for dl in sorted(per_layer, key=lambda d: d.layer):
        chips = []
        for tok, sc in zip(dl.tokens, dl.scores):
            chips.append(
                f'<span class="chip" style="background:{_bg(sc)};color:{_fg(sc)}" '
                f'title="{_esc(tok)} {sc:.4f}">{_esc(tok)}'
                f'<span class="sc">{sc:.2f}</span></span>'
            )
        rows.append(
            f'<tr><td class="rowhdr">L{dl.layer}</td>'
            f'<td style="text-align:left;white-space:normal">{"".join(chips)}</td></tr>'
        )

    body = (
        _header(title, result, extra=f" &middot; showing position index <code>{position_index}</code>")
        + '<table>\n<thead><tr><th>layer</th><th>top-k lens tokens (darker = higher prob)</th></tr></thead>\n'
        + "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>\n"
        + _legend()
    )
    return _doc(title, body)


# -- grid (all positions) -----------------------------------------------------
def grid_html(result: LensResult, title: str | None = None) -> str:
    """Render all positions as columns, all layers as rows; top-1 per cell."""
    _require_decoded(result)
    title = title or "J-lens grid"

    position_indices = sorted(result.decoded)
    # Union of layers across positions, sorted.
    layers = sorted({dl.layer for pl in result.decoded.values() for dl in pl})

    # Index: position_index -> {layer -> DecodedLayer}
    by_pos: dict[int, dict[int, DecodedLayer]] = {
        pi: {dl.layer: dl for dl in pl} for pi, pl in result.decoded.items()
    }

    head_cols = "".join(
        f'<th>pos {pi}<br><small>({result.positions[pi]})</small></th>'
        if pi < len(result.positions)
        else f"<th>pos {pi}</th>"
        for pi in position_indices
    )

    rows = []
    for layer in layers:
        cells = []
        for pi in position_indices:
            dl = by_pos.get(pi, {}).get(layer)
            if dl is None or not dl.tokens:
                cells.append('<td class="cell">&mdash;</td>')
                continue
            tok, sc = dl.tokens[0], dl.scores[0]
            cells.append(
                f'<td class="cell" style="background:{_bg(sc)};color:{_fg(sc)}" '
                f'title="{_title_attr(dl)}">{_esc(tok)}'
                f'<span class="sc">{sc:.3f}</span></td>'
            )
        rows.append(f'<tr><td class="rowhdr">L{layer}</td>{"".join(cells)}</tr>')

    body = (
        _header(title, result)
        + f'<table>\n<thead><tr><th>layer \\ pos</th>{head_cols}</tr></thead>\n'
        + "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>\n"
        + _legend()
    )
    return _doc(title, body)


def _legend() -> str:
    return (
        '<div class="legend">Color = top-1 probability: '
        '<span>low</span> <span class="bar"></span> <span>high</span>. '
        "Hover a cell/chip for the full top-k list. "
        "Spaces shown as &middot;, newlines as ↵.</div>\n"
    )


# -- saving -------------------------------------------------------------------
def save_html(html_str: str, path: str | Path) -> Path:
    """Write ``html_str`` to ``path`` (utf-8), creating parents. Returns Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_str, encoding="utf-8")
    return path


def render(result: LensResult, path: str | Path, kind: str = "grid") -> Path:
    """Build ``grid_html`` or ``slice_html`` and save it to ``path``."""
    if kind == "grid":
        doc = grid_html(result)
    elif kind == "slice":
        doc = slice_html(result)
    else:
        raise ValueError(f"kind must be 'grid' or 'slice', got {kind!r}.")
    return save_html(doc, path)
