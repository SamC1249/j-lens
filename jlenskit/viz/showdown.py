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
