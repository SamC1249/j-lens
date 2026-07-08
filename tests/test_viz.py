"""Tests for the self-contained HTML slice/grid viewers."""

from __future__ import annotations

import pytest

from jlenskit.viz import grid_html, render, save_html, slice_html


@pytest.fixture
def result(toy_lens, toy_adapter):
    return toy_lens.apply(toy_adapter, "the quick brown fox", positions=[-2, -1], top_k=4)


def _decoded_tokens(result):
    return [
        tok
        for per_layer in result.decoded.values()
        for dl in per_layer
        for tok in dl.tokens
    ]


def test_slice_html_is_complete_document(result):
    h = slice_html(result)
    assert h
    assert h.startswith("<!DOCTYPE html>")
    assert "<html" in h
    assert "<style" in h


def test_grid_html_is_complete_document(result):
    h = grid_html(result)
    assert h
    assert h.startswith("<!DOCTYPE html>")
    assert "<html" in h
    assert "<style" in h


def test_html_contains_a_decoded_token_escaped(result):
    import html as _html

    h = grid_html(result)
    tokens = _decoded_tokens(result)
    assert tokens, "expected some decoded tokens"
    # StubTokenizer decodes ids as "<NN>", so escaped tokens look like "&lt;NN&gt;".
    escaped = [_html.escape(t) for t in tokens]
    assert any(e in h for e in escaped), "no decoded (escaped) token found in HTML"


def test_special_chars_are_escaped(result):
    # StubTokenizer emits tokens containing "<" and ">", so html.escape must run.
    slice_doc = slice_html(result)
    grid_doc = grid_html(result)
    assert "&lt;" in slice_doc
    assert "&lt;" in grid_doc
    # Raw unescaped token markup must not leak into the body as a fake tag.
    assert "<0>" not in slice_doc


def test_save_html_writes_nonempty_file(tmp_path, result):
    out = save_html(grid_html(result), tmp_path / "sub" / "g.html")
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_returns_existing_path(tmp_path, result):
    out = render(result, tmp_path / "v.html")
    assert out.exists()
    assert out.stat().st_size > 0

    out_slice = render(result, tmp_path / "s.html", kind="slice")
    assert out_slice.exists()


def test_empty_decoded_raises(result):
    result.decoded = {}
    with pytest.raises(ValueError, match="decode=True"):
        grid_html(result)
    with pytest.raises(ValueError, match="decode=True"):
        slice_html(result)
