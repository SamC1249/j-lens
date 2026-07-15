from jlenskit.baselines import TunedLens
from jlenskit.config import ShowdownCfg
from jlenskit.showdown import layers_to_coherence, run_showdown, write_showdown_outputs
from jlenskit.viz import render_showdown


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
    write_showdown_outputs(results, tmp_path)
    assert (tmp_path / "showdown_metrics.json").exists()
    assert (tmp_path / "showdown.md").exists()
    assert "logit" in (tmp_path / "showdown.md").read_text()


def test_render_showdown_writes_html(tmp_path):
    results = {"lenses": {
        "logit": {"forward_kl": {0: 9.0, 1: 7.0, 2: 2.0}, "entropy": {}, "topk_accuracy": {2: 0.5}, "layers_to_coherence": 2},
        "jacobian": {"forward_kl": {0: 4.0, 1: 2.0, 2: 1.0}, "entropy": {}, "topk_accuracy": {2: 0.6}, "layers_to_coherence": 1},
    }, "elicitation": None}
    p = render_showdown(results, tmp_path / "showdown.html")
    html = p.read_text()
    assert "<svg" in html and "logit" in html and "jacobian" in html
