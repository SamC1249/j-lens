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
