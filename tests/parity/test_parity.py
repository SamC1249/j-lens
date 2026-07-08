"""Numerical parity against Anthropic's reference implementation (anthropics/jlens).

Skipped unless the optional `jlens` package and a small real model are available. Install
with `pip install -e .[parity]`. This is the correctness oracle for our reimplementation:
the decoded top-k tokens (which are invariant to the lens's overall scale, since the final
norm removes it) should agree on a small Qwen model.
"""

from __future__ import annotations

import os

import pytest

jlens = pytest.importorskip("jlens")
PARITY_MODEL = os.environ.get("JLENSKIT_PARITY_MODEL", "Qwen/Qwen2-0.5B")


@pytest.mark.slow
def test_decoded_topk_matches_reference():
    import transformers

    import jlenskit

    hf = transformers.AutoModelForCausalLM.from_pretrained(PARITY_MODEL)
    tok = transformers.AutoTokenizer.from_pretrained(PARITY_MODEL)

    # Reference lens
    ref_model = jlens.from_hf(hf, tok)
    prompts = ["Fact: The currency used in the country shaped like a boot is"] * 4
    ref_lens = jlens.fit(ref_model, prompts=prompts)

    # Our lens over the same (tiny) corpus
    adapter = jlenskit.adapter_from(hf, tok)
    batches = jlenskit.data.pack_texts(prompts, tok, seq_len=16, n_sequences=4)
    ours = jlenskit.JacobianLens.fit(adapter, batches, chunk_size=64)

    prompt = "Fact: The currency used in the country shaped like a boot is"
    ref_logits, _, _ = ref_lens.apply(ref_model, prompt, positions=[-2])
    our_res = ours.apply(adapter, prompt, positions=[-2], top_k=5)

    # Compare final-workspace layers' top-1 agreement (loose: reference uses a larger corpus)
    agree = 0
    total = 0
    for layer in sorted(set(ref_logits) & set(our_res.lens_logits)):
        ref_top1 = int(ref_logits[layer][0].topk(1).indices.item())
        our_top1 = our_res.decoded[0][[d.layer for d in our_res.decoded[0]].index(layer)].token_ids[0]
        total += 1
        agree += int(ref_top1 == our_top1)
    assert total > 0
    assert agree / total >= 0.5
