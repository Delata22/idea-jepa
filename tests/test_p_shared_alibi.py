"""P depth / position variants: p_shared = T (one block + its memory applied T times, pass embedding, distinct
output block) and p_pos = "alibi" (recency bias instead of the absolute idea-position embedding)."""
import pytest
import torch
import torch.nn.functional as F

from ideajepa.model.ideajepa import IdeaPredictor


def predictor(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(d=32, n_heads=4, r=2, max_ideas=24, pred_layers=4, p_window=0, idea_memory="none")
    cfg.update(kw)
    return IdeaPredictor(cfg).eval()


def slots(n_ideas, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = F.normalize(torch.randn(n_ideas, 2, 32, generator=g), dim=-1)
    v = torch.randn(n_ideas, 2, 32, generator=g)
    return k, v, torch.arange(n_ideas)[None]


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("mem", ["none", "delta"])
def test_shared_params_do_not_grow_with_passes(mem):
    p4, p8 = predictor(p_shared=4, idea_memory=mem), predictor(p_shared=8, idea_memory=mem)
    one = predictor(pred_layers=2, idea_memory=mem)   # 2 distinct blocks = shared + output block
    extra = 4 * 32   # pass embeddings: T x d
    assert n_params(p8) - n_params(p4) == extra
    assert n_params(p4) - n_params(one) == 4 * 32
    assert len(p8.blocks) == 9 and p8.blocks[0] is p8.blocks[7] and p8.blocks[8] is not p8.blocks[0]
    if mem != "none":
        assert p8.mems[0] is p8.mems[7] and p8.mems[8] is not p8.mems[0]


@pytest.mark.parametrize("kw", [dict(p_shared=4), dict(p_shared=4, idea_memory="delta"), dict(p_pos="alibi"),
                                dict(p_pos="alibi", idea_memory="delta", p_window=2), dict(p_shared=3, p_pos="alibi")])
def test_causal(kw):
    P = predictor(**kw)
    k, v, di = slots(10)
    k2, v2 = k.clone(), v.clone()
    v2[6] += 2.0
    with torch.no_grad():
        a, _ = P(k, v, di)
        b, _ = P(k2, v2, di)
    # the output at position n predicts idea n from ideas < n: positions <= 6 cannot see idea 6
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5)
    assert not torch.allclose(a[:, 7:], b[:, 7:], atol=1e-5)


def test_alibi_is_translation_invariant():
    """Without a position embedding, the prediction from the last 3 ideas does not depend on how many ideas precede
    when attention cannot reach them (window 3, one layer)."""
    P = predictor(p_pos="alibi", p_window=3, pred_layers=1)   # one layer: receptive field = the window
    k, v, _ = slots(12)
    with torch.no_grad():
        a, _ = P(k[5:], v[5:], torch.arange(7)[None])
        b, _ = P(k, v, torch.arange(12)[None])
    assert torch.allclose(a[:, -1], b[:, -1], atol=1e-5)


def test_shared_state_dict_roundtrip():
    P = predictor(p_shared=4, idea_memory="delta")
    Q = predictor(seed=1, p_shared=4, idea_memory="delta")
    Q.load_state_dict(P.state_dict())
    k, v, di = slots(8)
    with torch.no_grad():
        assert torch.allclose(P(k, v, di)[1], Q(k, v, di)[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast path")
def test_alibi_autocast_cuda():
    P = predictor(p_pos="alibi", idea_memory="delta").cuda().train()
    k, v, di = slots(9)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pk, pv = P(k.cuda(), v.cuda(), di.cuda())
    (pk.sum() + pv.sum()).backward()
