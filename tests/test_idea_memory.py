"""Idea-level memory ("B3 on ideas"): DeltaMemory(mode="sum"/"delta") and IdeaPredictor(p_window, idea_memory).
Tests only; the code under test is in model/deltanet.py and model/ideajepa.py."""
import pytest
import torch
import torch.nn.functional as F

from ideajepa.model.deltanet import DeltaMemory
from ideajepa.model.ideajepa import IdeaPredictor

torch.backends.cuda.matmul.allow_tf32 = False


def sum_memory_reference(mem, x):
    """Explicit recurrence for mode="sum": S_t = S_{t-1} + b_t v_t k_t^T, o_t = S_t q_t, then the module's
    per-head LayerNorm, gate and output projection."""
    B, T, d = x.shape
    q, k, v = mem.qkv(x).view(B, T, 3, mem.h, mem.dh).permute(2, 0, 3, 1, 4)
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    beta = torch.sigmoid(mem.b(x)).transpose(1, 2)
    S = x.new_zeros(B, mem.h, mem.dh, mem.dh)
    outs = []
    for t in range(T):
        S = S + beta[:, :, t, None, None] * v[:, :, t, :, None] @ k[:, :, t, None, :]
        outs.append((S @ q[:, :, t, :, None])[..., 0])
    o = torch.stack(outs, 2)
    o = mem.ln(o).transpose(1, 2).reshape(B, T, d)
    return mem.o(o * torch.sigmoid(mem.g(x)))


def test_sum_memory_matches_recurrence():
    torch.manual_seed(0)
    mem = DeltaMemory(32, 4, mode="sum")  # the module computes the memory in float32 internally
    x = torch.randn(2, 11, 32)
    with torch.no_grad():
        assert torch.allclose(mem(x), sum_memory_reference(mem, x), atol=1e-5)


@pytest.mark.parametrize("mode", ["sum", "delta"])
def test_memory_causal(mode):
    torch.manual_seed(1)
    mem = DeltaMemory(32, 4, mode=mode).eval()
    x = torch.randn(2, 20, 32)
    x2 = x.clone()
    x2[:, 12] += 3.0
    o1, o2 = mem(x), mem(x2)
    assert torch.allclose(o1[:, :12], o2[:, :12], atol=1e-5)
    assert not torch.allclose(o1[:, 12:], o2[:, 12:], atol=1e-5)


def predictor(p_window=0, idea_memory="none", layers=4, seed=0):
    torch.manual_seed(seed)
    cfg = dict(d=32, n_heads=4, r=2, max_ideas=24, pred_layers=layers, p_window=p_window, idea_memory=idea_memory)
    return IdeaPredictor(cfg).eval()


def slots(n_ideas, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = F.normalize(torch.randn(n_ideas, 2, 32, generator=g), dim=-1)
    v = torch.randn(n_ideas, 2, 32, generator=g)
    return k, v, torch.arange(n_ideas)[None]  # one document, ideas 0..n-1


@pytest.mark.parametrize("kw", [dict(), dict(p_window=2), dict(p_window=2, idea_memory="sum"),
                                dict(p_window=2, idea_memory="delta")])
def test_predictor_causal(kw):
    """Changing idea j never changes the predictions for ideas <= j (prediction n uses ideas < n)."""
    P = predictor(**kw)
    k, v, di = slots(12)
    pk1, pv1 = P(k, v, di)
    j = 6
    v2 = v.clone()
    v2[j] += 2.0
    pk2, pv2 = P(k, v2, di)
    assert torch.allclose(pv1[:, :j + 1], pv2[:, :j + 1], atol=1e-5)
    assert not torch.allclose(pv1[:, j + 1:], pv2[:, j + 1:], atol=1e-5)


def receptive_ideas(w, layers):
    """Ideas visible to one prediction through `layers` stacked windows of `w` idea positions."""
    return layers * (w - 1) + 1


@pytest.mark.parametrize("w,layers", [(2, 4), (3, 2)])
def test_window_without_memory_forgets_old_ideas(w, layers):
    """Without memory, idea 0 cannot influence a prediction more than layers*(w-1)+1 ideas later.
    NB: with p_window=2 and 4 layers the effective window is 5 ideas, not 2."""
    P = predictor(p_window=w, layers=layers)
    R = receptive_ideas(w, layers)
    n = R + 4
    k, v, di = slots(n)
    v2 = v.clone()
    v2[0] += 5.0
    pv1, pv2 = P(k, v, di)[1], P(k, v2, di)[1]
    # prediction for idea m sits at position m and sees positions m-R+1..m, i.e. ideas m-R..m-1
    assert torch.allclose(pv1[0, R + 1:], pv2[0, R + 1:], atol=1e-5)      # idea 0 out of reach
    assert not torch.allclose(pv1[0, R], pv2[0, R], atol=1e-5)            # last prediction that still sees it


@pytest.mark.parametrize("mode", ["sum", "delta"])
def test_memory_reaches_old_ideas(mode):
    """With the idea memory, idea 0 does influence the last prediction, beyond the attention window."""
    P = predictor(p_window=2, idea_memory=mode)
    n = receptive_ideas(2, 4) + 6
    k, v, di = slots(n)
    v2 = v.clone()
    v2[0] += 5.0
    assert not torch.allclose(P(k, v, di)[1][0, -1], P(k, v2, di)[1][0, -1], atol=1e-5)


def test_padding_documents_do_not_leak():
    """A second, shorter document in the batch (padded with -1) does not change the first one."""
    P = predictor(p_window=2, idea_memory="delta")
    k, v, _ = slots(10)
    di_one = torch.arange(6)[None]
    di_two = torch.stack([torch.arange(6), torch.tensor([6, 7, 8, -1, -1, -1])])
    a = P(k, v, di_one)[1][0]
    b = P(k, v, di_two)[1][0]
    assert torch.allclose(a, b, atol=1e-5)


def test_mask_predictor_with_small_slots():
    """IdeaMaskPredictor must work when the slot dimension differs from the model dimension."""
    from ideajepa.model.ideajepa import IdeaMaskPredictor
    torch.manual_seed(0)
    cfg = dict(d=32, n_heads=4, r=2, max_ideas=24, pred_layers=2, d_slot=8)
    PM = IdeaMaskPredictor(cfg).eval()
    k = F.normalize(torch.randn(7, 2, 8), dim=-1)
    v = torch.randn(7, 2, 8)
    di = torch.stack([torch.arange(4), torch.tensor([4, 5, 6, -1])])
    masked = torch.zeros(7, dtype=torch.bool)
    masked[[1, 5]] = True
    pk, pv = PM(k, v, di, masked)
    assert pk.shape == (7, 2, 8) and pv.shape == (7, 2, 8)
    assert torch.allclose(pk.norm(dim=-1), torch.ones(7, 2), atol=1e-5)


def test_role_embedding_is_causal_and_used():
    """p_role_emb: the role of idea j changes only the predictions after j; off by default."""
    torch.manual_seed(0)
    cfg = dict(d=32, n_heads=4, r=2, max_ideas=24, pred_layers=2, p_window=2, idea_memory="delta", p_role_emb=True)
    P = IdeaPredictor(cfg).eval()
    k, v, di = slots(10)
    role = (torch.arange(10) >= 6).long()
    _, pv1 = P(k, v, di, role=role)
    role2 = role.clone(); role2[4] = 1
    _, pv2 = P(k, v, di, role=role2)
    assert torch.allclose(pv1[:, :5], pv2[:, :5], atol=1e-5)
    assert not torch.allclose(pv1[:, 5:], pv2[:, 5:], atol=1e-5)
    with pytest.raises(AssertionError):
        P(k, v, di)
    assert predictor().role_emb is None


def test_role_configs_load():
    from ideajepa.train import load_config
    cfg = load_config("configs/v3_stepB_w1_role.yaml")
    assert cfg["model"]["p_role_emb"] is True and cfg["loss"]["w_nce"] == 0.1
    assert cfg["data"].get("surfaces") and cfg["model"].get("idea_memory") == "delta"
