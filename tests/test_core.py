"""Mandatory unit tests (spec section 6, stage 0)."""
import copy

import pytest
import torch

from ideajepa.data.dataset import Collator
from ideajepa.model.baseline import GPT
from ideajepa.model.ideajepa import IdeaJEPA, erase_step, slot_norm
from ideajepa.model.objective import idea_losses

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V = 50


def cfg(**kw):
    c = dict(model="ideajepa", d=64, n_heads=4, r=4, vocab_size=V, max_idea_len=12, m_prefix=3,
             enc_layers=1, pred_layers=2, dec_layers=3, read_layers=2, out_window=4, gate=True,
             erase="local", e_mode="context", read_mode="softmax", max_ideas=8, max_window=15,
             max_seq=64, arch="parallel", pre_read=True)
    c.update(kw)
    return c


LOSS = dict(w_jepa=1.0, w_vic=1.0, w_exh=1.0, w_contrib=0.0, w_sparse=0.01, ema_tau=0.996, tau=0.1)


def rand_docs(n_docs=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    docs = []
    for _ in range(n_docs):
        n = int(torch.randint(2, 5, (1,), generator=g))
        docs.append({"ideas": [{"tokens": torch.randint(4, V, (int(torch.randint(3, 12, (1,), generator=g)),),
                                                        generator=g).tolist()} for _ in range(n)]})
    return docs


def batch_of(docs, **ck):
    b = Collator(**ck)(docs)
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}


def make(seed=0, **kw):
    torch.manual_seed(seed)
    return IdeaJEPA(cfg(**kw)).to(DEV).eval()


ARCHS = [dict(arch="parallel"), dict(arch="deep")]


# ---------------------------------------------------------------- shapes
@pytest.mark.parametrize("kw", ARCHS)
def test_shapes(kw):
    m = make(**kw)
    b = batch_of(rand_docs())
    NI, L = b["idea_tok"].shape
    k, v = m.encode(b)
    assert k.shape == (NI, 4, 64) and v.shape == (NI, 4, 64)
    assert torch.allclose(k.norm(dim=-1), torch.ones(NI, 4, device=DEV), atol=1e-4)
    pk, pv = m.predict(k, v, b)
    assert pk.shape == (NI, 4, 64) and pv.shape == (NI, 4, 64)
    e = m.O(b["win"], b["M"])
    assert e.shape == (NI, L, 64)
    assert torch.allclose(e.norm(dim=-1), torch.ones(NI, L, device=DEV), atol=1e-4)
    logits, tr = m.decode(b, k, v)
    assert logits.shape == (NI, L, V)
    assert tr["rho"].shape == (NI, L + 1) and tr["g"].shape == (NI, L)
    out, _ = idea_losses(m, b, LOSS, p_gen=0.5)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


# ---------------------------------------------------------------- erasure property
def test_erasure_removes_projection():
    torch.manual_seed(0)
    Vr = torch.randn(5, 4, 32, dtype=torch.float64)
    e = torch.nn.functional.normalize(torch.randn(5, 32, dtype=torch.float64), dim=-1)
    out = erase_step(Vr, torch.ones(5, 4, dtype=torch.float64), torch.ones(5, dtype=torch.float64), e)
    assert torch.einsum("nrd,nd->nr", out, e).abs().max() < 1e-10
    # orthogonal complement untouched
    perp = Vr - torch.einsum("nrd,nd->nr", Vr, e)[..., None] * e[:, None]
    assert torch.allclose(out, perp)


def test_rank_h_erasure_removes_subspace():
    torch.manual_seed(0)
    Vr = torch.randn(5, 4, 32, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(5, 32, 3, dtype=torch.float64))
    E = q.transpose(-1, -2)
    out = erase_step(Vr, torch.ones(5, 4, dtype=torch.float64), torch.ones(5, dtype=torch.float64), E)
    assert torch.einsum("nrd,nhd->nrh", out, E).abs().max() < 1e-10
    assert (out.norm(dim=-1) <= Vr.norm(dim=-1) + 1e-9).all()


def test_erasure_never_overshoots():
    torch.manual_seed(1)
    for _ in range(50):
        Vr = torch.randn(7, 4, 16)
        e = torch.nn.functional.normalize(torch.randn(7, 16), dim=-1)
        a = torch.softmax(torch.randn(7, 4) * 3, -1)
        g = torch.rand(7)
        out = erase_step(Vr, a, g, e)
        assert (out.norm(dim=-1) <= Vr.norm(dim=-1) + 1e-5).all()
        # remaining projection keeps its sign (no overshoot past zero)
        p0 = torch.einsum("nrd,nd->nr", Vr, e)
        p1 = torch.einsum("nrd,nd->nr", out, e)
        assert (p0 * p1 >= -1e-6).all() and (p1.abs() <= p0.abs() + 1e-6).all()


@pytest.mark.parametrize("kw", ARCHS + [dict(erase="global"), dict(e_mode="static"), dict(read_mode="linear"),
                                dict(erase_rank=3), dict(erase="free"), dict(erase_norm="max"), dict(stop_head=True), dict(e_mode="frozen"), dict(local_window=2), dict(d_slot=16), dict(d_slot=16, arch="deep"), dict(z_norm="idea")])
def test_rho_nonincreasing_and_no_norm_growth(kw):
    m = make(**kw)
    b = batch_of(rand_docs(4, seed=3))
    k, v = m.encode(b)
    v = v * 3 + torch.randn_like(v)  # arbitrary content
    _, tr = m.decode(b, k, v)
    rho = tr["rho"]
    assert (rho[:, 1:] <= rho[:, :-1] + 1e-5).all()
    assert (rho <= 1 + 1e-5).all()
    assert (tr["V_end"].norm(dim=-1) <= v.norm(dim=-1) + 1e-4).all()


def test_slot_norm_rho_equals_frobenius_for_orthonormal_keys():
    from ideajepa.model.ideajepa import frob_norm
    q, _ = torch.linalg.qr(torch.randn(16, 4))
    K = q.T[None]
    Vv = torch.randn(1, 4, 16)
    assert torch.allclose(slot_norm(Vv), frob_norm(K, Vv), atol=1e-5)


# ---------------------------------------------------------------- causality
@pytest.mark.parametrize("kw", ARCHS)
def test_decoder_causal_within_idea(kw):
    m = make(**kw)
    b = batch_of(rand_docs(4, seed=5))
    k, v = m.encode(b)
    lg1, tr1 = m.decode(b, k, v)
    b2 = dict(b)
    win = b["win"].clone()
    t = 5  # change idea token t: logits at steps <= t must not change (step s predicts token s)
    win[:, b["M"] + t] = (win[:, b["M"] + t] + 7) % (V - 4) + 4
    b2["win"] = win
    lg2, tr2 = m.decode(b2, k, v)
    assert torch.allclose(lg1[:, :t + 1], lg2[:, :t + 1], atol=1e-4)
    assert torch.allclose(tr1["rho"][:, :t + 1], tr2["rho"][:, :t + 1], atol=1e-5)
    assert not torch.allclose(lg1[:, t + 1:], lg2[:, t + 1:], atol=1e-4)


@pytest.mark.parametrize("kw", ARCHS)
def test_decoder_sees_nothing_outside_window(kw):
    """Given the same Z, tokens of the previous idea beyond the last m=3 cannot influence the decoder."""
    m = make(**kw)
    docs = rand_docs(3, seed=7)
    for d in docs:  # make the first idea long
        d["ideas"][0]["tokens"] = list(range(10, 21))
    b = batch_of(docs)
    k, v = m.encode(b)
    lg1, _ = m.decode(b, k, v)
    docs2 = copy.deepcopy(docs)
    for d in docs2:
        d["ideas"][0]["tokens"][:8] = [30] * 8  # outside the 3-token prefix of idea 2
    b2 = batch_of(docs2)
    lg2, _ = m.decode(b2, k, v)
    second = b["idea_pos"] == 1
    assert torch.allclose(lg1[second], lg2[second], atol=1e-5)


def test_full_model_generation_path_causal():
    """Gen path: changing idea j+1 never changes predictions for ideas <= j (P, E, window)."""
    m = make()
    docs = rand_docs(3, seed=11)
    b = batch_of(docs)
    _, a1 = idea_losses(m, b, LOSS, force_path="gen", train=False)
    docs2 = copy.deepcopy(docs)
    for d in docs2:
        d["ideas"][-1]["tokens"] = [(t + 5) % (V - 4) + 4 for t in d["ideas"][-1]["tokens"]]
    b2 = batch_of(docs2)
    _, a2 = idea_losses(m, b2, LOSS, force_path="gen", train=False)
    last = torch.zeros_like(b["idea_pos"], dtype=torch.bool)
    for bi in range(len(docs)):
        last[b["doc_index"][bi, len(docs[bi]["ideas"]) - 1]] = True
    assert torch.allclose(a1["logits"][~last], a2["logits"][~last], atol=1e-4)


@pytest.mark.parametrize("kind", ["b0", "b1"])
def test_baselines_causal(kind):
    torch.manual_seed(0)
    m = GPT(cfg(model=kind, n_layers=2)).to(DEV).eval()
    docs = rand_docs(2, seed=2)
    b = batch_of(docs)
    lg1, _ = m.token_logits(b)
    if kind == "b0":
        seq = b["seq"].clone()
        seq[:, 6] = 40
        out1 = m.run(b["seq"][:, :-1])
        out2 = m.run(seq[:, :-1])
        assert torch.allclose(out1[:, :6], out2[:, :6], atol=1e-5)
    else:
        win = b["win"].clone()
        win[:, b["M"] + 4] = 40
        out1 = m.run(b["win"][:, :-1])
        out2 = m.run(win[:, :-1])
        assert torch.allclose(out1[:, :b["M"] + 4], out2[:, :b["M"] + 4], atol=1e-5)


# ---------------------------------------------------------------- EMA / checkpoint
def test_ema_update():
    m = make()
    ema_before = [p.clone() for p in m.E_ema.parameters()]
    with torch.no_grad():
        for p in m.E.parameters():
            p.add_(torch.randn_like(p))
    online = [p.clone() for p in m.E.parameters()]
    m.ema_update(0.9)
    for pe, p0, po in zip(m.E_ema.parameters(), ema_before, online):
        assert torch.allclose(pe, 0.9 * p0 + 0.1 * po, atol=1e-6)
    assert all(not p.requires_grad for p in m.E_ema.parameters())
    ema_ids = {id(p) for p in m.E_ema.parameters()}
    assert not any(id(p) in ema_ids for p in m.trainable_parameters())


def test_checkpoint_reload_identical(tmp_path):
    m = make()
    b = batch_of(rand_docs())
    out1, a1 = idea_losses(m, b, LOSS, force_path="gen", train=False)
    torch.save({"model": m.state_dict(), "cfg": m.cfg}, tmp_path / "ck.pt")
    ck = torch.load(tmp_path / "ck.pt", weights_only=False)
    m2 = IdeaJEPA(ck["cfg"]).to(DEV).eval()
    m2.load_state_dict(ck["model"])
    out2, a2 = idea_losses(m2, b, LOSS, force_path="gen", train=False)
    assert torch.equal(a1["logits"], a2["logits"])
    assert out1["L_tok"] == out2["L_tok"]


def test_local_window_limits_attention():
    """Stacked sliding windows: receptive field = 1 + n_layers*(w-1) tokens."""
    m = make(local_window=2)  # 3 layers -> receptive field 1 + 3*(2-1) = 4 tokens
    b = batch_of(rand_docs(3, seed=9))
    k, v = m.encode(b)
    with torch.no_grad():
        lg1, _ = m.dec(b["win"], b["idea_len"], b["M"], k, v, m.O(b["win"], b["M"]) * 0 + 1e-3)
        win = b["win"].clone()
        win[:, b["M"]] = 42  # idea token 0 (input of step 1)
        lg2, _ = m.dec(win, b["idea_len"], b["M"], k, v, m.O(b["win"], b["M"]) * 0 + 1e-3)
    # step s sees inputs at window positions > M-1+s-4; token 0 sits at position M -> invisible from s >= 5
    assert torch.allclose(lg1[:, 5:], lg2[:, 5:], atol=1e-4)
    assert not torch.allclose(lg1[:, 1:3], lg2[:, 1:3], atol=1e-4)
