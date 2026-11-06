"""Stage A: role forms / fact surfaces in the Collator, multi-form token loss, cross-encoding, word dropout."""
from pathlib import Path

import pytest
import torch

from ideajepa.data import reasoning as R
from ideajepa.data.dataset import Collator
from ideajepa.model.ideajepa import IdeaJEPA
from ideajepa.model.objective import idea_losses
from test_core import LOSS, cfg

VAL = Path("data/reasoning/val.pt")
pytestmark = pytest.mark.skipif(not VAL.exists(), reason="needs data/reasoning")


@pytest.fixture(scope="module")
def data():
    d = torch.load(VAL, weights_only=False)
    vocab = d["vocab"]
    return vocab, {w: i for i, w in enumerate(vocab)}, d["docs"][:40]


def txt(vocab, row):
    return " ".join(vocab[t] for t in row.tolist() if t != 0)


def test_collator_surfaces(data):
    vocab, w2i, docs = data
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0)(docs[:6])
    flat = [i for d in docs[:6] for i in d["ideas"]]
    M = b["M"]
    for r, idea in enumerate(flat):
        f = idea["fact"]
        main = txt(vocab, b["idea_tok"][r])
        assert main == " ".join(R.role_forms(f, idea["step"])[0])
        if idea["step"] == "deduction":
            assert main.startswith("so ")
        enc = txt(vocab, b["enc_tok"][r])
        assert enc in {" ".join(s) for s in R.fact_surfaces(f)}
        assert txt(vocab, b["ref_win"][r, M:]) == " ".join(vocab[t] for t in idea["tokens"])
    ai = b["alt_idea"]
    assert torch.equal(b["alt_win"][:, :M], b["win"][ai, :M])            # alt rows share the main row's prefix
    for a_, r in enumerate(ai.tolist()):
        idea = flat[r]
        assert txt(vocab, b["alt_win"][a_, M:]) == " ".join(R.role_forms(idea["fact"], idea["step"])[1])


def test_dagger_docs_keep_reference(data):
    vocab, w2i, docs = data
    d = {**docs[0], "dagger": True}
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i)([d])
    for r, idea in enumerate(d["ideas"]):
        assert b["idea_tok"][r, :len(idea["tokens"])].tolist() == idea["tokens"]
        assert b["enc_tok"][r, :len(idea["tokens"])].tolist() == idea["tokens"]
    assert "alt_win" not in b


def model_for(vocab, **kw):
    torch.manual_seed(0)
    return IdeaJEPA(cfg(vocab_size=len(vocab), max_idea_len=24, max_window=27, max_ideas=24, **kw))


def test_multi_form_loss_bounds(data):
    vocab, w2i, docs = data
    m = model_for(vocab)
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=1)(docs[:6])
    lc = {**LOSS, "multi_form": True}
    out, _ = idea_losses(m, b, lc, p_gen=0.5, train=False)
    assert torch.isfinite(out["L_tok"]) and out["L_tok"] <= out["ce_min_role"] + 1e-5   # -log sum exp <= min CE
    assert "ce_ref" in out
    m.train()
    out, _ = idea_losses(m, b, lc, p_gen=0.5)
    out["loss"].backward()


def test_word_dropout_train_only_and_targets_intact(data):
    vocab, w2i, docs = data
    m = model_for(vocab, word_dropout=0.5)
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i)(docs[:4])
    win0 = b["win"].clone()
    k, v = m.encode(b)
    m.eval()
    e1, _ = m.decode(b, k, v)
    e2, _ = m.decode(b, k, v)
    assert torch.allclose(e1, e2)                                        # no dropout in eval
    m.train()
    torch.manual_seed(3)
    t1, _ = m.decode(b, k, v)
    assert not torch.allclose(t1, e1)                                    # dropout active in training
    assert torch.equal(b["win"], win0)                                   # targets never modified


def test_cross_encoding_changes_recon_z(data):
    vocab, w2i, docs = data
    m = model_for(vocab)
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=5)(docs[:6])
    torch.manual_seed(0); o1, a1 = idea_losses(m, b, {**LOSS, "cross_encode": True}, force_path="recon")
    torch.manual_seed(0); o2, a2 = idea_losses(m, b, {**LOSS, "cross_encode": False}, force_path="recon")
    assert not torch.allclose(a1["logits"], a2["logits"])
    torch.manual_seed(0); o3, a3 = idea_losses(m, b, {**LOSS, "cross_encode": True}, force_path="recon", train=False)
    torch.manual_seed(0); o4, a4 = idea_losses(m, b, {**LOSS, "cross_encode": False}, force_path="recon", train=False)
    assert torch.allclose(a3["logits"], a4["logits"])                    # eval: E reads the main row


def test_stepB_positives_and_keys(data):
    vocab, w2i, docs = data
    f1 = [d for d in docs if d["family"] == "F1"][:3]
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=2)(f1)
    flat = [i for d in f1 for i in d["ideas"]]
    pos_of = {int(i): r for r, i in enumerate(b["pos_idea"])}
    n_goes = 0
    for r, idea in enumerate(flat):
        f = idea["fact"]
        if f[0] == "goes":
            assert txt(vocab, b["pos_tok"][pos_of[r]]) == f"{f[1]} is in the {f[2]} ."   # implied position
            n_goes += 1
        if r in pos_of and f[0] != "goes":
            assert txt(vocab, b["pos_tok"][pos_of[r]]) != txt(vocab, b["idea_tok"][r])   # another surface
    assert n_goes > 0
    # same fact tuple -> same key (e.g. an F1 initial location and a final location that never moved)
    keys = {}
    for r, idea in enumerate(flat):
        keys.setdefault(tuple(idea["fact"]), set()).add(int(b["fact_key"][r]))
    assert all(len(v_) == 1 for v_ in keys.values())


def test_stepB_info_nce(data):
    from ideajepa.model.objective import _info_nce, nce_embed
    vocab, w2i, docs = data
    m = model_for(vocab)
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=3)(docs[:6])
    k, v = m.encode(b)
    loss = _info_nce(m, b, k, v, 0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.E.out_v.weight.grad.abs().sum() > 0
    out, _ = idea_losses(m, b, {**LOSS, "w_nce": 0.1}, p_gen=0.5)
    assert out["L_nce"] > 0


def test_stepB_false_negatives_masked():
    """Two pairs that encode the SAME fact must not be negatives of each other."""
    from ideajepa.model.objective import _info_nce
    torch.manual_seed(0)
    k = torch.nn.functional.normalize(torch.randn(3, 2, 4), dim=-1)
    v = torch.randn(3, 2, 4)
    k[1], v[1] = k[0], v[0]                                  # ideas 0 and 1: identical embeddings (same fact)

    class Stub:
        class E_:
            def __call__(self, tok, lens):
                return k[tok[:, 0]], v[tok[:, 0]]           # positive of idea i = same embedding as idea i
        E = E_()

    b = {"pos_idea": torch.arange(3), "pos_tok": torch.arange(3)[:, None], "pos_len": torch.ones(3, dtype=torch.long)}
    b_same = {**b, "fact_key": torch.tensor([7, 7, 9]), "pos_key": torch.tensor([7, 7, 9])}
    b_diff = {**b, "fact_key": torch.tensor([7, 8, 9]), "pos_key": torch.tensor([7, 8, 9])}
    l_same = _info_nce(Stub(), b_same, k, v, 0.1)
    l_diff = _info_nce(Stub(), b_diff, k, v, 0.1)
    assert l_same < l_diff - 0.1                              # masked: the twin no longer competes with the positive



def test_p_input_surface_random_and_mix(data):
    """p_input_surface: P's input rows are story surfaces of the idea's role (deduction templates / premise forms)."""
    vocab, w2i, docs = data
    docs = docs[:6]
    flat = [i for d in docs for i in d["ideas"]]
    for mode in ("random", "mix"):
        b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0, p_input_surface=mode)(docs)
        assert b["pin_tok"].shape[0] == b["idea_tok"].shape[0] == len(flat)
        n_other = 0
        for r, idea in enumerate(flat):
            f = tuple(tuple(x) if isinstance(x, list) else x for x in idea["fact"])
            got = txt(vocab, b["pin_tok"][r])
            ok = ([R.render_with_template(f, t).w for t in R.DEDUCTION_TEMPLATES] if idea["step"] == "deduction"
                  else R.premise_forms(f))
            assert got in {" ".join(x) for x in ok}
            n_other += got != txt(vocab, b["idea_tok"][r])
        assert n_other > 0
    assert "pin_tok" not in Collator(m_prefix=3, surfaces=True, w2i=w2i)(docs)


def test_stepC_transcr_gradient_in_P_only(data):
    """L_transcr: finite, positive; with transcr_detach_E the gradient reaches P and not E; no gradient through decoding."""
    vocab, w2i, docs = data
    model = model_for(vocab, stop_head=True).train()
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0)(docs[:4])
    lc = dict(LOSS, multi_form=True, w_transcr=1.0, transcr_frac=0.5, transcr_max_len=12)
    from ideajepa.model.objective import _transcr_loss
    k, v = model.encode(b)
    pk, pv = model.predict(k, v, b)
    L, stats = _transcr_loss(model, b, lc, k, v, pk, pv)
    assert torch.isfinite(L) and float(L) > 0 and stats["transcr_rows"] >= 1
    model.zero_grad()
    L.backward()
    gP = sum(p.grad.abs().sum() for p in model.P.parameters() if p.grad is not None)
    gE = sum(p.grad.abs().sum() for p in model.E.parameters() if p.grad is not None)
    gD = sum(p.grad.abs().sum() for p in model.dec.parameters() if p.grad is not None)
    assert float(gP) > 0 and float(gE) == 0 and float(gD) == 0
    out, _ = idea_losses(model, b, lc)
    assert "L_transcr" in out and torch.isfinite(out["loss"])


def test_since_antecedent_matches_generator(data):
    """The antecedent recomputed with the generator's rule equals the one written in restated sentences."""
    from ideajepa.data import verifier as V
    vocab, w2i, _ = data
    docs = torch.load(VAL, weights_only=False)["docs"][:300]
    c = Collator(m_prefix=3, surfaces=True, w2i=w2i, since_surfaces="nce")
    n = same = 0
    for d in docs:
        facts = [tuple(tuple(x) if isinstance(x, list) else x for x in i["fact"]) for i in d["ideas"]]
        antes = c._antecedents(d)
        for ii, idea in enumerate(d["ideas"]):
            if idea.get("restated"):
                parsed = V.parse_sentence(txt(vocab, torch.tensor(idea["tokens"])))[1]
                rule = R.antecedent(d["family"], facts[ii], facts[:ii])
                assert rule in facts[:ii] and parsed in facts[:ii]      # both are facts already stated
                same += rule == parsed
                assert antes[ii] == (facts[ii], parsed)                 # restated sentences: the written antecedent
                assert " ".join(vocab[t] for t in idea["tokens"]) in {" ".join(s) for s in R.since_forms(facts[ii], parsed)}
                n += 1
        assert all(d["ideas"][i]["step"] == "deduction" and antes[i][1] in facts[:i] for i in antes)
    # the data predate the deterministic antecedent rule: ~5 % of restated sentences carry another valid antecedent
    assert n > 20 and same / n > 0.9


def test_since_positives_and_keys(data):
    """since_surfaces: "since A , B" joins B's positives (with A as pos_key2), not the cross-encoding pool ("nce");
    off by default (recipe B unchanged)."""
    vocab, w2i, _ = data
    docs = [d for d in torch.load(VAL, weights_only=False)["docs"][:200] if d["family"] in ("F2", "F3")][:12]
    assert "pos_key2" not in Collator(m_prefix=3, surfaces=True, w2i=w2i)(docs)
    seen_since = 0
    for seed in range(6):
        b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=seed, since_surfaces="nce")(docs)
        pk2 = b["pos_key2"]
        for r_, i in enumerate(b["pos_idea"].tolist()):
            s = txt(vocab, b["pos_tok"][r_])
            assert s.startswith("since") == bool(pk2[i] >= 0)
            seen_since += s.startswith("since")
        for r in range(b["enc_tok"].shape[0]):
            assert not txt(vocab, b["enc_tok"][r]).startswith("since")
    assert seen_since > 0
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0, since_surfaces="all")(docs)
    assert b["enc_tok"].shape[0] == b["idea_tok"].shape[0]


def test_since_positive_not_negative_of_antecedent():
    """A "since A , B" positive must not be a negative of A's anchor: with pos_key2 = A the loss drops."""
    from ideajepa.model.objective import _info_nce
    torch.manual_seed(0)
    k = torch.nn.functional.normalize(torch.randn(3, 2, 4), dim=-1)
    v = torch.randn(3, 2, 4)
    kp, vp = k.clone(), v.clone()
    kp[0], vp[0] = k[1], v[1]                  # positive of idea 0 ("since A , B") looks like idea 1 (fact A)

    class Stub:
        class E_:
            def __call__(self, tok, lens):
                return kp[tok[:, 0]], vp[tok[:, 0]]
        E = E_()

    b = {"pos_idea": torch.arange(3), "pos_tok": torch.arange(3)[:, None], "pos_len": torch.ones(3, dtype=torch.long),
         "fact_key": torch.tensor([7, 8, 9]), "pos_key": torch.tensor([7, 8, 9])}
    l_plain = _info_nce(Stub(), b, k, v, 0.1)
    l_since = _info_nce(Stub(), {**b, "pos_key2": torch.tensor([8, -1, -1])}, k, v, 0.1)
    l_none = _info_nce(Stub(), {**b, "pos_key2": torch.tensor([-1, -1, -1])}, k, v, 0.1)
    assert l_since < l_plain - 0.1
    assert torch.allclose(l_none, l_plain)     # no since positive: identical to recipe B


def test_stepC_start_step(data):
    """transcr_start_step: no L_transcr before it, active after."""
    vocab, w2i, docs = data
    model = model_for(vocab, stop_head=True).train()
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0)(docs[:3])
    lc = dict(LOSS, multi_form=True, w_transcr=1.0, transcr_frac=0.5, transcr_max_len=8, transcr_start_step=10)
    assert float(idea_losses(model, b, lc, step=5)[0]["L_transcr"]) == 0.0
    assert float(idea_losses(model, b, lc, step=10)[0]["L_transcr"]) > 0.0


def test_memsep_rows_aligned_and_loss(data):
    """w_memsep: the rows written for idea p are the memory keys/values at P's input position p + 1; the term is
    positive on a random model, trains P's memory projections and not the decoder; template keys = (predicate, role)."""
    vocab, w2i, docs = data
    model = model_for(vocab, stop_head=True, p_window=2, idea_memory="delta").train()
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0)(docs[:6])
    flat = [i for d in docs[:6] for i in d["ideas"]]
    tk = b["tmpl_key"].tolist()
    for i in range(len(flat)):
        for j in range(len(flat)):
            same = (flat[i]["fact"][0], flat[i]["step"]) == (flat[j]["fact"][0], flat[j]["step"])
            assert (tk[i] == tk[j]) == same
    lc = dict(LOSS, multi_form=True, w_memsep=1.0, memsep_tau=0.0)
    out, aux = idea_losses(model, b, lc)
    rows = model.P.mem_rows
    m0 = model.P.mems[0]
    k0, v0 = m0.kept
    B, H, T, dh = k0.shape
    r = model.cfg["r"]
    di = b["doc_index"]
    bi, p = 1, 2                                   # idea at position 2 of document 1 -> written at position 3
    row = int(di[bi, p])
    ref = k0[bi].reshape(H, T // r, r, dh)[:, p + 1].reshape(-1)
    assert torch.allclose(rows["k"][row, 0], ref) and bool(rows["valid"][row])
    assert float(out["L_memsep"]) > 0 and aux["stats"]["memsep_pairs"] > 0
    model.zero_grad()
    out["L_memsep"].requires_grad_(False)
    from ideajepa.model.objective import _memsep_loss
    L, _ = _memsep_loss(rows, b, lc)
    L.backward()
    assert m0.qkv.weight.grad is not None and float(m0.qkv.weight.grad.abs().sum()) > 0
    assert all(p_.grad is None or float(p_.grad.abs().sum()) == 0 for p_ in model.dec.blocks.parameters())  # emb is shared with E
    out_off, _ = idea_losses(model, b, dict(LOSS, multi_form=True))
    assert float(out_off["L_memsep"]) == 0.0


def test_transcr_guard_oracle(data, monkeypatch):
    """transcr_guard oracle: only rows whose decoded sentence is a valid next deduction are pulled."""
    import ideajepa.model.objective as O
    vocab, w2i, docs = data
    model = model_for(vocab, stop_head=True).train()
    model.vocab = vocab
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0)(docs[:4])
    lc = dict(LOSS, multi_form=True, w_transcr=1.0, transcr_frac=1.0, transcr_max_len=12, transcr_guard="oracle")
    k, v = model.encode(b)
    pk, pv = model.predict(k, v, b)
    L, st = O._transcr_loss(model, b, lc, k, v, pk, pv)            # random decoder: garbage, nothing kept
    assert st["transcr_guard_kept"] == 0.0 and float(L) == 0.0

    def ref_decode(model_, win, M, K, V, max_len=24):               # decode = the reference (valid) sentence
        lens = (win[:, M:] != 0).sum(1)
        return win[:, M:M + int(lens.max())], lens
    monkeypatch.setattr(O, "greedy_decode_rows", ref_decode)
    L, st = O._transcr_loss(model, b, lc, k, v, pk, pv)
    assert st["transcr_guard_kept"] > 0.9 and float(L) > 0


def test_qoracle_pointer_and_loss(data):
    """w_qoracle: the pointer of an F1 deduction is the premise stating its object's position; the
    score of idea p for the query block n uses the key block p + 1; the term trains P's memory query, is 0 when off,
    and a few steps on it raise its own accuracy."""
    vocab, w2i, docs = data
    from ideajepa.data.dataset import qoracle_ptr
    f1 = [d for d in docs if d["family"] == "F1"][:4] + [d for d in docs if d["family"] != "F1"][:2]
    for d in f1:
        for j, (i, p) in enumerate(zip(d["ideas"], qoracle_ptr(d["ideas"]))):
            if d["family"] == "F1" and i["step"] == "deduction":
                pf = d["ideas"][p]["fact"]
                assert 0 <= p < j and d["ideas"][p]["step"] == "premise" and pf[0] == "objin" and pf[1] == i["fact"][1]
            else:
                assert p == -1
    model = model_for(vocab, stop_head=True, p_window=2, idea_memory="delta").train()
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0)(f1)
    lc = dict(LOSS, multi_form=True, w_qoracle=1.0)
    out, aux = idea_losses(model, b, lc)
    n_t = sum(p >= 0 for d in f1 for p in qoracle_ptr(d["ideas"]))
    assert float(out["L_qoracle"]) > 0 and aux["stats"]["qoracle_n"] == n_t
    assert all(getattr(m, "kept_qk", None) is None and not m.keep_q for m in model.P.mems)
    out_off, _ = idea_losses(model, b, dict(LOSS, multi_form=True))
    assert float(out_off["L_qoracle"]) == 0.0
    # direct check of the score against the diagnostic's definition, one (document, n, p)
    for m in model.P.mems:
        m.keep_q = True
    from ideajepa.model.objective import _qoracle_loss
    model.predict(*model.encode(b), b)
    q, k = model.P.mems[0].kept_qk
    B, H, T, dh = q.shape
    r = model.cfg["r"]
    qb, kb = q.reshape(B, H, T // r, r, dh), k.reshape(B, H, T // r, r, dh)
    ref = float(torch.einsum("hid,hjd->", qb[0, :, 16], kb[0, :, 6]) / (H * r * r))   # query block 16 vs idea 5
    sim = float(torch.einsum("hd,hd->", qb[0, :, 16].mean(1), kb[0, :, 6].mean(1)) / H)
    assert abs(ref - sim) < 1e-5
    # training on the term alone raises its accuracy (the memory query CAN be steered)
    opt = torch.optim.Adam(model.P.parameters(), lr=3e-3)
    acc0 = None
    for it in range(40):
        for m in model.P.mems:
            m.keep_q = True
        model.predict(*model.encode(b), b)
        L, st_ = _qoracle_loss(model.P, b, lc)
        acc0 = st_["qoracle_acc"] if acc0 is None else acc0
        opt.zero_grad(); L.backward(); opt.step()
    assert st_["qoracle_acc"] > acc0 + 0.2, (acc0, st_)


def _grad_E(model, L):
    """Gradient norm on E's own parameters (the word embedding is shared with the decoder: excluded)."""
    model.zero_grad()
    L.backward()
    shared = {id(p) for p in model.dec.parameters()}
    return float(sum(p.grad.abs().sum() for p in model.E.parameters() if p.grad is not None and id(p) not in shared))


@pytest.mark.parametrize("psurf", [False, True])
def test_e_warmup_shields_E_from_prediction(data, psurf):
    """loss.e_warmup_steps: before N, L_jepa (and the P-decoded rows) send no gradient into E, InfoNCE is off; after N
    both reach E again."""
    vocab, w2i, docs = data
    model = model_for(vocab, stop_head=True).train()
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0, p_input_surface="random" if psurf else None,
                 nce_positives="dropout")(docs[:4])
    # every term that reaches E directly is off: E's gradient can only come through P's input (or InfoNCE)
    lc = dict(LOSS, multi_form=True, e_warmup_steps=10, w_nce=0.3, w_vic=0.0, w_sparse=0.0, w_center=0.0,
              w_contrib=0.0)
    out, _ = idea_losses(model, b, lc, step=3, force_path="gen")
    assert _grad_E(model, out["loss"]) == 0.0 and float(out["L_nce"]) == 0.0
    out, _ = idea_losses(model, b, dict(lc, w_nce=0.0), step=10, force_path="gen")
    assert _grad_E(model, out["loss"]) > 0.0
    out, _ = idea_losses(model, b, lc, step=10, force_path="gen")
    assert float(out["L_nce"]) > 0.0


def test_dec_noise_train_only(data):
    """loss.dec_noise: the decoder input is noised in training only (token loss changes), not at evaluation."""
    vocab, w2i, docs = data
    model = model_for(vocab, stop_head=True)
    b = Collator(m_prefix=3, surfaces=True, w2i=w2i, seed=0)(docs[:4])
    lc0, lc3 = dict(LOSS, multi_form=True), dict(LOSS, multi_form=True, dec_noise=3.0)
    model.eval()
    with torch.no_grad():
        e0 = idea_losses(model, b, lc0, train=False, force_path="recon")[0]["L_tok"]
        e3 = idea_losses(model, b, lc3, train=False, force_path="recon")[0]["L_tok"]
        t0 = idea_losses(model, b, lc0, train=True, force_path="recon")[0]["L_tok"]
        t3 = idea_losses(model, b, lc3, train=True, force_path="recon")[0]["L_tok"]
    assert torch.allclose(e0, e3) and not torch.allclose(t0, t3)
