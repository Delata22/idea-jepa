"""IdeaPredictor p_id_keys: entity codes in the first quarter of P's attention keys."""
import torch

from ideajepa.model.ideajepa import IdeaJEPA
from test_core import cfg

NAMES, OBJS = [10, 11], [20, 21]


def lm(**kw):
    torch.manual_seed(0)
    c = cfg(vocab_size=40, max_idea_len=12, max_window=15, max_ideas=12, p_window=0, idea_memory="none",
            p_id_keys={"names": ["a", "b"], "objects": ["x", "y"]}, p_id_name_ids=NAMES, p_id_obj_ids=OBJS, **kw)
    return IdeaJEPA(c).eval()


def ideas():
    # "a went to L" / "a got x" / "b went to M" / "where is x ?"
    return torch.tensor([[10, 5, 6, 30, 0], [10, 7, 20, 0, 0], [11, 5, 6, 31, 0], [8, 9, 20, 4, 0]]), torch.tensor([4, 3, 4, 4])


def test_codes_first_name_and_object():
    m = lm()
    tok, _ = ideas()
    c = m.P._entity_codes(tok)                                  # [4, H, 2s]
    s = m.P.id_sub
    assert c.shape[-1] == 2 * s
    assert torch.allclose(c[0, :, :s], c[1, :, :s]) and not torch.allclose(c[0, :, :s], c[2, :, :s])   # same name
    assert c[0, :, s:].abs().sum() == 0 and c[3, :, :s].abs().sum() == 0                            # absent -> 0
    assert torch.allclose(c[1, :, s:], c[3, :, s:])                                                  # same object


def test_predict_uses_last_encoded_tokens_and_is_causal():
    m = lm()
    tok, ln = ideas()
    b = {"doc_index": torch.arange(4)[None], "idea_doc": torch.zeros(4, dtype=torch.long), "idea_pos": torch.arange(4)}
    with torch.no_grad():
        k, v = m.E(tok, ln)
        p1 = m.P(k, v, b["doc_index"], tokens=m._p_tokens(k))[1]
        tok2 = tok.clone(); tok2[2, 0] = 10                      # idea 2 about "a" instead of "b": only positions > 2 change
        k2, v2 = m.E(tok2, ln)
        p2 = m.P(k2, v2, b["doc_index"], tokens=m._p_tokens(k2))[1]
    assert torch.allclose(p1[:, :3], p2[:, :3], atol=1e-5) and not torch.allclose(p1[:, 3:], p2[:, 3:], atol=1e-5)
    assert "P.word_emb.weight" not in m.state_dict()


def test_config_resolves_word_ids():
    from ideajepa.train import model_config
    from ideajepa.utils.common import load_config
    import pathlib
    if not pathlib.Path("data/babi/meta.pt").exists():
        return
    m = model_config(load_config("configs/babi_im_full_id.yaml", ["data.path=data/babi/qa2_10k"]))
    assert len(m["p_id_name_ids"]) == 4 and len(m["p_id_obj_ids"]) == 3


def test_bow_keys_mean_of_words():
    torch.manual_seed(0)
    c = cfg(vocab_size=40, max_idea_len=12, max_window=15, max_ideas=12, p_window=0, idea_memory="none", p_bow_keys=True)
    m = IdeaJEPA(c).eval()
    tok, ln = ideas()
    codes = m.P._entity_codes(tok)
    perm = torch.tensor([[6, 5, 10, 30, 0]])                  # same words, other order, same code (bag of words)
    assert torch.allclose(m.P._entity_codes(perm), codes[:1], atol=1e-6)
    b = torch.arange(4)[None]
    with torch.no_grad():
        k, v = m.E(tok, ln)
        p1 = m.P(k, v, b, tokens=m._p_tokens(k))[1]
    assert torch.isfinite(p1).all() and "P.word_emb.weight" not in m.state_dict()


def _wordkv_model(mode):
    torch.manual_seed(0)
    c = cfg(vocab_size=40, max_idea_len=12, max_window=15, max_ideas=12, p_window=0, idea_memory="none", p_word_kv=mode)
    return IdeaJEPA(c).eval()


def test_word_kv_causal_and_oracle_selection():
    tok, ln = ideas()
    di = torch.arange(4)[None]
    for mode in ("all", "support"):
        m = _wordkv_model(mode)
        sup = {"idea_support": torch.tensor([True, False, True, False])}
        with torch.no_grad():
            k, v = m.E(tok, ln)
            p1 = m.P(k, v, di, tokens=None, words=m._p_words(k, sup))[1]
            tok2 = tok.clone(); tok2[2, 1] = 9                   # change idea 2: positions <= 2 cannot see it
            k2, v2 = m.E(tok2, ln)
            p2 = m.P(k2, v2, di, words=m._p_words(k2, sup))[1]
        assert torch.allclose(p1[:, :3], p2[:, :3], atol=1e-5) and not torch.allclose(p1[:, 3:], p2[:, 3:], atol=1e-5)
    # oracle: the words of a NON-support idea are invisible (only its slots are): replacing its word states changes nothing
    m = _wordkv_model("support")
    with torch.no_grad():
        k, v = m.E(tok, ln)
        ws, wv, sel = m._p_words(k, {"idea_support": torch.tensor([True, False, True, False])})
        a = m.P(k, v, di, words=(ws, wv, sel))[1]
        ws2 = ws.clone(); ws2[1] += torch.randn_like(ws2[1])   # random (a constant is removed by LayerNorm)
        b = m.P(k, v, di, words=(ws2, wv, sel))[1]
        ws3 = ws.clone(); ws3[0] += torch.randn_like(ws3[0])
        c_ = m.P(k, v, di, words=(ws3, wv, sel))[1]
    assert torch.allclose(a, b, atol=1e-5) and not torch.allclose(a, c_, atol=1e-5)


def test_bow_in_memory_keys_changes_writes_only():
    torch.manual_seed(0)
    c = cfg(vocab_size=40, max_idea_len=12, max_window=15, max_ideas=12, p_window=2, idea_memory="delta",
            p_bow_keys=True, p_bow_where="mem")
    m = IdeaJEPA(c).eval()
    tok, ln = ideas()
    di = torch.arange(4)[None]
    with torch.no_grad():
        k, v = m.E(tok, ln)
        a = m.P(k, v, di, tokens=m._p_tokens(k))[1]
        m.P.bow.weight.mul_(0)                                  # zero codes: must differ from the learned codes
        b = m.P(k, v, di, tokens=m._p_tokens(k))[1]
        tok2 = tok.clone(); tok2[2, 1] = 9
        k2, v2 = m.E(tok2, ln)
        c2 = m.P(k2, v2, di, tokens=m._p_tokens(k2))[1]
    assert not torch.allclose(a, b) and torch.allclose(b[:, :3], c2[:, :3], atol=1e-5)


def test_word_kv_outside_window_with_memory():
    """p_word_kv on rung (1) (window 2 + memory): a supporting idea OUT of the slot window still reaches the prediction
    through its words; a non-supporting one does not (only through the memory, which is disabled here by zeroing it)."""
    torch.manual_seed(0)
    c = cfg(vocab_size=40, max_idea_len=12, max_window=15, max_ideas=12, p_window=2, idea_memory="none", p_word_kv="support")
    m = IdeaJEPA(c).eval()
    tok, ln = ideas()
    di = torch.arange(4)[None]
    with torch.no_grad():
        k, v = m.E(tok, ln)
        ws, wv, sel = m._p_words(k, {"idea_support": torch.tensor([True, False, False, False])})
        a = m.P(k, v, di, words=(ws, wv, sel))[1]
        ws2 = ws.clone(); ws2[0] += torch.randn_like(ws2[0])
        b = m.P(k, v, di, words=(ws2, wv, sel))[1]
    # position 3 predicts idea 3 from input positions 2-3 (ideas 1-2) in the window: idea 0 is out of it, seen by its words
    assert not torch.allclose(a[:, 3], b[:, 3], atol=1e-5)
