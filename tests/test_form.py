"""'Separate form' variant: surface template index given to the decoder (not carried by the idea)."""
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
    return d["vocab"], d["docs"][:300]


def test_template_roundtrip(data):
    vocab, docs = data
    n = 0
    for d in docs:
        for i in d["ideas"]:
            t = R.template_id(i)
            assert 0 <= t < R.N_TEMPLATES
            if i["step"] == "deduction" and not i["restated"]:
                assert t in R.DEDUCTION_TEMPLATES
                assert " ".join(R.render_with_template(i["fact"], t).w) == " ".join(vocab[x] for x in i["tokens"])
                n += 1
    assert n > 500


def test_collator_form_ids(data):
    _, docs = data
    b = Collator(m_prefix=3)(docs[:5])
    flat = [i for d in docs[:5] for i in d["ideas"]]
    assert b["form_id"].tolist() == [R.template_id(i) for i in flat]


def test_decoder_depends_on_template(data):
    vocab, docs = data
    torch.manual_seed(0)
    m = IdeaJEPA(cfg(vocab_size=len(vocab), form_cond=True, max_idea_len=24, max_window=27, max_ideas=24)).eval()
    b = Collator(m_prefix=3)(docs[:3])
    k, v = m.encode(b)
    lg1, _ = m.decode(b, k, v)
    b2 = dict(b); b2["form_id"] = (b["form_id"] + 1) % R.N_TEMPLATES
    lg2, _ = m.decode(b2, k, v)
    assert not torch.allclose(lg1, lg2)
    out, _ = idea_losses(m, b, LOSS, p_gen=0.5)
    out["loss"].backward()
    assert m.dec.form_emb.weight.grad.abs().sum() > 0
    # no form_cond: the form ids are ignored
    torch.manual_seed(0)
    m0 = IdeaJEPA(cfg(vocab_size=len(vocab), max_idea_len=24, max_window=27, max_ideas=24)).eval()
    assert torch.allclose(m0.decode(b, k, v)[0], m0.decode(b2, k, v)[0])
    missing, unexpected = m.load_state_dict(m0.state_dict(), strict=False)
    assert missing == ["dec.form_emb.weight"] and not unexpected
