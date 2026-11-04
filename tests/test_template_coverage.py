"""Spec chantier 3 bis §2: evaluation templates ⊆ training templates, per family / role / model input (scripts/
template_coverage.py). The base recipe passes; the checker finds the two known train/eval shifts (P's input on a
single surface in A/B; restated "since" sentences absent from B's training)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, "scripts")
from template_coverage import views  # noqa: E402
from ideajepa.utils.common import load_config

VAL = Path("data/reasoning/val.pt")
pytestmark = pytest.mark.skipif(not VAL.exists(), reason="needs data/reasoning")


@pytest.fixture(scope="module")
def data():
    d = torch.load(VAL, weights_only=False)
    vocab = d["vocab"]
    return d["docs"][:150], vocab, {w: i for i, w in enumerate(vocab)}


def missing(cfg_path, data):
    docs, vocab, w2i = data
    train, ev = views(load_config(cfg_path), docs, vocab, w2i)
    return {k: {t for t in ev[k] - train.get(k, set()) if not t.endswith("?")} for k in ev if ev[k] - train.get(k, set())}


def test_base_recipe_covers_evaluation(data):
    assert not {k: v for k, v in missing("configs/v2_pwin_delta.yaml", data).items() if v}


def test_detects_known_shifts(data):
    m = missing("configs/v3_stepB_w3.yaml", data)
    assert "then/f0" in m[("F1", "deduction", "P_input")]                      # P reads the main form only
    assert {"since/f0", "since/f1"} <= m[("F2", "deduction", "E_input")]      # "since" never seen in training
    m = missing("configs/v3_stepA_psurf.yaml", data)
    assert ("F1", "deduction", "P_input") not in m or not m[("F1", "deduction", "P_input")]


def test_compound_views_have_no_since(data):
    for cfg in ("configs/v3_stepB_w3_prop.yaml", "configs/v3_stepB_w3_split.yaml"):
        m = missing(cfg, data)
        assert not any(t.startswith("since") for v in m.values() for t in v), (cfg, m)


def test_p_input_covers_story_surfaces(data):
    """Guard (notes §34.14): P's input at training must cover the story surfaces it reads at evaluation. The portable recipe
    (dropout positives + psurf) passes; the same recipe without psurf is flagged (the shift that cost v3sd ~35 points)."""
    def p_missing(cfg):
        m = missing(cfg, data)
        return {k: v for k, v in m.items() if k[2] == "P_input" and v}
    assert not p_missing("configs/v3_stepB_w3_split_dropnce_psurf.yaml")
    assert p_missing("configs/v3_stepB_w3_split_dropnce.yaml")
