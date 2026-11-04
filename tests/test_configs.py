import glob

import pytest

from ideajepa.utils.common import load_config

NUMERIC = ["train.lr", "train.steps", "train.batch_tokens", "loss.w_jepa", "loss.w_exh", "loss.ema_tau",
           "model.d", "model.r"]


@pytest.mark.parametrize("path", [p for p in glob.glob("configs/*.yaml")])
def test_numeric_hyperparameters_are_numbers(path):
    cfg = load_config(path)
    for key in NUMERIC:
        sec, k = key.split(".")
        if k in cfg.get(sec, {}):
            assert isinstance(cfg[sec][k], (int, float)), f"{path}: {key}={cfg[sec][k]!r}"


def test_override_scientific_notation():
    cfg = load_config("configs/syn_b0.yaml", ["train.lr=5e-4"])
    assert cfg["train"]["lr"] == 5e-4
