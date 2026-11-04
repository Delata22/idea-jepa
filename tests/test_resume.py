"""Checkpoint / resume: worker_iterator replays the same data stream; 40 steps in one go == 20 + resume 20.
Small data (a copy of 700 val documents) and one subprocess per training run, to keep memory low."""
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ideajepa.data.dataset import worker_iterator
from ideajepa.data.loaders import make_loader
from ideajepa.utils.common import ROOT, load_config

VAL = Path("data/reasoning/val.pt")
pytestmark = pytest.mark.skipif(not VAL.exists(), reason="needs data/reasoning")
TMP = ROOT / "runs/test_resume"
SMALL = ["model.d=32", "model.n_heads=2", "model.r=2", "model.enc_layers=1", "model.pred_layers=1",
         "model.dec_layers=2", "model.read_layers=1", "model.z_read_layers=1", "train.batch_tokens=1024",
         "train.warmup=5", "train.eval_every=20", "train.eval_batches=2", "train.log_every=1", "train.device=cpu",
         "train.steps=40", "group=test_resume", "data.path=runs/test_resume/data"]


@pytest.fixture(scope="module")
def small_data():
    d = TMP / "data"
    d.mkdir(parents=True, exist_ok=True)
    obj = torch.load(VAL, weights_only=False)
    torch.save({"docs": obj["docs"][:600], "vocab": obj["vocab"]}, d / "train.pt")
    torch.save({"docs": obj["docs"][600:700], "vocab": obj["vocab"]}, d / "val.pt")
    shutil.copy(VAL.parent / "meta.pt", d / "meta.pt")
    yield d
    shutil.rmtree(TMP, ignore_errors=True)


def same_batch(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


def test_worker_iterator_resume_replays_stream(small_data):
    """Worker processes (the training path when num_workers > 0; num_workers = 0 trains through DocLoader)."""
    cfg = load_config("configs/v3_stepB_w3.yaml", SMALL)
    loader = make_loader(cfg, "train", shuffle=True, seed=0)
    loader.docs = loader.docs[:60]; loader.lens = loader.lens[:60]        # short epochs: resume across a boundary
    full = worker_iterator(loader, 2, seed=0)
    ref = [next(full) for _ in range(30)]
    full.close()
    for start in (3, 11, 17):
        it = worker_iterator(loader, 2, seed=0, start_batch=start)
        for j in range(start, 30):
            assert same_batch(next(it), ref[j]), (start, j)
        it.close()


def rows(run_dir, split):
    with open(run_dir / f"{split}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def train(name, *extra, gpu=False):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run([sys.executable, "-m", "ideajepa.train", "--config", "configs/v3_stepB_w3.yaml",
                           "--name", name, "--set", *SMALL, *extra], cwd=ROOT, env=env,
                          capture_output=True, text=True).returncode


@pytest.mark.skipif(not os.environ.get("IDEAJEPA_SLOW"), reason="~10 min on CPU: set IDEAJEPA_SLOW=1")
@pytest.mark.parametrize("nw", [0, 2])
def test_train_40_equals_20_plus_resume_20(small_data, nw):
    from ideajepa.train import EXIT_INTERRUPTED
    d_full, d_res = TMP / f"full{nw}", TMP / f"res{nw}"
    for d in (d_full, d_res):
        shutil.rmtree(d, ignore_errors=True)
    assert train(f"full{nw}", f"train.num_workers={nw}", "train.ckpt_every=0") == 0
    assert not (d_full / "last.pt").exists()
    assert train(f"res{nw}", f"train.num_workers={nw}", "train.stop_at=20") == EXIT_INTERRUPTED
    assert (d_res / "last.pt").exists() and len(rows(d_res, "train")) == 20
    assert train(f"res{nw}", f"train.num_workers={nw}") == 0                # resumes from last.pt
    assert (d_res / "final.json").exists()
    tf, tr = rows(d_full, "train"), rows(d_res, "train")
    assert [r["step"] for r in tr] == [r["step"] for r in tf]             # no duplicated / missing rows
    for a, b in zip(tf, tr):
        for k in ("loss", "L_tok", "L_jepa", "L_nce", "grad_norm"):
            assert abs(float(a[k]) - float(b[k])) <= 1e-5 * max(1.0, abs(float(a[k]))), (a["step"], k, a[k], b[k])
    vf, vr = rows(d_full, "val"), rows(d_res, "val")
    assert [r["step"] for r in vr] == [r["step"] for r in vf]
    for a, b in zip(vf, vr):
        assert abs(float(a["ce_recon"]) - float(b["ce_recon"])) <= 1e-5 * max(1.0, abs(float(a["ce_recon"])))


@pytest.mark.skipif(not torch.cuda.is_available() or not os.environ.get("IDEAJEPA_GPU_TEST"),
                    reason="GPU resume test: set IDEAJEPA_GPU_TEST=1 with a free GPU")
def test_gpu_resume_runs_and_matches(small_data):
    """Regression (28/09): on GPU, last.pt is loaded with map_location=cuda and the RNG byte tensors must go back to the
    CPU; the resumed run must finish, with losses close to the uninterrupted run (bf16 kernels: loose tolerance)."""
    from ideajepa.train import EXIT_INTERRUPTED
    gpu = [s for s in SMALL if s != "train.device=cpu"] + ["train.device=cuda", "train.num_workers=0"]
    d_full, d_res = TMP / "gfull", TMP / "gres"
    for d in (d_full, d_res):
        shutil.rmtree(d, ignore_errors=True)
    env_small = [x for x in gpu]
    assert subprocess.run([sys.executable, "-m", "ideajepa.train", "--config", "configs/v3_stepB_w3.yaml", "--name", "gfull",
                           "--set", *env_small, "train.ckpt_every=0"], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                          capture_output=True, text=True).returncode == 0
    run = lambda *x: subprocess.run([sys.executable, "-m", "ideajepa.train", "--config", "configs/v3_stepB_w3.yaml", "--name", "gres",
                                     "--set", *env_small, *x], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                                    capture_output=True, text=True)
    assert run("train.stop_at=20").returncode == EXIT_INTERRUPTED and (d_res / "last.pt").exists()
    r2 = run()
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert (d_res / "final.json").exists()
    tf, tr = rows(d_full, "train"), rows(d_res, "train")
    assert [r["step"] for r in tr] == [r["step"] for r in tf]
    for a, b in zip(tf[-5:], tr[-5:]):
        assert abs(float(a["loss"]) - float(b["loss"])) <= 5e-2 * max(1.0, abs(float(a["loss"]))), (a["step"], a["loss"], b["loss"])


def test_snapshot_at(small_data):
    """train.snapshot_at: model-only checkpoints in sibling dirs <name>@<step>, loadable by load_model."""
    from ideajepa.train import load_model
    shutil.rmtree(TMP / "snap", ignore_errors=True)
    assert train("snap", "train.steps=6", "train.eval_every=6", "train.snapshot_at=[3]", "train.ckpt_every=0") == 0
    m, cfg, _ = load_model(str(TMP / "snap@3" / "ckpt.pt"), device="cpu")
    assert (TMP / "snap@3" / "config.json").exists() and (TMP / "snap" / "final.json").exists()


def test_grad_accum_runs_and_resumes(small_data):
    """train.grad_accum: k micro-batches per optimisation step; the run finishes and a resume replays k batches per step."""
    from ideajepa.train import EXIT_INTERRUPTED
    shutil.rmtree(TMP / "acc", ignore_errors=True)
    assert train("acc", "train.steps=6", "train.eval_every=6", "train.grad_accum=2", "train.num_workers=0",
                 "train.stop_at=3") == EXIT_INTERRUPTED
    assert train("acc", "train.steps=6", "train.eval_every=6", "train.grad_accum=2", "train.num_workers=0") == 0
    assert (TMP / "acc" / "final.json").exists() and len(rows(TMP / "acc", "train")) == 6
