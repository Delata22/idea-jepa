from __future__ import annotations

import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]


def deep_update(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path, overrides=()):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = deep_update(base, cfg)
    for ov in overrides:
        key, val = ov.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        v = yaml.safe_load(val)
        if isinstance(v, str):  # YAML 1.1 reads "5e-4" as a string
            try:
                v = float(v)
            except ValueError:
                pass
        d[parts[-1]] = v
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, run_dir: Path, use_tb=True, resume_step=None):
        """resume_step (resumed run, = the next training step): the existing train.csv / val.csv are kept up to that
        point (train rows with step < resume_step, val rows with step <= resume_step) and appended to afterwards;
        tensorboard events from resume_step on are purged."""
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.tb = None
        if resume_step is not None:
            for split, keep in (("train", lambda s: s < resume_step), ("val", lambda s: s <= resume_step)):
                f = self.dir / f"{split}.csv"
                if not f.exists():
                    continue
                with open(f, newline="") as fh:
                    rd = csv.DictReader(fh)
                    cols = list(rd.fieldnames or [])
                    rows = [r for r in rd if keep(float(r["step"]))]
                with open(f, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=cols)
                    w.writeheader()
                    w.writerows(rows)
                self.files[split] = (rows, cols)
        if use_tb:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(self.dir / "tb"), purge_step=resume_step)

    def log(self, split, step, metrics: dict):
        metrics = {k: (float(v) if torch.is_tensor(v) or isinstance(v, (int, float, np.floating)) else v)
                   for k, v in metrics.items()}
        row = {"step": step, **metrics}
        rows, cols = self.files.setdefault(split, ([], []))
        rows.append(row)
        f = self.dir / f"{split}.csv"
        if any(k not in cols for k in row):
            cols.extend(k for k in row if k not in cols)
            with open(f, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
        else:
            with open(f, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=cols).writerow(row)
        if self.tb:
            for k, v in metrics.items():
                if isinstance(v, float):
                    self.tb.add_scalar(f"{split}/{k}", v, step)

    def json(self, name, obj):
        (self.dir / name).write_text(json.dumps(obj, indent=2))

    def close(self):
        if self.tb:
            self.tb.close()
