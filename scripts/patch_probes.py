"""Add the passive surface/meaning probe to diag.json files computed before it existed."""
import json
import sys
from pathlib import Path

from ideajepa.data.loaders import load_split
from ideajepa.eval.diag_synthetic import run_path, z0_probe
from ideajepa.model.ideajepa import IdeaJEPA
from ideajepa.train import load_model

for rd in map(Path, sys.argv[1:]):
    f = rd / "diag.json"
    d = json.loads(f.read_text())
    if "z0_probe" not in d or "passive_probe_meaning_agent" in d["z0_probe"]:
        continue
    model, cfg, _ = load_model(rd / "ckpt.pt")
    if not isinstance(model, IdeaJEPA):
        continue
    tr, _ = load_split(cfg["data"], "val")
    te, _ = load_split(cfg["data"], "test_iid")
    d["z0_probe"] = z0_probe(run_path(model, cfg, tr[:800], "recon"), run_path(model, cfg, te[:800], "recon"))
    f.write_text(json.dumps(d, indent=1, default=float))
    print(rd.name, d["z0_probe"], flush=True)
