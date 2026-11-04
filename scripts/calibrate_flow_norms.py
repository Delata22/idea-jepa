"""Fill the e_slot_norm buffer (mean per-slot |v| of E_ema on val deductions) of an existing checkpoint, and set
model.flow_norms = E in its saved config. usage: uv run python scripts/calibrate_flow_norms.py <run>"""
import sys, torch
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device
run = sys.argv[1]
model, cfg, ck = load_model(f"{run}/ckpt.pt")
loader = make_loader(cfg, "val", False); acc = []; n = 0
with torch.no_grad():
    for bi, batch in enumerate(loader.batches()):
        if bi >= 10: break
        batch = to_device(batch, "cuda")
        kt, vt = model.E_ema(batch["idea_tok"], batch["idea_len"])
        acc.append(vt[batch["is_deduction"]].norm(dim=-1).sum(0)); n += int(batch["is_deduction"].sum())
norms = torch.stack(acc).sum(0) / n
ck["model"]["e_slot_norm"] = norms.cpu(); ck["cfg"]["model"]["flow_norms"] = "E"
torch.save(ck, f"{run}/ckpt.pt"); print(run, "e_slot_norm =", [round(float(x), 3) for x in norms], "| flow_norms=E")
