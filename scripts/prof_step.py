"""Step-time profile of a config on bAbI qa2 (collation time on one process, and compute time per optimisation step).
usage: python scripts/prof_step.py <overrides...>"""
import sys, time, torch
from ideajepa.utils.common import load_config
from ideajepa.train import build_model, compute_losses
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device
ov = sys.argv[1:]
cfg = load_config("configs/babi_im.yaml", ["data.path=data/babi/qa2_10k", "data.p_input_surface=random"] + ov)
m = build_model(cfg).cuda()
if cfg["train"].get("compile"): m.compile_parts(cfg["train"]["compile"])
opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4, fused=True)
L = make_loader(cfg, "train", True)
it = iter(L)
t_data = t_comp = 0.0; n = 60
batches = []
t0 = time.time()
for i in range(n): batches.append(next(it))
t_data = (time.time() - t0) / n
for i, b in enumerate(batches):
    if i == 10: torch.cuda.synchronize(); t0 = time.time()
    b = to_device(b, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = compute_losses(m, b, cfg, p_gen=0.5, train=True, step=1000)
    opt.zero_grad(set_to_none=True); out["loss"].backward(); opt.step()
torch.cuda.synchronize(); t_comp = (time.time() - t0) / (n - 10)
print(" ".join(ov), f"| collation {t_data*1000:.0f} ms/lot (1 processus) | calcul {t_comp*1000:.0f} ms/pas | lot {cfg['train']['batch_tokens']} tokens")
