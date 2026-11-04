"""Throughput benchmark: python scripts/bench.py <config> [overrides...]  (200 steps, tokens/s)"""
import sys, time, torch
from ideajepa.utils.common import load_config
from ideajepa.train import build_model, compute_losses
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import Prefetcher, to_device, worker_iterator
cfg = load_config(sys.argv[1], ["data.max_docs_train=40000"] + sys.argv[2:])
steps = int(cfg["train"].get("bench_steps", 200))
m = build_model(cfg).cuda()
if cfg["train"].get("compile"):
    m.compile_parts(cfg["train"]["compile"])
opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4, fused=True)
nw = cfg["train"].get("num_workers", 3)
it = worker_iterator(make_loader(cfg, "train", True), nw) if nw else Prefetcher(make_loader(cfg, "train", True))
torch.cuda.reset_peak_memory_stats()
ntok = 0
for i in range(steps + 20):
    if i == 20:
        torch.cuda.synchronize(); t = time.time(); ntok = 0
    b = to_device(next(it), "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = compute_losses(m, b, cfg, p_gen=0.5)
    opt.zero_grad(set_to_none=True); out["loss"].backward(); opt.step()
    ntok += int(b["idea_len"].sum())
torch.cuda.synchronize(); dt = time.time() - t
print(f"{sys.argv[1]} {' '.join(sys.argv[2:])}: {dt/steps*1000:.0f} ms/step, {ntok/dt/1e3:.0f}k tok/s, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
