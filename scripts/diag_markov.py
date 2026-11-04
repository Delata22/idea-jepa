"""Is P Markov? Predict idea n from only the last k past ideas vs from all; report cosine to the full prediction
and to the E_ema target. usage: uv run python scripts/diag_markov.py <run> [n_batches]"""
import sys, statistics as st, torch, collections
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 15
model, cfg, _ = load_model(f"{sys.argv[1]}/ckpt.pt")
loader = make_loader(cfg, "val", False)

def zcos(z1, z2):
    n1, n2 = z1.norm(dim=-1), z2.norm(dim=-1); c = F.cosine_similarity(z1, z2, dim=-1)
    return (c * n1 * n2).sum(-1) / (n1 * n2).sum(-1).clamp_min(1e-6)

res = collections.defaultdict(list)
with torch.no_grad():
    for bi, batch in enumerate(loader.batches()):
        if bi >= NB: break
        batch = to_device(batch, "cuda")
        k, v = model.E(batch["idea_tok"], batch["idea_len"]); kt, vt = model.E_ema(batch["idea_tok"], batch["idea_len"])
        di = batch["doc_index"]; ded = batch["is_deduction"]
        full_k, full_v = model.P(k, v, di)
        for K in (0, 1, 2):
            # mask: keep only the last K ideas before each position (others -> zero slots), by per-position masking
            # implemented by running P separately per target position on a truncated doc_index
            B, Nm = di.shape
            cos_full, cos_tgt_full, cos_tgt_K = [], [], []
            for b in range(B):
                n = int((di[b] >= 0).sum())
                rows = [int(di[b, t]) for t in range(n)]
                prem = [t for t in range(n) if not bool(ded[rows[t]])]
                for j in range(n):                 # target position j = a deduction with > K earlier deductions
                    r = rows[j]
                    earlier_ded = [t for t in range(j) if bool(ded[rows[t]])]
                    if not bool(ded[r]) or len(earlier_ded) <= K: continue
                    keep = sorted([t for t in prem if t < j] + (earlier_ded[len(earlier_ded) - K:] if K > 0 else []))   # all premises + last K deductions
                    dsub = torch.full((1, len(keep) + 1), -1, dtype=torch.long, device="cuda")
                    dsub[0, :len(keep)] = torch.tensor([rows[t] for t in keep], device="cuda"); dsub[0, len(keep)] = r
                    pk, pv = model.P(k, v, dsub)
                    zK = pv[0, len(keep)]; zfull = full_v[b, j]
                    cos_full.append(float(zcos(zK[None], zfull[None])))
                    cos_tgt_K.append(float(zcos(zK[None], vt[r][None])))
                    cos_tgt_full.append(float(zcos(zfull[None], vt[r][None])))
            res[K].append((st.mean(cos_full), st.mean(cos_tgt_K), st.mean(cos_tgt_full)))
for K in (0, 1, 2):
    a = [x[0] for x in res[K]]; b = [x[1] for x in res[K]]; c = [x[2] for x in res[K]]
    print(f"contexte = prémisses + {K} dernière(s) déduction(s) : cos(pred_K, pred_tout) = {st.mean(a):.3f} | cos(pred_K, cible) = {st.mean(b):.3f} | cos(pred_tout, cible) = {st.mean(c):.3f}")
