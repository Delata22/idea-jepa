"""Paraphrase invariance of E on synthetic v2 val: cosine between slot encodings of ideas sharing an idea_id
(same fact, different formulation/story) vs ideas with different idea_ids; AUC of same-vs-different.
usage: uv run python scripts/eval_para.py <run> [<run> ...] [n_batches]"""
import sys, json, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device

NB = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 20
runs = [a for a in sys.argv[1:] if not a.isdigit()]


def auc(pos, neg):
    pos, neg = torch.tensor(pos), torch.tensor(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n = min(len(neg), 20000); neg = neg[torch.randperm(len(neg))[:n]]
    return float((pos[:, None] > neg[None, :]).float().mean())


@torch.no_grad()
def run(path):
    model, cfg, _ = load_model(f"{path}/ckpt.pt")
    loader = make_loader(cfg, "val", False)
    same_c, diff_c, same_k, diff_k = [], [], [], []
    for bi, batch in enumerate(loader.batches()):
        if bi >= NB: break
        batch = to_device(batch, "cuda")
        k, v = model.E(batch["idea_tok"], batch["idea_len"])
        ded = batch["is_deduction"]; ids = batch["idea_id"]
        k, v, ids = k[ded], v[ded], ids[ded]
        vf = F.normalize(F.normalize(v, dim=-1).reshape(len(ids), -1), dim=-1)
        kf = F.normalize(k.reshape(len(ids), -1), dim=-1)
        cv, ck = vf @ vf.T, kf @ kf.T
        same = (ids[:, None] == ids[None, :]); eye = torch.eye(len(ids), dtype=torch.bool, device="cuda")
        same_c += cv[same & ~eye].tolist(); diff_c += cv[~same].tolist()
        same_k += ck[same & ~eye].tolist(); diff_k += ck[~same].tolist()
    t = lambda x: sum(x) / max(len(x), 1)
    res = {"cos_v_same": round(t(same_c), 3), "cos_v_diff": round(t(diff_c), 3), "auc_v": round(auc(same_c, diff_c), 3),
           "cos_k_same": round(t(same_k), 3), "cos_k_diff": round(t(diff_k), 3), "auc_k": round(auc(same_k, diff_k), 3),
           "n_same_pairs": len(same_c)}
    print(path.rstrip("/").split("/")[-1], json.dumps(res))


for r in runs: run(r)
