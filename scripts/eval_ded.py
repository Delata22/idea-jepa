"""Deduction-only CE on synthetic v2 val, for Idea-JEPA (gen/recon paths) and GPT baselines.
usage: uv run python eval_ded.py runs/v2/<run> [...]  (run from the project root)"""
import sys, json, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device
from ideajepa.model.ideajepa import IdeaJEPA
from ideajepa.model.objective import idea_losses, per_idea_ce

NB = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 60
runs = [a for a in sys.argv[1:] if not a.isdigit()]


def seq_ded_mask(batch):
    """[B, T-1] bool: target token belongs to a deduction idea (order = doc history)."""
    B, T = batch["seq"].shape
    out = torch.zeros(B, T - 1, dtype=torch.bool, device=batch["seq"].device)
    for b in range(B):
        p = 0
        for j in range(batch["doc_index"].shape[1]):
            r = int(batch["doc_index"][b, j])
            if r < 0:
                break
            n = int(batch["idea_len"][r])
            if bool(batch["is_deduction"][r]):
                out[b, p:p + n] = True
            p += n
    return out


@torch.no_grad()
def run(path):
    model, cfg, ck = load_model(f"{path}/ckpt.pt")
    loader = make_loader(cfg, "val", False)
    acc = {}

    def add(key, ce, mask):
        s, n = acc.get(key, (0.0, 0))
        acc[key] = (s + float((ce * mask).sum()), n + int(mask.sum()))

    for i, batch in enumerate(loader.batches()):
        if i >= NB:
            break
        batch = to_device(batch, "cuda")
        ded = batch["is_deduction"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if isinstance(model, IdeaJEPA):
                for path_ in ("gen", "recon"):
                    out, aux = idea_losses(model, batch, cfg["loss"], force_path=path_, train=False)
                    ce, m = per_idea_ce(aux["logits"], batch["win"], batch["M"], batch["idea_len"])
                    add(f"{path_}_all", ce, m)
                    add(f"{path_}_ded", ce, m * ded[:, None].float())
                    add(f"{path_}_prem", ce, m * (~ded)[:, None].float())
            else:
                kind = model.kind
                if kind in ("b0", "b3"):
                    seq = batch["seq"]
                    logits = model.run(seq[:, :-1])
                    tgt = seq[:, 1:]
                    m = (tgt != 0).float()
                    ce = F.cross_entropy(logits.float().transpose(1, 2), tgt, reduction="none")
                    dm = seq_ded_mask(batch).float()
                else:  # b1: window path, ideas as rows
                    win, M, length = batch["win"], batch["M"], batch["idea_len"]
                    logits = model.run(win[:, :-1])[:, M - 1:]
                    ce, m = per_idea_ce(logits, win, M, length)
                    dm = ded[:, None].float().expand_as(m)
                add("gen_all", ce, m)
                add("gen_ded", ce, m * dm)
                add("gen_prem", ce, m * (1 - dm))
    res = {k: round(s / max(n, 1), 4) for k, (s, n) in acc.items()}
    res["n_ded_tokens"] = acc["gen_ded"][1]
    print(path.rstrip("/").split("/")[-1], json.dumps(res))
    return res


for r in runs:
    run(r)
