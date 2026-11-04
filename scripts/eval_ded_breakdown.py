"""Deduction CE broken down by family (F1 state / F2 F3 chain), restated flag and b_eff.
usage: uv run python scripts/eval_ded_breakdown.py runs/v2/<run> [...] [n_batches]"""
import sys, json, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device, Collator
from ideajepa.model.ideajepa import IdeaJEPA
from ideajepa.model.objective import idea_losses, per_idea_ce

NB = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 60
runs = [a for a in sys.argv[1:] if not a.isdigit()]


VOCAB = torch.load("data/reasoning/val.pt", weights_only=False)["vocab"]
F3W = {"key", "box", "opens", "holds", "opened", "gets", "open"}
F2W = {"uncle", "aunt", "sister", "brother", "mother", "father", "grandmother", "grandfather", "son", "daughter",
       "grandson", "granddaughter", "nephew", "niece", "parent", "child"}


def family_of(doc):
    words = {VOCAB[t] for i in doc["ideas"] for t in i["tokens"]}
    return "F3" if words & F3W else "F2" if words & F2W else "F1"


class TagCollator(Collator):
    def __call__(self, docs):
        b = super().__call__(docs)
        fam, rest, beff = [], [], []
        for d in docs:
            f = d.get("family") or family_of(d)
            for i in d["ideas"]:
                fam.append(f); rest.append(bool(i.get("restated"))); beff.append(int(i.get("b_eff", 0)))
        b["family"] = fam; b["restated"] = torch.tensor(rest); b["b_eff"] = torch.tensor(beff)
        return b


def per_idea_ce_any(model, batch, cfg):
    """returns ce [NI, L], mask [NI, L] over idea rows (gen path for Idea-JEPA)."""
    if isinstance(model, IdeaJEPA):
        out, aux = idea_losses(model, batch, cfg["loss"], force_path="gen", train=False)
        return per_idea_ce(aux["logits"], batch["win"], batch["M"], batch["idea_len"])
    if model.kind == "b1":
        win, M, length = batch["win"], batch["M"], batch["idea_len"]
        logits = model.run(win[:, :-1])[:, M - 1:]
        return per_idea_ce(logits, win, M, length)
    # b0 / b3: CE over seq, then scatter back to idea rows in doc order
    seq = batch["seq"]; logits = model.run(seq[:, :-1]); tgt = seq[:, 1:]
    ce_seq = F.cross_entropy(logits.float().transpose(1, 2), tgt, reduction="none")
    NI, L = batch["win"].shape[0], batch["win"].shape[1] - batch["M"]
    ce = torch.zeros(NI, L, device=seq.device); m = torch.zeros(NI, L, device=seq.device)
    for b in range(seq.shape[0]):
        p = 0
        for j in range(batch["doc_index"].shape[1]):
            r = int(batch["doc_index"][b, j])
            if r < 0: break
            n = int(batch["idea_len"][r]); ce[r, :n] = ce_seq[b, p:p + n]; m[r, :n] = 1; p += n
    return ce, m


@torch.no_grad()
def run(path):
    model, cfg, ck = load_model(f"{path}/ckpt.pt")
    loader = make_loader(cfg, "val", False)
    loader.collate = TagCollator(loader.collate.m, loader.collate.full, loader.collate.max_prefix)
    acc = {}
    def add(key, s, n):
        a, b = acc.get(key, (0.0, 0)); acc[key] = (a + s, b + n)
    for i, batch in enumerate(loader.batches()):
        if i >= NB: break
        fam = batch.pop("family"); batch = to_device(batch, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, m = per_idea_ce_any(model, batch, cfg)
        ded = batch["is_deduction"]; rest = batch["restated"]; beff = batch["b_eff"]
        s_idea = (ce * m).sum(1); n_idea = m.sum(1)
        for r in range(ce.shape[0]):
            if not bool(ded[r]): continue
            s, n = float(s_idea[r]), int(n_idea[r])
            add("all", s, n); add(f"fam_{fam[r]}", s, n)
            add("restated" if bool(rest[r]) else "implicit", s, n)
            add(f"b{min(int(beff[r]),3)}", s, n)
    res = {k: round(a / max(b, 1), 3) for k, (a, b) in sorted(acc.items())}
    print(path.rstrip("/").split("/")[-1], json.dumps(res))


for r in runs: run(r)
