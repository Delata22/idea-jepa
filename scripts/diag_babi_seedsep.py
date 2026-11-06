"""What separates a converging seed of (2c) from the others before step 3000? For each given snapshot:
(a) cos between the delta-memory WRITE keys of "X takes O" and "X goes to L" ideas of the same story, same X vs different X,
    per layer (keys averaged over the idea's r slot tokens, heads concatenated; 200 validation stories);
(b) gradient norms per module (E, D, O, each P block's attention / MLP / memory) on 8 fixed training batches, at the loss
    of that step (p_gen of that step), no update;
(c) training L_jepa on those batches split into deductions (answers) and non-deductions (premises + question).
usage: python scripts/diag_babi_seedsep.py <snapshot dir> [...] [--json out.json]"""
import argparse, json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.data.dataset import to_device
from ideajepa.data.loaders import make_loader
from ideajepa.train import compute_losses, load_model

PAD = 0
MOVE = {"moved", "went", "journeyed", "travelled"}
TAKE = {"got", "picked", "grabbed", "took"}
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--json", default=None)
ap.add_argument("--n", type=int, default=200); ap.add_argument("--batches", type=int, default=8)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
out = {}
for run in a.runs:
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    m = re.search(r"@(\d+)$", run); step = int(m.group(1)) if m else cfg["train"]["steps"]
    r = model.P.r
    mems = list(model.P.mems)
    res = {}
    # (a) write keys
    d = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
    vocab = d["vocab"]
    same, diff = [[] for _ in mems], [[] for _ in mems]
    model.eval()
    with torch.no_grad():
        for doc in d["docs"][:a.n]:
            ctx = [i["tokens"] for i in doc["ideas"] if i["step"] != "deduction"]
            Lm = max(len(t) for t in ctx)
            tok = torch.full((len(ctx), Lm), PAD, dtype=torch.long)
            for i, t in enumerate(ctx):
                tok[i, :len(t)] = torch.tensor(t)
            tok = tok.to(DEV); lens = torch.tensor([len(t) for t in ctx], device=DEV)
            k, v = model.E(tok, lens)
            n = len(ctx)
            di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
            for m_ in mems:
                m_.capture = True
            model.P(k, v, di, role=torch.zeros(n, dtype=torch.long, device=DEV), tokens=model._p_tokens(k))
            words = [[vocab[t] for t in c] for c in ctx[:-1]]
            takes = [(i, w[0]) for i, w in enumerate(words) if len(w) > 2 and w[1] in TAKE]
            moves = [(i, w[0]) for i, w in enumerate(words) if len(w) > 2 and w[1] in MOVE]
            for li, m_ in enumerate(mems):
                m_.capture = False
                kk = m_.last_qk[1][0]                                             # [H, T, dh]
                idea_k = lambda i: F.normalize(kk[:, (i + 1) * r:(i + 2) * r].mean(1).reshape(-1), dim=0)
                for it_, x in takes:
                    for im_, y in moves:
                        c = float(idea_k(it_) @ idea_k(im_))
                        (same if x == y else diff)[li].append(c)
    res["key_cos_take_move"] = [{"same_actor": float(np.mean(s)), "diff_actor": float(np.mean(t)),
                                 "gap": float(np.mean(s) - np.mean(t))} for s, t in zip(same, diff)]
    # (b) + (c)
    model.train()
    cfg["train"]["batch_tokens"] = 2048        # small fixed batches: the GPU is shared with a training
    loader = make_loader(cfg, "train", shuffle=True, seed=1234)
    it = iter(loader)
    tc = cfg["train"]
    p_gen = tc.get("p_gen_max", 0.5) * min(1.0, step / max(1, tc.get("p_gen_ramp", 0.5) * tc["steps"]))
    groups = {"E": model.E, "D": model.dec, "O": model.O}
    for li in range(len(model.P.blocks)):
        groups[f"P{li}.attn"] = model.P.blocks[li].attn
        groups[f"P{li}.mlp"] = model.P.blocks[li].mlp
        if mems:
            groups[f"P{li}.mem"] = mems[li]
    gn = {g: 0.0 for g in groups}
    jd_ded, jd_non = [], []
    torch.manual_seed(0)
    for _ in range(a.batches):
        b = to_device(next(it), DEV)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            o, _ = compute_losses(model, b, cfg, p_gen=p_gen, train=True, step=step)
        o["loss"].backward()
        for g, mod in groups.items():
            gn[g] += float(sum(p.grad.float().pow(2).sum() for p in mod.parameters() if p.grad is not None)) ** 0.5 / a.batches
        with torch.no_grad():
            k, v = model.encode(b)
            kt, vt = model.E_ema(b["idea_tok"], b["idea_len"])
            pk, pv = model.predict(k, v, b)
            jd = ((pk - kt).pow(2).sum(-1) + (F.normalize(pv, dim=-1) - F.normalize(vt, dim=-1)).pow(2).sum(-1)).mean(-1)
            ded = b["is_deduction"]
            jd_ded.append(float(jd[ded].mean())); jd_non.append(float(jd[~ded].mean()))
    res["grad_norm"] = gn
    res["train_jepa_deductions"] = float(np.mean(jd_ded)); res["train_jepa_premises_question"] = float(np.mean(jd_non))
    out[run] = res
    kc = " ".join(f"L{li} {x['same_actor']:.2f}/{x['diff_actor']:.2f}" for li, x in enumerate(res["key_cos_take_move"]))
    pg = " ".join(f"{li}:{gn[f'P{li}.attn']:.2f}/{gn[f'P{li}.mlp']:.2f}" + (f"/{gn[f'P{li}.mem']:.2f}" if mems else "")
                  for li in range(len(model.P.blocks)))
    print(f"{run}: clés prendre/aller même X / X diff : {kc}\n  gradients E {gn['E']:.2f} D {gn['D']:.2f} O {gn['O']:.2f} | P attn/mlp/mem {pg}"
          f"\n  L_jepa train déductions {res['train_jepa_deductions']:.3f} · prémisses+question {res['train_jepa_premises_question']:.3f}", flush=True)
    del model
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
