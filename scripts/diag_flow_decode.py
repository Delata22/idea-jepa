"""Why do flow samples hit the modes (81 %) but decode badly (24-29 % valid in generation)?
Teacher forcing on v2 val deduction steps; decode the idea from several sources and judge with the verifier:
  A  P regression (pk, pv)                          baseline
  B  flow sample as in generation: flow direction, value norms from P
  C  flow direction, value norms from the TRUE E_ema target
  D  true E_ema direction, value norms from P
  E  true E_ema target (kt, vt)                      ceiling
  F  flow direction, value norms = mean per-slot norm of E_ema over the batch (fixed norms)
usage: uv run python scripts/diag_flow_decode.py <flow_run> [n_batches] [--flow_steps 32] [--flow_temp 0.5]"""
import sys, argparse, collections, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device, Collator
from ideajepa.data import verifier as V
from ideajepa.model.flow_head import from_target

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("nb", nargs="?", type=int, default=3)
ap.add_argument("--flow_steps", type=int, default=32); ap.add_argument("--flow_temp", type=float, default=0.5)
ap.add_argument("--max_steps", type=int, default=150)
a = ap.parse_args()
torch.manual_seed(0)
model, cfg, _ = load_model(f"{a.run}/ckpt.pt")
loader = make_loader(cfg, "val", False)
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
w = lambda toks: " ".join(vocab[t] for t in toks)
is_end = torch.zeros(len(vocab), dtype=torch.bool, device="cuda"); is_end[w2i["."]] = True
FORCE = w2i["so"]
FAM = {"objin": "F1", "goes": "F1", "takes": "F1", "drops": "F1", "at": "F1", "rel": "F2", "about": "F2",
       "has": "F3", "opens": "F3", "holds": "F3", "canopen": "F3", "gets": "F3"}


class TagCollator(Collator):
    def __call__(self, docs):
        b = super().__call__(docs)
        b["_meta"] = [(next(FAM[i["fact"][0]] for i in d["ideas"] if i["fact"][0] in FAM), V.doc_premises(d),
                       [tuple(x["fact"]) if isinstance(x["fact"], list) else x["fact"] for x in d["ideas"][:idx] if x["step"] == "deduction"])
                      for d in docs for idx, i in enumerate(d["ideas"])]
        return b


loader.collate = TagCollator(loader.collate.m, loader.collate.full, loader.collate.max_prefix)


@torch.no_grad()
def decode(win_row, M, K, Vv, max_len=48):
    prefix = win_row[:M][None]; toks = [FORCE]
    for s_ in range(1, max_len):
        win = torch.cat([prefix, torch.tensor([toks + [PAD]], device="cuda")], 1)
        e = model.O(win, M)
        logits, tr = model.dec(win, torch.tensor([s_ + 1], device="cuda"), M, K[None], Vv[None], e)
        if is_end[toks[-1]] or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)): break
        lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf"); toks.append(int(lg.argmax()))
    return toks


def judge(fam, prem, said, toks):
    f, _ = V.parse_sentence(w(toks))
    if f is None: return "unparsed"
    if f in said: return "repeat"
    if f in V.valid_next(fam, prem, list(said)): return "valid"
    if f in set(prem): return "premise"
    if f in V.derivable(fam, prem): return "premature"
    return "false"


res = collections.defaultdict(collections.Counter); nsteps = 0; cosv = collections.defaultdict(list)
with torch.no_grad():
    for bi, batch in enumerate(loader.batches()):
        if bi >= a.nb or nsteps >= a.max_steps: break
        meta = batch.pop("_meta"); batch = to_device(batch, "cuda")
        k, v = model.E(batch["idea_tok"], batch["idea_len"]); kt, vt = model.E_ema(batch["idea_tok"], batch["idea_len"])
        pk, pv = model.predict(k, v, batch)
        z = model.Fh.sample(model.flow_context(pk, pv), n_steps=a.flow_steps, temperature=a.flow_temp)
        kf, vf_dir = from_target(z)                       # unit direction per slot
        mean_norm = vt.norm(dim=-1).mean(0, keepdim=True)  # [1, r]
        for i in range(k.shape[0]):
            if not bool(batch["is_deduction"][i]): continue
            fam, prem, said = meta[i]; wr, M = batch["win"][i], batch["M"]
            npv, nvt = pv[i].norm(dim=-1, keepdim=True), vt[i].norm(dim=-1, keepdim=True)
            variants = {
                "A P": (pk[i], pv[i]),
                "B flow+|P|": (kf[i], vf_dir[i] * npv),
                "C flow+|E|": (kf[i], vf_dir[i] * nvt),
                "D E_dir+|P|": (kt[i], F.normalize(vt[i], dim=-1) * npv),
                "E E_ema": (kt[i], vt[i]),
                "F flow+mean|E|": (kf[i], vf_dir[i] * mean_norm[0][:, None]),
            }
            for name, (K, Vv) in variants.items():
                res[name][judge(fam, prem, said, decode(wr, M, K, Vv))] += 1
            cosv["flow dir vs E dir (values)"].append(float(F.cosine_similarity(F.normalize(vf_dir[i], dim=-1).reshape(-1), F.normalize(vt[i], dim=-1).reshape(-1), dim=0)))
            cosv["P norms / E norms (ratio)"].append(float((npv / nvt.clamp_min(1e-6)).mean()))
            nsteps += 1
            if nsteps >= a.max_steps: break
print(f"{a.run.split('/')[-1]}  ({nsteps} étapes de déduction, teacher forcing)")
for name, c in res.items():
    tot = sum(c.values()); print(f"  {name:16s} " + "  ".join(f"{kk}={vv/tot:.2f}" for kk, vv in sorted(c.items())))
for kk, vv in cosv.items(): print(f"  {kk}: {sum(vv)/len(vv):.3f}")
