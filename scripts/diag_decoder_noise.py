"""Decoder tolerance to off-manifold idea codes (calibration for a generative idea head).

For N val stories, every deduction idea is decoded (greedy) from Z = E_ema(true sentence) perturbed in the
flow space [k ; v/|v|] by Gaussian noise of std sigma x (per-dimension std of the E_ema targets), keys
re-normalised and each value's true norm restored (only the direction is noised, as a flow head would
produce). Context = the true previous sentences (teacher forcing). Verdicts per sigma and family:
  exact        the true fact of that position
  valid_other  another fact of the exact frontier (valid here, not the reference one)
  repeat / premise / premature / false / unparsed   (see src/ideajepa/data/verifier.py)
usage: uv run python scripts/diag_decoder_noise.py <run> [--n 90] [--sigmas 0 0.1 0.2 0.3 0.5 0.8]
"""
import argparse
import collections
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ideajepa.data import verifier as V
from ideajepa.data.dataset import Collator, to_device
from ideajepa.eval.generate import generate_idea
from ideajepa.model.flow_head import to_target
from ideajepa.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--n", type=int, default=90)
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

DEV = a.device
model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
data = torch.load(f"{cfg['data']['path']}/val.pt", weights_only=False)
vocab = data["vocab"]
maxi = cfg["data"]["max_ideas"]
docs = [d for d in data["docs"] if len(d["ideas"]) <= maxi]
fams = ["F1", "F2", "F3"]
docs = [d for f in fams for d in [x for x in docs if x["family"] == f][:a.n // 3]]  # n/3 stories per family
w = lambda toks: " ".join(vocab[t] for t in toks)
is_end = torch.zeros(len(vocab), dtype=torch.bool, device=DEV)
is_end[vocab.index(".")] = True
m = cfg["model"]["m_prefix"]
b = to_device(Collator(m_prefix=m)(docs), DEV)

# idea rows of deductions + their verdict context (true premises, true previous deductions)
rows, ctx = [], []
r = 0
for d in docs:
    prem = V.doc_premises(d)
    said = []
    for i in d["ideas"]:
        if i["step"] == "deduction":
            rows.append(r)
            ctx.append((d["family"], prem, list(said), i["fact"]))
            said.append(i["fact"])
        r += 1
rows = torch.tensor(rows, device=DEV)

with torch.no_grad():
    kt, vt = model.E_ema(b["idea_tok"], b["idea_len"])
    kt, vt = kt[rows].float(), vt[rows].float()
    z = to_target(kt, vt)                               # flow space [k ; v/|v|]
    std = z.reshape(-1, z.shape[-1]).std(0)             # per-dimension std of the targets
    vnorm = vt.norm(dim=-1, keepdim=True)
    prefix = b["win"][rows, :b["M"]]
    g = torch.Generator(device=DEV).manual_seed(a.seed)


def verdict(fam, prem, said, true_fact, sent):
    f, _ = V.parse_sentence(sent)
    if f is None:
        return "unparsed"
    if f == true_fact:
        return "exact"
    front = V.valid_next(fam, prem, said)
    if f in front:
        return "valid_other"
    if f in said:
        return "repeat"
    if f in set(prem):
        return "premise"
    return "premature" if f in V.derivable(fam, prem) else "false"


results = {}
for sigma in a.sigmas:
    with torch.no_grad():
        zn = z + sigma * std * torch.randn(z.shape, device=DEV, generator=g)
        kn, vn = zn.chunk(2, -1)
        kn = F.normalize(kn, dim=-1)
        vn = F.normalize(vn, dim=-1) * vnorm              # keep each value's true norm: noise the direction only
        if cfg["model"].get("z_norm", "none") == "idea":  # same rescaling as in training (objective.py)
            vn = vn / vn.pow(2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()[:, None, None]
        cos = F.cosine_similarity(zn.flatten(1), z.flatten(1), dim=-1).mean().item()
        out, _, _ = generate_idea(model, prefix, kn, vn, 25, is_end, "punct")
    counts = collections.defaultdict(collections.Counter)
    for (fam, prem, said, tf), toks in zip(ctx, out):
        v = verdict(fam, prem, said, tf, w(toks))
        counts[fam][v] += 1
        counts["all"][v] += 1
    results[sigma] = {"cos_to_clean": cos, **{k: dict(c) for k, c in counts.items()}}
    tot = sum(counts["all"].values())
    ok = counts["all"]["exact"] + counts["all"]["valid_other"]
    per = "  ".join(f"{f}: exact {counts[f]['exact'] / max(1, sum(counts[f].values())):.2f}" for f in fams)
    print(f"sigma={sigma:<4}  cos(noisy, clean)={cos:.3f}  exact {counts['all']['exact'] / tot:.3f}  "
          f"valid (exact+other) {ok / tot:.3f}  false {counts['all']['false'] / tot:.3f}  "
          f"unparsed {counts['all']['unparsed'] / tot:.3f}   | {per}", flush=True)

name = a.run.rstrip("/").split("/")[-1]
json.dump({"run": name, "n_ideas": len(ctx), "results": results}, open(f"{a.run}/decoder_noise.json", "w"), indent=1)
print(f"wrote {a.run}/decoder_noise.json ({len(ctx)} deduction ideas)")
