"""Does P's memory query, when predicting an F1 deduction "obj ends up in L", point to the premise that stores obj's
position, or to deductions/premises about OTHER objects? (hypothesis: premise and deduction surfaces differ, so the
query built in 'deduction surface' matches other deductions better than the right premise)
For each F1 deduction idea n about object o: query vectors of P's memory at the token block that predicts idea n
(block n holds the slots of idea n-1), keys of earlier ideas. Mean cos over (query slot, key slot, head) pairs,
per layer, for key groups:
  premise_same_obj : premise stating o's initial position
  premise_other_obj: premises stating other objects' positions
  ded_other_obj    : earlier deductions about other objects
usage: uv run python scripts/diag_query_keys.py <run> [n_batches]"""
import sys, os, json, collections, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device, Collator

NB = int(sys.argv[2]) if len(sys.argv) > 2 else 8
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
model, cfg, _ = load_model(f"{sys.argv[1]}/ckpt.pt", device=DEV)
P = model.P
assert getattr(P, "idea_memory", "none") != "none"
loader = make_loader(cfg, "val", False)


class FactCollator(Collator):
    def __call__(self, docs):
        b = super().__call__(docs)
        b["_facts"] = [(tuple(i["fact"]) if isinstance(i["fact"], list) else i["fact"], i["step"]) for d in docs for i in d["ideas"]]
        return b


loader.collate = FactCollator(loader.collate.m, loader.collate.full, loader.collate.max_prefix)
cap = []
hooks = [m.qkv.register_forward_hook(lambda mod, inp, out: cap.append(out.detach())) for m in P.mems]
acc = collections.defaultdict(lambda: collections.defaultdict(list))
with torch.no_grad():
    for bi, batch in enumerate(loader.batches()):
        if bi >= NB: break
        facts = batch.pop("_facts"); batch = to_device(batch, DEV)
        k, v = model.E(batch["idea_tok"], batch["idea_len"])
        cap.clear(); P(k, v, batch["doc_index"])
        di = batch["doc_index"]; B, Nm = di.shape; r = P.r; H = P.mems[0].h; dh = P.mems[0].dh
        for li, out in enumerate(cap):
            qkv = out.view(B, Nm, r, 3, H, dh).float()
            q = F.normalize(qkv[:, :, :, 0], dim=-1); kk = F.normalize(qkv[:, :, :, 1], dim=-1)
            if P.mems[li].center:
                q = F.normalize(qkv[:, :, :, 0] - P.mems[li].q_mu, dim=-1); kk = F.normalize(qkv[:, :, :, 1] - P.mems[li].k_mu, dim=-1)
            for b in range(B):
                n_ideas = int((di[b] >= 0).sum()); rows = [int(di[b, j]) for j in range(n_ideas)]
                for n in range(1, n_ideas):
                    f, step = facts[rows[n]]
                    if step != "deduction" or f is None or f[0] != "objin": continue
                    obj = f[1]
                    qn = q[b, n]                                              # [r, H, dh] : predicts idea n
                    groups = collections.defaultdict(list)
                    for p in range(n):
                        fp, sp = facts[rows[p]]
                        if fp is None or fp[0] != "objin": continue
                        g = ("premise_" if sp == "premise" else "ded_") + ("same_obj" if fp[1] == obj else "other_obj")
                        kp = kk[b, p + 1]                                     # block p+1 holds idea p
                        groups[g].append(float(torch.einsum("ihd,jhd->ijh", qn, kp).mean()))
                    for g, vals in groups.items():
                        acc[li][g].append(sum(vals) / len(vals))
                    if groups.get("premise_same_obj") and groups.get("ded_other_obj"):
                        acc[li]["P(premise_same > max ded_other)"].append(float(max(groups["premise_same_obj"]) > max(groups["ded_other_obj"])))
                    if groups.get("premise_same_obj") and groups.get("premise_other_obj"):
                        acc[li]["P(premise_same > max premise_other)"].append(float(max(groups["premise_same_obj"]) > max(groups["premise_other_obj"])))
                        acc[li]["chance(premise_same > max premise_other)"].append(1.0 / (1 + len(groups["premise_other_obj"])))
for h in hooks: h.remove()
print(f"{sys.argv[1].split('/')[-1]} : cos(requête de P au moment de déduire la position d'un objet, clés des idées passées)")
for li in sorted(acc):
    print(f"  couche {li}: " + "  ".join(f"{k}={sum(v)/len(v):.3f} (n={len(v)})" for k, v in sorted(acc[li].items())))
json.dump({"run": sys.argv[1], "layers": {li: {k: sum(v) / len(v) for k, v in acc[li].items()} for li in acc}},
          open(f"{sys.argv[1]}/query_keys.json", "w"), indent=1)
