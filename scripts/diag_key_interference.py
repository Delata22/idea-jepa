"""Interference test for P's idea-level associative memory (DeltaMemory in each P block).
For every pair of ideas in the same val story, cosine between the memory KEYS written for them (normalised
per head, averaged over the r slot tokens, heads and layers), grouped by:
  same_pred_diff_ent : same predicate (template, e.g. objin/objin), different entities
  diff_pred_same_ent : different predicates, sharing an entity (e.g. objin(book,.) / takes(sam,book))
  diff_pred_diff_ent : different predicates, no shared entity
If keys encode the template more than the entity, same_pred_diff_ent >> diff_pred_same_ent: reads of one object's
position return a mixture of all objects' positions.
usage: uv run python scripts/diag_key_interference.py <run> [n_batches]"""
import sys, collections, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device, Collator

CENTER = "--center" in sys.argv
args = [a for a in sys.argv[1:] if a != "--center"]
NB = int(args[1]) if len(args) > 1 else 8
model, cfg, _ = load_model(f"{args[0]}/ckpt.pt")
P = model.P
assert getattr(P, "idea_memory", "none") != "none", "this run has no idea memory"
loader = make_loader(cfg, "val", False)


class FactCollator(Collator):
    def __call__(self, docs):
        b = super().__call__(docs)
        b["_facts"] = [tuple(i["fact"]) if isinstance(i["fact"], list) else i["fact"] for d in docs for i in d["ideas"]]
        return b


loader.collate = FactCollator(loader.collate.m, loader.collate.full, loader.collate.max_prefix)
captured = []
hooks = [m.qkv.register_forward_hook(lambda mod, inp, out: captured.append(out.detach())) for m in P.mems]


def ents(f):
    out = set()
    for x in f[1:]:
        if isinstance(x, str): out.add(x)
        elif isinstance(x, tuple): out.add(x[-1])
    return out


acc = collections.defaultdict(list)
per_layer = collections.defaultdict(lambda: collections.defaultdict(list))
with torch.no_grad():
    for bi, batch in enumerate(loader.batches()):
        if bi >= NB: break
        facts = batch.pop("_facts"); batch = to_device(batch, "cuda")
        k, v = model.E(batch["idea_tok"], batch["idea_len"])
        captured.clear()
        P(k, v, batch["doc_index"])
        di = batch["doc_index"]; B, Nm = di.shape; r = P.r; H = P.mems[0].h; dh = P.mems[0].dh
        for li, out in enumerate(captured):                       # out: [B, Nm*r, 3*d]
            keys = out.view(B, Nm, r, 3, H, dh)[:, :, :, 1]         # [B, Nm, r, H, dh]
            keys = F.normalize(keys.float(), dim=-1)
            for b in range(B):
                n = int((di[b] >= 0).sum())
                # token block j holds the slots of idea j-1 (P shifts by one start block)
                rows = [int(di[b, j - 1]) for j in range(1, n)]
                kb = keys[b, 1:n]                                    # [n-1, r, H, dh]
                if CENTER:  # remove the story's mean key (per slot and head), then renormalise
                    kb = F.normalize(kb - kb.mean(0, keepdim=True), dim=-1)
                for x in range(len(rows)):
                    for y in range(x + 1, len(rows)):
                        fx, fy = facts[rows[x]], facts[rows[y]]
                        if fx is None or fy is None or fx[0] in ("about", "marker") or fy[0] in ("about", "marker"): continue
                        c = float((kb[x] * kb[y]).sum(-1).mean())
                        same_pred = fx[0] == fy[0]; shared = bool(ents(fx) & ents(fy))
                        key = "same_pred_diff_ent" if same_pred and not shared else \
                              "identical_fact" if fx == fy else \
                              "same_pred_same_ent" if same_pred else \
                              "diff_pred_same_ent" if shared else "diff_pred_diff_ent"
                        per_layer[li][key].append(c)
                        if li == len(captured) - 1: acc[key].append(c)
for h in hooks: h.remove()
print(("[centré] " if CENTER else "") + f"{args[0].split('/')[-1]} : cos moyen des clés mémoire (normalisées par tête, moyenne slots×têtes)")
for li in sorted(per_layer):
    print(f"  couche {li}: " + "  ".join(f"{k}={sum(v)/len(v):.3f} (n={len(v)})" for k, v in sorted(per_layer[li].items())))
