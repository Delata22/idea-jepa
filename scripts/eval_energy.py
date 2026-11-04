"""Masked-idea predictor as an energy: mask one deduction per val document, score candidates by cosine
between the predictor's output and E_ema(candidate). Candidates = true idea + distractors (other deductions
of the same story = valid-but-elsewhere, incl. already-said ones; deductions of other stories).
Reports top-1 accuracy and MRR, overall and per family (F1 "state", F2/F3 "chain"), and splits the
same-story distractors into EARLIER deductions (already said before the masked one) and LATER ones.
usage: uv run python scripts/eval_energy.py <stageA_run> [n_batches]"""
import sys, json, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device

NB = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 15
run = sys.argv[1]
model, cfg, _ = load_model(f"{run}/ckpt.pt")
assert getattr(model, "PM", None) is not None, "no masked predictor in this checkpoint"
loader = make_loader(cfg, "val", False)
FAM = {"objin": "F1", "goes": "F1", "takes": "F1", "drops": "F1", "at": "F1", "rel": "F2", "about": "F2",
       "has": "F3", "opens": "F3", "holds": "F3", "canopen": "F3", "gets": "F3"}
_coll = loader.collate


def collate_with_family(docs):  # the collator drops the family label: infer it from the facts' predicates
    b = _coll(docs)
    b["_fam"] = [next(FAM[i["fact"][0]] for i in d["ideas"] if i["fact"][0] in FAM) for d in docs]
    return b


loader.collate = collate_with_family


def flat(v):
    return F.normalize(F.normalize(v, dim=-1).reshape(v.shape[0], -1), dim=-1)


import collections
acc = collections.defaultdict(lambda: collections.defaultdict(float))  # acc[family][metric]


def rank_of(pf_i, cands):
    s = (pf_i[None] @ tf[cands].T)[0]
    r = int((s > s[0]).sum()) + 1
    return r
with torch.no_grad():
    for bi, batch in enumerate(loader.batches()):
        if bi >= NB: break
        batch = to_device(batch, "cuda")
        k, v = model.E(batch["idea_tok"], batch["idea_len"]); kt, vt = model.E_ema(batch["idea_tok"], batch["idea_len"])
        ded = batch["is_deduction"]; docs = batch["idea_doc"]
        score = torch.rand(len(ded), device="cuda") * ded
        best = torch.zeros(int(docs.max()) + 1, device="cuda").scatter_reduce(0, docs, score, reduce="amax")
        masked = (score == best[docs]) & ded & (score > 0)
        mk, mv = model.PM(k, v, batch["doc_index"], masked)
        pf, tf = flat(mv), flat(vt)
        pos = batch["idea_pos"]
        for i in masked.nonzero()[:, 0].tolist():
            same_doc = ((docs == docs[i]) & ded).nonzero()[:, 0]
            same_doc = same_doc[same_doc != i]
            if len(same_doc) < 1: continue
            others = ((docs != docs[i]) & ded).nonzero()[:, 0]
            others = others[torch.randperm(len(others))[:9]]
            me = torch.tensor([i], device="cuda")
            res = {"same": rank_of(pf[i], torch.cat([me, same_doc])),
                   "other": rank_of(pf[i], torch.cat([me, others]))}
            earlier, later = same_doc[pos[same_doc] < pos[i]], same_doc[pos[same_doc] > pos[i]]
            if len(earlier): res["earlier"] = rank_of(pf[i], torch.cat([me, earlier]))
            if len(later): res["later"] = rank_of(pf[i], torch.cat([me, later]))
            for fam in (batch["_fam"][int(docs[i])], "all"):
                a = acc[fam]
                a["n"] += 1
                for key, r in res.items():
                    a[f"{key}_n"] += 1; a[f"{key}_top1"] += r == 1; a[f"{key}_mrr"] += 1.0 / r
name = run.rstrip("/").split("/")[-1]
print(f"{name}  (top-1 / MRR; 'other' = 1 true vs 9 deductions of other stories, chance top-1 0.10)")
print(f"{'family':6s} {'n':>5s}  {'same story':>13s}  {'vs earlier':>13s}  {'vs later':>13s}  {'other stories':>13s}")
out = {}
for fam in ("F1", "F2", "F3", "all"):
    a = acc[fam]
    if not a["n"]: continue
    cell = lambda k: (f"{a[k+'_top1']/a[k+'_n']:.3f} / {a[k+'_mrr']/a[k+'_n']:.3f}" if a[k + "_n"] else "     –      ")
    print(f"{fam:6s} {int(a['n']):5d}  {cell('same'):>13s}  {cell('earlier'):>13s}  {cell('later'):>13s}  {cell('other'):>13s}")
    out[fam] = {k: (a[k + "_top1"] / a[k + "_n"], a[k + "_mrr"] / a[k + "_n"]) for k in ("same", "earlier", "later", "other") if a[k + "_n"]}
json.dump(out, open(f"{run}/energy_by_family.json", "w"), indent=1)
