"""bAbI diagnostics without training, for an Idea-JEPA run on qa1 / qa2:
(1) slot probe: are actor / action / object / place linearly decodable from each of E's slots (and from all slots), on
    held-out stories? A high accuracy in ONE slot = the attribute has a stable place across sentences;
(2) identity: cosine between the "actor" slot (best actor probe) of two ideas of the SAME story about the same person vs two
    different persons (and the same on the whole idea); flagged if the gap is < 0.1;
(3) addressing: P's attention mass on the supporting facts (bAbI annotation) when predicting the answer idea, per layer
    (call order, shared blocks included) and head, against chance = supporting facts / ideas of the story (and, for a
    local window, the chance over the ideas visible to that query).
    With a local window (p_window) most supporting facts are out of reach: the rate of reachable ones is reported.
usage: python scripts/diag_babi_slots.py <run> [--split val] [--n 1000] [--json out.json]
"""
import argparse
import collections
import json
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from ideajepa.train import load_model

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--split", default="val")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--json", default=None)
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
PAD = 0

MOVE = {"moved", "went", "journeyed", "travelled"}
TAKE = {"got", "picked", "grabbed", "took"}
DROP = {"dropped", "left", "discarded", "put"}


def parse(ws):
    """(actor, action, object, place) of a qa1-3 statement; None where not applicable."""
    w = [x for x in ws if x not in (".", "there", "back", "up", "down")]
    if len(w) < 3:
        return None
    actor, verb = w[0], w[1]
    if verb in MOVE:
        return actor, "move", None, w[-1]
    if verb in TAKE:
        return actor, "take", w[-1], None
    if verb in DROP:
        return actor, "drop", w[-1], None
    return None


model, cfg, _ = load_model(f"{a.run}/ckpt.pt", device=DEV)
model.eval()
data = torch.load(f"{cfg['data']['path']}/{a.split}.pt", weights_only=False)
docs, vocab = data["docs"][:a.n], data["vocab"]
w = lambda t: [vocab[x] for x in t]
r = model.P.r


def enc(ideas):
    L = max(len(t) for t in ideas)
    tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
    for i, t in enumerate(ideas):
        tok[i, :len(t)] = torch.tensor(t)
    return model.E(tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV))


# ---------------------------------------------------------------- (1) + (2): E's slots on the story statements
feats, labels, story = [], [], []
with torch.no_grad():
    for si, d in enumerate(docs):
        st = [i["tokens"] for i in d["ideas"] if i["step"] == "premise"]
        k, v = enc(st)
        f = torch.cat([k, F.normalize(v, dim=-1)], -1).float().cpu()         # [n, r, 2 ds]
        for j, t in enumerate(st):
            p = parse(w(t))
            if p is not None:
                feats.append(f[j]); labels.append(p); story.append(si)
X = torch.stack(feats).numpy()                                              # [N, r, 2 ds]
story = np.array(story)
test_m = story % 5 == 0                                                     # held-out stories
res = {"run": a.run, "split": a.split, "n_stories": len(docs), "n_statements": len(labels), "probe": {}}
from sklearn.linear_model import LogisticRegression

ATTR = ["actor", "action", "object", "place"]
for ai, name in enumerate(ATTR):
    y = np.array([l[ai] if l[ai] is not None else "" for l in labels])
    has = y != ""
    tr, te = has & ~test_m, has & test_m
    maj = collections.Counter(y[tr]).most_common(1)[0][1] / tr.sum()
    row = {"chance_majority": maj, "n_test": int(te.sum())}
    for s in list(range(r)) + ["all"]:
        Xs = X[:, s] if s != "all" else X.reshape(len(X), -1)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[tr], y[tr])
        row[f"slot{s}" if s != "all" else "all"] = float((clf.predict(Xs[te]) == y[te]).mean())
    res["probe"][name] = row

act_slot = max(range(r), key=lambda s: res["probe"]["actor"][f"slot{s}"])
same, diff, same_all, diff_all = [], [], [], []
Xn = X / np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-8)
Xf = X.reshape(len(X), -1); Xf = Xf / np.linalg.norm(Xf, axis=-1, keepdims=True).clip(1e-8)
actors = np.array([l[0] for l in labels])
for si in np.unique(story):
    idx = np.where(story == si)[0]
    for i_ in range(len(idx)):
        for j_ in range(i_ + 1, len(idx)):
            i, j = idx[i_], idx[j_]
            c, cf = float(Xn[i, act_slot] @ Xn[j, act_slot]), float(Xf[i] @ Xf[j])
            (same if actors[i] == actors[j] else diff).append(c)
            (same_all if actors[i] == actors[j] else diff_all).append(cf)
res["identity"] = {"actor_slot": act_slot, "cos_same_person": float(np.mean(same)), "cos_diff_person": float(np.mean(diff)),
                   "gap": float(np.mean(same) - np.mean(diff)), "whole_idea_same": float(np.mean(same_all)),
                   "whole_idea_diff": float(np.mean(diff_all)), "gap_whole": float(np.mean(same_all) - np.mean(diff_all)),
                   "pairs_same": len(same), "pairs_diff": len(diff), "flag_gap_below_0.1": bool(np.mean(same) - np.mean(diff) < 0.1)}

# ---------------------------------------------------------------- (3): P's attention mass on the supporting facts
CAP = []


def capturing_attend(self, q, k, v, mask=None, is_causal=False):
    qh = self.split(self.q(q))
    s = (qh.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.dh)
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else s + mask.float()
    CAP.append(torch.softmax(s, -1)[0].detach().cpu())                    # [H, T, T]
    return type(self).attend(self, q, k, v, mask, is_causal)


for b in dict.fromkeys(model.P.blocks):     # shared blocks once
    b.attn.attend = types.MethodType(capturing_attend, b.attn)

n_layers = len(model.P.blocks)
H = model.P.blocks[0].attn.h
mass = torch.zeros(n_layers, H); chance_sum = chance_vis_sum = 0.0; count = 0
reach, n_sup = 0, 0
win = int(getattr(model.P, "p_window", 0))
with torch.no_grad():
    for d in docs:
        ctx = [i["tokens"] for i in d["ideas"] if i["step"] != "deduction"]
        sup = [s for s in d.get("support", []) if s < len(ctx) - 1]
        if not sup:
            continue
        k, v = enc(ctx)
        n = len(ctx)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        CAP.clear()
        model.P(k, v, di, role=torch.zeros(n, dtype=torch.long, device=DEV), tokens=model._p_tokens(k),
                words=model._p_words(k, {"idea_support": torch.tensor([j in set(sup) for j in range(n)], device=DEV)}))
        q_rows = slice(n * r, n * r + r)                                    # position n predicts idea n (the answer)
        keys = torch.tensor([t for s in sup for t in range((s + 1) * r, (s + 2) * r)])
        visible = min(n + 1, win) if win else n + 1                          # positions the query can attend to
        reach += sum(1 for s in sup if (n - (s + 1)) < (win if win else 10 ** 9)); n_sup += len(sup)
        for li, att in enumerate(CAP[:n_layers]):
            mass[li] += att[:, q_rows][:, :, keys].sum(-1).mean(-1)
        chance_sum += len(sup) / (n + 1)                                    # chance over the whole story
        chance_vis_sum += min(len(sup), visible) / visible                   # chance over what the query can see
        count += 1
mass /= max(count, 1)
chance = chance_sum / max(count, 1)
chance_vis = chance_vis_sum / max(count, 1)
res["attention"] = {"examples": count, "chance_mean": chance, "chance_visible_mean": chance_vis, "p_window": win,
                    "support_reachable_by_attention": reach / max(n_sup, 1),
                    "mass_per_layer_head": mass.tolist(), "ratio_to_chance_per_layer_head": (mass / chance).tolist(),
                    "max_ratio": float((mass / chance).max()), "criterion_5x": bool(float((mass / chance).max()) >= 5)}

print(f"{a.run} [{a.split}, {len(docs)} histoires, {len(labels)} phrases]")
print("  sonde par slot (exactitude test ; hasard = classe majoritaire) :")
for name, row in res["probe"].items():
    print(f"    {name:7s} hasard {row['chance_majority']:.2f} | " + " ".join(f"s{s} {row[f'slot{s}']:.2f}" for s in range(r))
          + f" | tous {row['all']:.2f}")
idn = res["identity"]
print(f"  identité (slot acteur = s{idn['actor_slot']}) : cos même personne {idn['cos_same_person']:.3f} / autre "
      f"{idn['cos_diff_person']:.3f} → écart {idn['gap']:.3f} (idée entière : {idn['gap_whole']:.3f})"
      + ("  ÉCART < 0.1" if idn["flag_gap_below_0.1"] else ""))
at = res["attention"]
print(f"  attention de P sur les faits d'appui ({at['examples']} ex., hasard {at['chance_mean']:.3f}, fenêtre {win or 'toutes'}, "
      f"hasard sur la partie visible {chance_vis:.3f}, appuis atteignables par l'attention {at['support_reachable_by_attention']:.2f}) :")
for li in range(n_layers):
    print(f"    couche {li} : " + " ".join(f"t{h} {mass[li, h]:.3f} (×{mass[li, h] / chance:.1f})" for h in range(H)))
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
