"""Recall diagnostics of P's idea-level delta memory, for F1 deductions "obj ends up in L" (obj position premise earlier).
Reconstructs each head's memory state S_t with the exact delta recurrence from the captured q, k, v, beta:
    S <- S - beta (S k) k^T + beta v k^T,   read r = S q
and reports, per layer:
  beta         : distribution of the write strength over all memory tokens
  retention    : read with the premise's OWN keys (oracle address), right after its write vs at the deduction step:
                 cos(read, v_premise) and cos(read, v_other_premises)  -> conservation vs overwrite
  actual       : read with P's actual query at the deduction step: cos to v_premise_same vs v_premise_other
  slot_best    : per (query slot, key slot) pair, P(cos(q, k_same) > max cos(q, k_other)); best pair reported
usage: uv run python scripts/diag_recall.py <run> [n_batches]"""
import sys, os, json, collections, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device, Collator

NB = int(sys.argv[2]) if len(sys.argv) > 2 else 4
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
model, cfg, _ = load_model(f"{sys.argv[1]}/ckpt.pt", device=DEV)
P = model.P
assert getattr(P, "idea_memory", "none") == "delta"
loader = make_loader(cfg, "val", False)


class FactCollator(Collator):
    def __call__(self, docs):
        b = super().__call__(docs)
        b["_facts"] = [(tuple(i["fact"]) if isinstance(i["fact"], list) else i["fact"], i["step"]) for d in docs for i in d["ideas"]]
        return b


loader.collate = FactCollator(loader.collate.m, loader.collate.full, loader.collate.max_prefix)
cap_qkv, cap_b = [], []
hooks = []
for m in P.mems:
    hooks.append(m.qkv.register_forward_hook(lambda mod, i, o: cap_qkv.append(o.detach().float())))
    hooks.append(m.b.register_forward_hook(lambda mod, i, o: cap_b.append(torch.sigmoid(o.detach().float()))))

acc = collections.defaultdict(lambda: collections.defaultdict(list))
betas = collections.defaultdict(list)
slot_hits = collections.defaultdict(lambda: None)
slot_n = collections.defaultdict(int)


def qkv_heads(out, B, Nm, r, H, dh, mem):
    q, k, v = out.view(B, Nm, r, 3, H, dh).unbind(3)
    if mem.center:
        q = q - mem.q_mu; k = k - mem.k_mu
    return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v


with torch.no_grad():
    for bi, batch in enumerate(loader.batches()):
        if bi >= NB: break
        facts = batch.pop("_facts"); batch = to_device(batch, DEV)
        k, v = model.E(batch["idea_tok"], batch["idea_len"])
        cap_qkv.clear(); cap_b.clear(); P(k, v, batch["doc_index"])
        di = batch["doc_index"]; B, Nm = di.shape; r = P.r; H = P.mems[0].h; dh = P.mems[0].dh
        for li in range(len(P.mems)):
            q_, k_, v_ = qkv_heads(cap_qkv[li], B, Nm, r, H, dh, P.mems[li])
            beta = cap_b[li].view(B, Nm, r, H)
            for b in range(B):
                n_ideas = int((di[b] >= 0).sum()); rows = [int(di[b, j]) for j in range(n_ideas)]
                betas[li].append(beta[b, 1:n_ideas].reshape(-1).cpu())
                # delta recurrence over token blocks 0..n_ideas (block j holds idea j-1), token order = (block, slot)
                S = torch.zeros(H, dh, dh, device=DEV)                # [H, dv, dk]
                states = {}
                for j in range(n_ideas):
                    for s in range(r):
                        kk, vv, bb = k_[b, j, s], v_[b, j, s], beta[b, j, s]     # [H, dh], [H, dh], [H]
                        Sk = torch.einsum("hvk,hk->hv", S, kk)
                        S = S + bb[:, None, None] * torch.einsum("hv,hk->hvk", vv - Sk, kk)
                    states[j] = S.clone()                                 # state after writing block j
                prem_pos = {}                                             # object -> idea index of its position premise
                for p in range(n_ideas):
                    f, st = facts[rows[p]]
                    if st == "premise" and f is not None and f[0] == "objin": prem_pos.setdefault(f[1], p)
                for n in range(1, n_ideas):
                    f, st = facts[rows[n]]
                    if st != "deduction" or f is None or f[0] != "objin" or f[1] not in prem_pos: continue
                    p = prem_pos[f[1]]; others = [pp for o, pp in prem_pos.items() if o != f[1] and pp < n]
                    if not others: continue
                    kb = k_[b, p + 1]; vb = v_[b, p + 1]                   # premise p is written at block p+1
                    def read(Sx, qx):  # qx [r, H, dh] -> read [r, H, dv]
                        return torch.einsum("hvk,rhk->rhv", Sx, qx)
                    def cosv(x, y): return float(F.cosine_similarity(x.reshape(-1), y.reshape(-1), dim=0))
                    r_after = read(states[p + 1], kb); r_late = read(states[n], kb)   # state before block n+1 = after block n
                    acc[li]["retention_after_write"].append(cosv(r_after, vb))
                    acc[li]["retention_at_deduction"].append(cosv(r_late, vb))
                    acc[li]["oracle_read_to_other_v"].append(sum(cosv(r_late, v_[b, pp + 1]) for pp in others) / len(others))
                    acc[li]["baseline_v_same_to_v_other"].append(sum(cosv(vb, v_[b, pp + 1]) for pp in others) / len(others))
                    # same, after subtracting the mean value of all ideas of the story (per slot and head)
                    vmean = v_[b, 1:n_ideas].mean(0)
                    acc[li]["centered_v_same_to_v_other"].append(sum(cosv(vb - vmean, v_[b, pp + 1] - vmean) for pp in others) / len(others))
                    # does the centred value identify the location? compare premises of other objects AT THE SAME place
                    same_loc = [pp for o, pp in prem_pos.items() if o != f[1] and pp < n and facts[rows[pp]][0][2] == facts[rows[p]][0][2]]
                    diff_loc = [pp for pp in others if pp not in same_loc]
                    if same_loc and diff_loc:
                        acc[li]["centered_v_other_obj_same_loc"].append(sum(cosv(vb - vmean, v_[b, pp + 1] - vmean) for pp in same_loc) / len(same_loc))
                        acc[li]["centered_v_other_obj_diff_loc"].append(sum(cosv(vb - vmean, v_[b, pp + 1] - vmean) for pp in diff_loc) / len(diff_loc))
                    r_act = read(states[n], q_[b, n])                     # P's query at block n predicts idea n
                    acc[li]["actual_read_to_same_v"].append(cosv(r_act, vb))
                    acc[li]["actual_read_to_other_v"].append(sum(cosv(r_act, v_[b, pp + 1]) for pp in others) / len(others))
                    # slot-by-slot addressing
                    cs = torch.einsum("ihd,jhd->ijh", q_[b, n], kb).mean(-1)                         # [r, r]
                    co = torch.stack([torch.einsum("ihd,jhd->ijh", q_[b, n], k_[b, pp + 1]).mean(-1) for pp in others]).max(0).values
                    hit = (cs > co).float().cpu()
                    ci = float(torch.einsum("ihd,jhd->ijh", q_[b, n], kb).mean())            # whole idea, mean over slots
                    coi = max(float(torch.einsum("ihd,jhd->ijh", q_[b, n], k_[b, pp + 1]).mean()) for pp in others)
                    acc[li]["idea_hit(q->same premise > other premises)"].append(float(ci > coi))
                    acc[li]["idea_hit_chance"].append(1.0 / (1 + len(others)))
                    slot_hits[li] = hit if slot_hits[li] is None else slot_hits[li] + hit
                    slot_n[li] += 1
for h in hooks: h.remove()
name = sys.argv[1].split("/")[-1]
print(f"{name} : rappel dans la mémoire delta de P (déductions F1 sur un objet ayant une prémisse de position)")
for li in sorted(acc):
    bt = torch.cat(betas[li])
    print(f"  couche {li}: beta médiane {bt.median():.2f} (q10 {bt.quantile(0.1):.2f}, q90 {bt.quantile(0.9):.2f})  " +
          "  ".join(f"{k}={sum(v)/len(v):.3f}" for k, v in sorted(acc[li].items())) + f"  (n={len(acc[li]['retention_at_deduction'])})")
    sh = slot_hits[li] / slot_n[li]
    i, j = divmod(int(sh.argmax()), sh.shape[1])
    print(f"           slot par slot : P(requête→bonne prémisse > autres) max {float(sh.max()):.2f} (slot requête {i}, slot clé {j}), moyenne {float(sh.mean()):.2f}")
json.dump({"run": sys.argv[1], "layers": {li: {**{k: sum(v) / len(v) for k, v in acc[li].items()},
                                               "beta_median": float(torch.cat(betas[li]).median()),
                                               "slot_best": float((slot_hits[li] / slot_n[li]).max()),
                                               "slot_mean": float((slot_hits[li] / slot_n[li]).mean())} for li in acc}},
          open(f"{sys.argv[1]}/recall.json", "w"), indent=1)
