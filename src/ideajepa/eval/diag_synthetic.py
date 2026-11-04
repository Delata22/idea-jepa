"""Stage-1 diagnostics on the synthetic data.

python -m ideajepa.eval.diag_synthetic runs/stage1/v_full_s0 [more run dirs...]
Writes <run>/diag.json (and prints a summary).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..data import synthetic as S
from ..data.dataset import Collator, chunk_docs, to_device
from ..data.loaders import load_split
from ..model.baseline import GPT
from ..model.ideajepa import IdeaJEPA, ParallelDecoder
from ..model.objective import idea_losses
from ..train import load_model
from .generate import generate_idea

DEV = "cuda"
TYPE_NAMES = S.TYPES
ROLE_NAMES = S.ROLES


def batches(docs, cfg, bs_docs=256, full_prefix=None):
    m = cfg["model"]
    coll = Collator(m_prefix=m["m_prefix"], full_prefix=m.get("full_prefix", False) if full_prefix is None
                    else full_prefix)
    docs = chunk_docs(docs, cfg["data"]["max_ideas"], m["max_idea_len"])
    for i in range(0, len(docs), bs_docs):
        yield to_device(coll(docs[i:i + bs_docs]), DEV), docs[i:i + bs_docs]


LOSS_EVAL = None


@torch.no_grad()
def run_path(model, cfg, docs, path):
    """Teacher-forced pass over docs; returns per-idea / per-token arrays."""
    rec = {k: [] for k in ["rho", "len", "len_class", "g", "a", "types", "roles", "correct", "ce",
                           "k", "v", "agent", "patient", "idea_id", "voice", "tok", "ev",
                           "p_norm_ratio", "p_cos", "stop"]}
    for b, chunk in batches(docs, cfg):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, aux = idea_losses(model, b, cfg["loss"], force_path=path, train=False)
        lg = aux["logits"].float()
        tgt = b["win"][:, b["M"]:]
        ce = F.cross_entropy(lg.transpose(1, 2), tgt, reduction="none")
        tr = aux["trace"]
        rec["rho"] += list(tr["rho"].float().cpu().numpy())
        rec["len"] += b["idea_len"].tolist()
        rec["g"] += list(tr["g"].float().cpu().numpy())
        rec["a"] += list(tr["a"].float().cpu().numpy())
        rec["types"] += list(b["types"].cpu().numpy())
        rec["roles"] += list(b["roles"].cpu().numpy())
        rec["correct"] += list((lg.argmax(-1) == tgt).cpu().numpy())
        rec["ce"] += list(ce.cpu().numpy())
        rec["tok"] += list(tgt.cpu().numpy())
        if "stop_logit" in tr:
            rec["stop"] += list(tr["stop_logit"].float().cpu().numpy())
        if "ev" in tr:
            rec["ev"] += list(tr["ev"].float().cpu().numpy())
        # shrinkage of P's prediction relative to the encoder slots of the true idea
        pv, v_ = aux["pv"].float(), aux["v"].float()
        rec["p_norm_ratio"] += (pv.norm(dim=(-1, -2)) / v_.norm(dim=(-1, -2)).clamp_min(1e-6)).tolist()
        w_ = v_.norm(dim=-1)  # per-slot cosine weighted by the encoder slot norm (empty slots do not count)
        rec["p_cos"] += ((F.cosine_similarity(pv, v_, dim=-1) * w_).sum(-1) / w_.sum(-1).clamp_min(1e-8)).tolist()
        K0 = aux["pk"] if path == "gen" else aux["k"]
        V0 = aux["pv"] if path == "gen" else aux["v"]
        rec["k"] += list(K0.float().cpu().numpy())
        rec["v"] += list(V0.float().cpu().numpy())
        ideas = [i for d in chunk for i in d["ideas"]]
        rec["len_class"] += [i["len_class"] for i in ideas]
        rec["agent"] += [i["agent"] for i in ideas]
        rec["patient"] += [i["patient"] for i in ideas]
        rec["idea_id"] += [i["idea_id"] for i in ideas]
        rec["voice"] += [i["voice"] for i in ideas]
    return rec


def exhaustion_stats(rec, tau):
    rho_end = np.array([r[n] for r, n in zip(rec["rho"], rec["len"])])
    lc = np.array(rec["len_class"])
    lens = np.array(rec["len"])
    buckets = {"short": lc <= 1, "medium": (lc >= 2) & (lc <= 4), "long": lc >= 5}
    out = {"rho_end_mean": float(rho_end.mean()), "rho_end_median": float(np.median(rho_end)),
           "frac_below_tau": float((rho_end < tau).mean()),
           "corr_rho_end_len": float(np.corrcoef(rho_end, lens)[0, 1]) if rho_end.std() > 0 else 0.0}
    for k, msk in buckets.items():
        out[f"frac_below_tau_{k}"] = float((rho_end[msk] < tau).mean())
        out[f"rho_end_{k}"] = float(rho_end[msk].mean())
        out[f"mean_len_{k}"] = float(lens[msk].mean())
        out[f"rho_end_q_{k}"] = [float(x) for x in np.quantile(rho_end[msk], [0.1, 0.25, 0.5, 0.75, 0.9])]
    # crossing position vs true end and vs last content word
    d_end, prem, cross_rel = [], [], []
    for r, n, t in zip(rec["rho"], rec["len"], rec["types"]):
        after = r[1:n + 1]  # rho after emitting token s
        below = np.nonzero(after < tau)[0]
        cross = below[0] if len(below) else n  # n = never crossed
        content = np.nonzero(t[:n] == S.T["content"])[0]
        last_c = content[-1] if len(content) else n - 1
        d_end.append((n - 1) - cross)
        prem.append(cross < last_c)
        cross_rel.append(cross - last_c)
    d_end, prem, cross_rel = np.array(d_end), np.array(prem), np.array(cross_rel)
    out.update({"cross_before_end_mean": float(d_end.mean()),
                "frac_cross_at_or_after_last_content": float(1 - prem.mean()),
                "frac_premature_cross": float(prem.mean()),
                "cross_minus_last_content_mean": float(cross_rel.mean()),
                "frac_cross_exactly_last_content": float((cross_rel == 0).mean())})
    return out, rho_end


def gate_by_type(rec):
    sums, cnt = np.zeros(len(TYPE_NAMES)), np.zeros(len(TYPE_NAMES))
    for g, t, n in zip(rec["g"], rec["types"], rec["len"]):
        for ti in range(len(TYPE_NAMES)):
            msk = t[:n] == ti
            sums[ti] += g[:n][msk].sum()
            cnt[ti] += msk.sum()
    res = {TYPE_NAMES[i]: float(sums[i] / max(cnt[i], 1)) for i in range(len(TYPE_NAMES))}
    # per-token g values for a significance test (function vs content)
    gf = np.concatenate([g[:n][t[:n] == S.T["function"]] for g, t, n in zip(rec["g"], rec["types"], rec["len"])])
    gc = np.concatenate([g[:n][t[:n] == S.T["content"]] for g, t, n in zip(rec["g"], rec["types"], rec["len"])])
    from scipy.stats import mannwhitneyu
    res["auc_content_vs_function"] = float(mannwhitneyu(gc, gf).statistic / (len(gc) * len(gf))) if gf.std() + gc.std() > 0 else 0.5
    res["diff_content_minus_function"] = float(gc.mean() - gf.mean())
    return res


def stop_stats(rec):
    """Closure by the stop head (teacher forcing): first step s with stop_logit > 0 vs true end n."""
    if not rec["stop"]:
        return None
    lc = np.array(rec["len_class"])
    diff, exact = [], []
    for st, n in zip(rec["stop"], rec["len"]):
        fire = np.nonzero(st[:n + 1] > 0)[0]
        c = fire[0] if len(fire) else n + 1
        diff.append(c - n)
        exact.append(c == n)
    diff, exact = np.array(diff), np.array(exact)
    out = {"stop_exact": float(exact.mean()), "stop_early": float((diff < 0).mean()),
           "stop_late_or_never": float((diff > 0).mean()), "stop_mean_abs_err": float(np.abs(diff).mean())}
    for k, msk in {"short": lc <= 1, "medium": (lc >= 2) & (lc <= 4), "long": lc >= 5}.items():
        out[f"stop_exact_{k}"] = float(exact[msk].mean())
    return out


def ev_by_type(rec):
    """Mean sum_i a_i |e_t . v_i^res| / ||Z_0|| when producing each word type (erasure *direction* selectivity)."""
    if not rec["ev"]:
        return None
    out = {}
    for ti, name in enumerate(TYPE_NAMES):
        vals = np.concatenate([ev[:n][t[:n] == ti] for ev, t, n in zip(rec["ev"], rec["types"], rec["len"])])
        out[name] = float(vals.mean()) if len(vals) else None
    return out


@torch.no_grad()
def paraphrase_invariance(model, cfg, vocab, n=1000, seed=7):
    """Cosine between E's slot values for two formulations of the same idea vs two different ideas."""
    w2i = {w: i for i, w in enumerate(vocab)}
    rng = random.Random(seed)
    ents = list(S.ENTITIES)
    pairs = []
    for _ in range(n):
        a, p = rng.sample(ents, 2)
        idea = S.Idea(a, rng.choice(list(S.VERBS)), p, rng.choice(S.PLACES) if rng.random() < 0.5 else None)
        for _ in range(2):
            surf, _ = S.realize(idea, random.Random(rng.random()), [], allow_pronoun=False)
            pairs.append((idea, surf))
    docs = single_idea_docs(pairs, w2i)
    coll = Collator(m_prefix=cfg["model"]["m_prefix"], full_prefix=cfg["model"].get("full_prefix", False))
    b = to_device(coll(docs), DEV)
    k, v = model.encode(b)
    zf = v.flatten(1)
    z = F.normalize(zf - zf.mean(0, keepdim=True), dim=-1).view(n, 2, -1)  # centred cosine
    same = (z[:, 0] * z[:, 1]).sum(-1)
    diff = (z[:, 0] * z.roll(1, 0)[:, 1]).sum(-1)
    lens = b["idea_len"].view(n, 2).float()
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(n), np.zeros(n)]
    sc = np.r_[same.cpu().numpy(), diff.cpu().numpy()]
    return {"cos_same_idea": float(same.mean()), "cos_diff_idea": float(diff.mean()),
            "auc_same_vs_diff": float(roc_auc_score(y, sc)),
            "mean_abs_len_diff_same": float((lens[:, 0] - lens[:, 1]).abs().mean())}


@torch.no_grad()
def patching_test(model, cfg, vocab, slots, n=500):
    """Activation patching: copy the given slots (k and v) of Z_0(p,v,a) into Z_0(a,v,p); greedy decode;
    success = entity order of the output follows the patched (inverted) idea."""
    if not slots:
        return {"patch_success": None}
    w2i = {w: i for i, w in enumerate(vocab)}
    pairs = S.inverted_pair_probe(n, seed=123)
    flat = [x for pr in pairs for x in pr]
    docs = single_idea_docs(flat, w2i)
    coll = Collator(m_prefix=cfg["model"]["m_prefix"], full_prefix=cfg["model"].get("full_prefix", False))
    b = to_device(coll(docs), DEV)
    k, v = model.encode(b)
    k1, v1, k2, v2 = k[0::2], v[0::2], k[1::2], v[1::2]
    kp, vp = k1.clone(), v1.clone()
    kp[:, slots], vp[:, slots] = k2[:, slots], v2[:, slots]
    is_end = torch.zeros(len(vocab), dtype=torch.bool, device=DEV)
    is_end[w2i["."]] = True
    prefix = b["win"][0::2, :b["M"]]
    ent_ids = {w2i[e]: e for e in S.ENTITIES}
    out, _, _ = generate_idea(model, prefix, kp, vp, 25, is_end, "punct")
    ok = []
    for pr, toks in zip(pairs, out):
        idea2, surf2 = pr[1]
        want = [w for w in surf2.words if w in S.ENTITIES]
        found = [ent_ids[t] for t in toks if t in ent_ids]
        ok.append(found[:2] == want[:2])
    return {"patch_success": float(np.mean(ok)), "patched_slots": list(map(int, slots))}


def role_slot_matrix(rec, r):
    """Mean read attention over slots when producing content words of each role."""
    roles = ["agent", "action", "patient", "loc", "time"]
    M = np.zeros((len(roles), r))
    for ri, role in enumerate(roles):
        rid = S.R[role]
        acc, n = np.zeros(r), 0
        for a, ro, t, L in zip(rec["a"], rec["roles"], rec["types"], rec["len"]):
            msk = (ro[:L] == rid) & (t[:L] == S.T["content"])
            if msk.any():
                acc += a[:L][msk].sum(0)
                n += msk.sum()
        M[ri] = acc / max(n, 1)
    return roles, M


def key_probe(rec_tr, rec_te):
    """Predict the role of a content word from the attention-weighted key read at that step."""
    def feats(rec, maxn=40000):
        X, y = [], []
        for a, k, ro, t, L in zip(rec["a"], rec["k"], rec["roles"], rec["types"], rec["len"]):
            msk = (t[:L] == S.T["content"])
            for s in np.nonzero(msk)[0]:
                X.append(a[s] @ k)
                y.append(ro[s])
            if len(X) > maxn:
                break
        return np.array(X), np.array(y)
    Xtr, ytr = feats(rec_tr)
    Xte, yte = feats(rec_te, 10000)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    return float(clf.score(sc.transform(Xte), yte))


def z0_probe(rec_tr, rec_te, n_tr=8000):
    ents = sorted(S.ENTITIES)
    def X(rec, n):
        return np.stack([np.concatenate([k.ravel(), v.ravel()]) for k, v in zip(rec["k"][:n], rec["v"][:n])])
    Xtr, Xte = X(rec_tr, n_tr), X(rec_te, 3000)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    for role in ("agent", "patient"):
        ytr = np.array([ents.index(e) for e in rec_tr[role][:n_tr]])
        yte = np.array([ents.index(e) for e in rec_te[role][:3000]])
        clf = LogisticRegression(max_iter=1000, C=0.1).fit(Xtr, ytr)
        out[f"probe_{role}"] = float(clf.score(Xte, yte))
    # surface vs meaning (passives only): "agent" (meaning) vs "first noun phrase" (surface) labels
    def first_np(rec, n):
        return [p if vo == "passive" else a for a, p, vo in zip(rec["agent"][:n], rec["patient"][:n], rec["voice"][:n])]
    pas = np.array([vo == "passive" for vo in rec_te["voice"][:3000]])
    if pas.sum() > 20:
        for name, lab_tr, lab_te in (("meaning_agent", rec_tr["agent"][:n_tr], rec_te["agent"][:3000]),
                                     ("surface_first_np", first_np(rec_tr, n_tr), first_np(rec_te, 3000))):
            ytr = np.array([ents.index(e) for e in lab_tr])
            yte = np.array([ents.index(e) for e in lab_te])
            clf = LogisticRegression(max_iter=1000, C=0.1).fit(Xtr, ytr)
            out[f"passive_probe_{name}"] = float(clf.score(Xte[pas], yte[pas]))
    return out


def single_idea_docs(pairs, w2i):
    docs = []
    for idea, surf in pairs:
        docs.append({"ideas": [{"tokens": [w2i[w] for w in surf.words], "roles": surf.roles,
                                "types": surf.types, "idea_id": 0, "agent": idea.agent,
                                "patient": idea.patient, "voice": "active", "len_class": 0}]})
    return docs


def entity_positions(types, roles, L):
    return [s for s in range(L) if types[s] == S.T["content"] and roles[s] in (S.R["agent"], S.R["patient"])]


@torch.no_grad()
def binding_inverted(model, cfg, vocab, n=1500):
    """Teacher-forced accuracy on the entity nouns of (a,v,p)/(p,v,a) pairs, recon path.
    first = the first entity noun of the sentence (pure binding decision: nothing emitted yet)."""
    w2i = {w: i for i, w in enumerate(vocab)}
    pairs = S.inverted_pair_probe(n)
    flat = [x for pr in pairs for x in pr]
    docs = single_idea_docs(flat, w2i)
    rec = run_path(model, cfg, docs, "recon")
    first, second, both = [], [], []
    for c, t, ro, L in zip(rec["correct"], rec["types"], rec["roles"], rec["len"]):
        pos = entity_positions(t, ro, L)
        first.append(c[pos[0]])
        second.append(c[pos[1]])
        both.append(c[pos[0]] and c[pos[1]])
    # pair-level: both sentences of the inverted pair fully right on the entities
    both = np.array(both).reshape(-1, 2)
    return {"bind_first_entity_acc": float(np.mean(first)), "bind_second_entity_acc": float(np.mean(second)),
            "bind_pair_acc": float(both.all(1).mean())}, rec, docs


def slot_ids_for_roles(M_roles):
    roles, M = M_roles
    return int(M[roles.index("agent")].argmax()), int(M[roles.index("patient")].argmax())


@torch.no_grad()
def swap_test(model, cfg, vocab, agent_slot, patient_slot, n=500):
    """Swap v_agent <-> v_patient in Z_0 (recon slots), greedy decode, check entities are exchanged."""
    if agent_slot == patient_slot:
        return {"swap_success": None, "swap_note": "agent and patient share a slot"}
    w2i = {w: i for i, w in enumerate(vocab)}
    rng = random.Random(5)
    ents = list(S.ENTITIES)
    pairs = []
    for _ in range(n):
        a, p = rng.sample(ents, 2)
        idea = S.Idea(a, rng.choice(list(S.VERBS)), p)
        surf, _ = S.realize(idea, random.Random(rng.random()), [], forms={"agent": "short", "patient": "short",
                            "verb": "short"}, voice="active", order="std", allow_pronoun=False)
        pairs.append((idea, surf))
    docs = single_idea_docs(pairs, w2i)
    coll = Collator(m_prefix=cfg["model"]["m_prefix"], full_prefix=cfg["model"].get("full_prefix", False))
    b = to_device(coll(docs), DEV)
    k, v = model.encode(b)
    v2 = v.clone()
    v2[:, agent_slot], v2[:, patient_slot] = v[:, patient_slot], v[:, agent_slot]
    is_end = torch.zeros(len(vocab), dtype=torch.bool, device=DEV)
    is_end[w2i["."]] = True
    prefix = b["win"][:, :b["M"]]
    res = {}
    ent_ids = {w2i[e]: e for e in ents}
    for name, vv in (("orig", v), ("swap", v2)):
        out, _, _ = generate_idea(model, prefix, k, vv, 20, is_end, "punct")
        ok = []
        for (idea, _), toks in zip(pairs, out):
            found = [ent_ids[t] for t in toks if t in ent_ids]
            want = [idea.agent, idea.patient] if name == "orig" else [idea.patient, idea.agent]
            ok.append(found[:2] == want)
        res[f"{name}_entities_as_expected"] = float(np.mean(ok))
    res["swap_success"] = res.pop("swap_entities_as_expected")
    return res


@torch.no_grad()
def slot_ablation(model, cfg, docs, r):
    """Zero one slot's value; teacher-forced accuracy drop per role (content words), recon path."""
    roles = ["agent", "action", "patient", "loc", "time"]
    base = None
    drop = np.zeros((r, len(roles)))
    def acc_by_role(kill):
        tot, cor = np.zeros(len(roles)), np.zeros(len(roles))
        for b, _ in batches(docs, cfg):
            k, v = model.encode(b)
            if kill is not None:
                v = v.clone()
                v[:, kill] = 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = model.decode(b, k, v)
            c = (lg.argmax(-1) == b["win"][:, b["M"]:]).cpu().numpy()
            ro, ty, L = b["roles"].cpu().numpy(), b["types"].cpu().numpy(), b["idea_len"].cpu().numpy()
            for ri, role in enumerate(roles):
                msk = (ro == S.R[role]) & (ty == S.T["content"])
                tot[ri] += msk.sum()
                cor[ri] += (c & msk).sum()
        return cor / np.maximum(tot, 1)
    base = acc_by_role(None)
    for i in range(r):
        drop[i] = base - acc_by_role(i)
    return {"roles": roles, "base_acc": base.tolist(), "drop": drop.tolist()}


@torch.no_grad()
def aphasia_examples(model, cfg, vocab, docs, n=6):
    if not isinstance(model.dec, ParallelDecoder):
        return None
    w2i = {w: i for i, w in enumerate(vocab)}
    is_end = torch.zeros(len(vocab), dtype=torch.bool, device=DEV)
    is_end[w2i["."]] = True
    one = [{"ideas": [d["ideas"][0]]} for d in docs[:n]]
    coll = Collator(m_prefix=cfg["model"]["m_prefix"])
    b = to_device(coll(one), DEV)
    k, v = model.encode(b)
    prefix = b["win"][:, :b["M"]]
    out = {"reference": [" ".join(vocab[t] for t in d["ideas"][0]["tokens"]) for d in one]}
    for name, fg, cut in (("normal", None, False), ("gate0", 0.0, False), ("no_local", None, True)):
        model.dec.force_gate, model.dec.cut_local = fg, cut
        toks, _, _ = generate_idea(model, prefix, k, v, 25, is_end, "punct")
        out[name] = [" ".join(vocab[t] for t in s) for s in toks]
    model.dec.force_gate, model.dec.cut_local = None, False
    return out


@torch.no_grad()
def eval_ce(model, cfg, docs):
    tot = {}
    n = 0
    for b, _ in batches(docs, cfg):
        ntok = int(b["idea_len"].sum())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if isinstance(model, IdeaJEPA):
                vals = {}
                for path in ("gen", "recon"):
                    out, _ = idea_losses(model, b, cfg["loss"], force_path=path, train=False)
                    vals[f"ce_{path}"] = out["L_tok"].item()
            else:
                out, _ = model.losses(b)
                vals = {"ce_gen": out["L_tok"].item()}
        for k, v in vals.items():
            tot[k] = tot.get(k, 0) + v * ntok
        n += ntok
    return {k: v / n for k, v in tot.items()}


@torch.no_grad()
def heldout_sentence_acc(model, cfg, docs, path):
    """Accuracy on entity nouns of held-out (agent, patient) sentences in test_comp."""
    held = set(map(tuple, torch.load(Path(cfg["_root"]) / cfg["data"]["path"] / "meta.pt",
                                     weights_only=False)["heldout_pairs"]))
    rec = run_path(model, cfg, docs, path)
    acc = []
    for c, t, ro, L, a, p in zip(rec["correct"], rec["types"], rec["roles"], rec["len"], rec["agent"], rec["patient"]):
        if (a, p) in held:
            pos = entity_positions(t, ro, L)
            acc.append(np.mean([c[s] for s in pos]))
    return float(np.mean(acc)) if acc else None


def diagnose(run_dir: Path, n_docs=2000, full=True):
    from ..utils.common import ROOT
    model, cfg, _ = load_model(run_dir / "ckpt.pt")
    cfg["_root"] = str(ROOT)
    vocab = torch.load(ROOT / cfg["data"]["path"] / "meta.pt", weights_only=False)["vocab"]
    test_iid, _ = load_split(cfg["data"], "test_iid")
    test_comp, _ = load_split(cfg["data"], "test_comp")
    train, _ = load_split(cfg["data"], "val")  # probes are fit on val, tested on test_iid
    test_iid, test_comp, train = test_iid[:n_docs], test_comp[:n_docs], train[:n_docs]
    res = {"run": run_dir.name, "model": cfg["model"]["model"]}
    res["iid"] = eval_ce(model, cfg, test_iid)
    res["comp"] = eval_ce(model, cfg, test_comp)
    if not isinstance(model, IdeaJEPA):
        return res
    tau = cfg["loss"].get("tau", 0.1)
    r = cfg["model"]["r"]
    rec_tr = run_path(model, cfg, train, "recon")
    rec_te = run_path(model, cfg, test_iid, "recon")
    rec_gen = run_path(model, cfg, test_iid, "gen")
    res["exhaustion_recon"], rho_r = exhaustion_stats(rec_te, tau)
    res["exhaustion_gen"], rho_g = exhaustion_stats(rec_gen, tau)
    res["stop_recon"] = stop_stats(rec_te)
    res["stop_gen"] = stop_stats(rec_gen)
    res["gate_recon"] = gate_by_type(rec_te)
    res["gate_gen"] = gate_by_type(rec_gen)
    res["a_max_content"] = float(np.mean(np.concatenate(
        [a[:n].max(-1)[t[:n] == S.T["content"]] for a, t, n in zip(rec_te["a"], rec_te["types"], rec_te["len"])])))
    res["ev_recon"] = ev_by_type(rec_te)
    res["ev_gen"] = ev_by_type(rec_gen)
    res["p_shrink"] = {"norm_ratio_mean": float(np.mean(rec_gen["p_norm_ratio"])),
                       "cos_mean": float(np.mean(rec_gen["p_cos"]))}
    res["paraphrase"] = paraphrase_invariance(model, cfg, vocab)
    roles, M = role_slot_matrix(rec_te, r)
    res["role_slot_matrix"] = {"roles": roles, "M": M.tolist()}
    res["key_var_per_slot"] = np.stack(rec_te["k"]).var(0).sum(-1).tolist()
    res["key_probe_role_acc"] = key_probe(rec_tr, rec_te)
    res["z0_probe"] = z0_probe(rec_tr, rec_te)
    res["binding"], _, _ = binding_inverted(model, cfg, vocab)
    a_slot, p_slot = slot_ids_for_roles((roles, M))
    res["swap"] = swap_test(model, cfg, vocab, a_slot, p_slot)
    ap_slots = [i for i in range(r) if M[roles.index("agent"), i] + M[roles.index("patient"), i] > 0.1]
    res["patching"] = patching_test(model, cfg, vocab, ap_slots)
    res["slot_ablation"] = slot_ablation(model, cfg, test_iid[:500], r)
    res["heldout_entity_acc_recon"] = heldout_sentence_acc(model, cfg, test_comp, "recon")
    res["heldout_entity_acc_gen"] = heldout_sentence_acc(model, cfg, test_comp, "gen")
    res["aphasia"] = aphasia_examples(model, cfg, vocab, test_iid)
    # raw arrays for figures
    np.savez_compressed(run_dir / "diag_arrays.npz", rho_end_recon=rho_r, rho_end_gen=rho_g,
                        len_class=np.array(rec_te["len_class"]), lens=np.array(rec_te["len"]),
                        rho_traj=np.array([np.pad(x, (0, 60 - len(x)), constant_values=np.nan) for x in rec_te["rho"]]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n_docs", type=int, default=2000)
    a = ap.parse_args()
    for r in a.runs:
        rd = Path(r)
        res = diagnose(rd, a.n_docs)
        (rd / "diag.json").write_text(json.dumps(res, indent=1, default=float))
        short = {k: v for k, v in res.items() if k not in ("aphasia", "slot_ablation", "role_slot_matrix")}
        print(json.dumps(short, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
