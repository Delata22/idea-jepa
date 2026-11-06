"""Phase 1 of the flow-matching study: does P predict the AVERAGE of the valid next deductions?
Teacher forcing on v2 val. For every deduction step t: V_t = valid unsaid deductions (generator's candidates),
each encoded by E_ema as the mean over ALL its paraphrases (3 connectors x 2 forms); p_t = P's prediction.
Cosines (JEPA normalisation: per-slot-normalised values, flattened, L2) to the best candidate, the centroid,
the true next sentence (raw E_ema of the text) and invalid deductions of the same story. Stratified by
n_t in {1,2,3,4+} and family. Extras: decoding from the centroid vs from each candidate (verifier verdicts),
semantic entropy (10 samples at T=1 grouped by fact), fixed-point gap 1 - cos(z, E_ema(D(z))) and its
calibration (cos between paraphrases of one fact).
usage: uv run python scripts/diag_averaging.py <run> [n_batches] [--decode 200]"""
import sys, json, random, math, argparse, collections, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data.loaders import make_loader
from ideajepa.data.dataset import to_device, Collator
from ideajepa.data import reasoning as R, verifier as V
from ideajepa.eval.samplers import idea_candidates

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("nb", nargs="?", type=int, default=12)
ap.add_argument("--decode", type=int, default=200, help="steps with n>=2 for centroid decoding / semantic entropy / fixed point")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sampler", default="P", choices=["P", "flow", "khyp"],
                help="idea analysed as 'P's prediction': the regression (P), one flow sample, or the best-scored hypothesis")
ap.add_argument("--flow_steps", type=int, default=16)
ap.add_argument("--flow_temp", type=float, default=1.0)
a = ap.parse_args()
rng = random.Random(a.seed); torch.manual_seed(a.seed)
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
        b["_meta"] = [(next(FAM[i["fact"][0]] for i in d["ideas"] if i["fact"][0] in FAM), i.get("candidates") or [],
                       tuple(i["fact"]) if isinstance(i["fact"], list) else i["fact"], V.doc_premises(d),
                       [tuple(x["fact"]) if isinstance(x["fact"], list) else x["fact"] for x in d["ideas"][:idx] if x["step"] == "deduction"])
                      for d in docs for idx, i in enumerate(d["ideas"])]
        return b


loader.collate = TagCollator(loader.collate.m, loader.collate.full, loader.collate.max_prefix)


def paraphrases(fact):
    """All rendered forms of a deduction (3 connectors x 2 clause forms), as token lists."""
    out = []
    for c in range(len(R.CONNECT)):
        for form in (0, 1):
            s = R.S(); s.add(R.CONNECT[c]); cl, _ = R.render_clause(fact, rng, form); s.ext(cl); s.add(".")
            toks = [w2i[x] for x in s.w]
            if toks not in out: out.append(toks)
    return out


_cache = {}


@torch.no_grad()
def encode_fact(fact):
    """(k_mean, v_mean, k_all, v_all) over the fact's paraphrases, E_ema; raw (unnormalised) values."""
    if fact in _cache: return _cache[fact]
    ps = paraphrases(fact); L = max(len(t) for t in ps)
    tok = torch.full((len(ps), L), PAD, dtype=torch.long)
    for i, t in enumerate(ps): tok[i, :len(t)] = torch.tensor(t)
    k, v = model.E_ema(tok.cuda(), torch.tensor([len(t) for t in ps], device="cuda"))
    res = (F.normalize(k.mean(0), dim=-1), v.mean(0), k, v)
    _cache[fact] = res
    return res


def flat_v(v):  # JEPA normalisation: per-slot normalised values, flattened, L2
    return F.normalize(F.normalize(v, dim=-1).reshape(-1), dim=0)


def flat_k(k):
    return F.normalize(k.reshape(-1), dim=0)


def cos(a_, b_): return float((a_ * b_).sum())


def invalid_facts(fam, fact, prem, said, truth):
    """Plausible invalid alternatives: already-said facts + perturbations of the true fact where ONE argument is
    replaced by a value seen in the same argument position of the same predicate in the story (premises or
    entailed facts), provided the result is not entailed."""
    pool = collections.defaultdict(set)
    for f in list(prem) + list(truth):
        for pos in range(1, len(f)):
            pool[(f[0], pos)].add(f[pos])
    alts = []
    for pos in range(1, len(fact)):
        for e in pool.get((fact[0], pos), ()):
            if e == fact[pos]: continue
            g = list(fact); g[pos] = e; g = tuple(g)
            if g not in truth: alts.append(g)
    rng.shuffle(alts)
    return list(dict.fromkeys(list(said) + alts[:3]))[:4]


def safe_encode(fact):
    try: return encode_fact(fact)
    except Exception: return None


# ------------------------------------------------------------------ pass 1: cosines
acc = collections.defaultdict(lambda: collections.defaultdict(list))
steps_n2 = []  # (kt, vt rows / candidate reps) kept for pass 2
with torch.no_grad():
    for bi, batch in enumerate(loader.batches()):
        if bi >= a.nb: break
        meta = batch.pop("_meta"); batch = to_device(batch, "cuda")
        k, v = model.E(batch["idea_tok"], batch["idea_len"]); kt, vt = model.E_ema(batch["idea_tok"], batch["idea_len"])
        CK, CV = idea_candidates(model, k, v, batch, a.sampler, 1, a.flow_steps, a.flow_temp)
        pk, pv = CK[:, 0], CV[:, 0]
        for i in range(k.shape[0]):
            fam, cands, fact, prem, said = meta[i]
            if not bool(batch["is_deduction"][i]) or not cands: continue
            cands = [tuple(c) if isinstance(c, list) else c for c in cands]
            cands = [tuple(tuple(x) if isinstance(x, list) else x for x in c) for c in cands]
            n = len(cands); nb_ = "4+" if n >= 4 else str(n)
            reps = [encode_fact(c) for c in cands]
            p_v, p_k = flat_v(pv[i]), flat_k(pk[i])
            cv = [cos(p_v, flat_v(r[1])) for r in reps]; ck = [cos(p_k, flat_k(r[0])) for r in reps]
            cen_v = flat_v(torch.stack([r[1] for r in reps]).mean(0)); cen_k = flat_k(torch.stack([r[0] for r in reps]).mean(0))
            truth = V.derivable(fam, prem)
            inv = invalid_facts(fam, fact, prem, said, truth)
            civ = [cos(p_v, flat_v(r_[1])) for r_ in (safe_encode(g) for g in inv) if r_ is not None]
            for key in (fam, "all"):
                for nkey in (nb_, "all"):
                    d = acc[(key, nkey)]
                    d["cos_best"].append(max(cv)); d["cos_centroid"].append(cos(p_v, cen_v))
                    d["delta"].append(cos(p_v, cen_v) - max(cv))
                    d["cos_true_sent"].append(cos(p_v, flat_v(vt[i]))); d["cos_true_fact"].append(cos(p_v, flat_v(encode_fact(fact)[1])))
                    if civ: d["cos_invalid_max"].append(max(civ)); d["cos_invalid_mean"].append(sum(civ) / len(civ))
                    d["cos_best_k"].append(max(ck)); d["cos_centroid_k"].append(cos(p_k, cen_k))
                    d["n"].append(n)
            if n >= 2 and len(steps_n2) < a.decode:
                steps_n2.append((i, fam, cands, fact, prem, said, reps, batch["win"][i].clone(), batch["M"], kt[i].clone(), vt[i].clone(), pk[i].clone(), pv[i].clone()))

print(f"\n== {a.run.split('/')[-1]} : cosinus P vs candidates (valeurs normalisées JEPA), {a.nb} batchs ==")
print(f"{'strate':10s} {'N':>5s} {'cos_best':>9s} {'centroid':>9s} {'Δ':>7s} {'true_sent':>9s} {'true_fact':>9s} {'inval_max':>9s} {'inval_mean':>10s} {'k_best':>7s} {'k_cent':>7s}")
out = {}
for fam in ("F1", "F2", "F3", "all"):
    for nk in ("1", "2", "3", "4+", "all"):
        d = acc.get((fam, nk))
        if not d: continue
        m = lambda key: sum(d[key]) / len(d[key]) if d.get(key) else float("nan")
        row = {key: m(key) for key in ("cos_best", "cos_centroid", "delta", "cos_true_sent", "cos_true_fact", "cos_invalid_max", "cos_invalid_mean", "cos_best_k", "cos_centroid_k")}
        row["N"] = len(d["n"]); out[f"{fam}|{nk}"] = row
        print(f"{fam+'|n='+nk:10s} {row['N']:5d} {row['cos_best']:9.3f} {row['cos_centroid']:9.3f} {row['delta']:+7.3f} {row['cos_true_sent']:9.3f} {row['cos_true_fact']:9.3f} {row['cos_invalid_max']:9.3f} {row['cos_invalid_mean']:10.3f} {row['cos_best_k']:7.3f} {row['cos_centroid_k']:7.3f}")

# ------------------------------------------------------------------ pass 2: decode centroid vs candidates, semantic entropy, fixed point
@torch.no_grad()
def decode(win_row, M, K, Vv, temperature=0.0, max_len=48):
    """Decode one idea from slots (K, Vv) given the real prefix tokens of win_row (first M tokens). Forced 'so'."""
    prefix = win_row[:M][None]
    toks = [FORCE]
    for s_ in range(1, max_len):
        win = torch.cat([prefix, torch.tensor([toks + [PAD]], device="cuda")], 1)
        e = model.O(win, M)
        logits, tr = model.dec(win, torch.tensor([s_ + 1], device="cuda"), M, K[None], Vv[None], e)
        if is_end[toks[-1]] or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)): break
        lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf")
        if temperature > 0:
            toks.append(int(torch.multinomial(F.softmax(lg / temperature, -1), 1)))
        else:
            toks.append(int(lg.argmax()))
    return toks


def _judge_one(fam, prem, said, toks):
    f, ante = V.parse_sentence(w(toks))
    truth = V.derivable(fam, prem)
    frontier = V.valid_next(fam, prem, list(said))
    if f is None: return "unparsed"
    if f in said: return "repeat"
    if f in frontier: return "valid"
    if f in set(prem): return "premise"
    if f in truth: return "premature"
    return "false"


def sem_entropy(fam, prem, said, K, Vv, win_row, M, n_samples=10):
    groups = collections.Counter()
    for _ in range(n_samples):
        f, _ = V.parse_sentence(w(decode(win_row, M, K, Vv, temperature=1.0)))
        groups[f] += 1
    ps = [c / n_samples for c in groups.values()]
    return -sum(p * math.log(p) for p in ps)


@torch.no_grad()
def fixed_point_gap(K, Vv, win_row, M):
    toks = decode(win_row, M, K, Vv)
    t = torch.tensor([toks], device="cuda")
    k2, v2 = model.E_ema(t, torch.tensor([len(toks)], device="cuda"))
    return 1 - cos(flat_v(Vv), flat_v(v2[0]))


ver = collections.defaultdict(collections.Counter); se = collections.defaultdict(list); fp = collections.defaultdict(list); calib = []
for (i, fam, cands, fact, prem, said, reps, win_row, M, kt_i, vt_i, pk_i, pv_i) in steps_n2:
    said_l = list(said)
    cen_k = F.normalize(torch.stack([r[0] for r in reps]).mean(0), dim=-1); cen_v = torch.stack([r[1] for r in reps]).mean(0)
    ver["centroid"][_judge_one(fam, prem, said_l, decode(win_row, M, cen_k, cen_v))] += 1
    for r in reps: ver["candidate"][_judge_one(fam, prem, said_l, decode(win_row, M, r[0], r[1]))] += 1
    ver["P"][_judge_one(fam, prem, said_l, decode(win_row, M, pk_i, pv_i))] += 1
    se["centroid"].append(sem_entropy(fam, prem, said_l, cen_k, cen_v, win_row, M))
    se["candidate"].append(sem_entropy(fam, prem, said_l, reps[0][0], reps[0][1], win_row, M))
    se["P"].append(sem_entropy(fam, prem, said_l, pk_i, pv_i, win_row, M)); se["ln_n"].append(math.log(len(cands)))
    fp["centroid"].append(fixed_point_gap(cen_k, cen_v, win_row, M)); fp["candidate"].append(fixed_point_gap(reps[0][0], reps[0][1], win_row, M)); fp["P"].append(fixed_point_gap(pk_i, pv_i, win_row, M))
    # calibration: mean pairwise cos between paraphrases of the true fact vs centroid-candidate cos
    _, _, ka, va = reps[0]
    fv = [flat_v(x) for x in va]
    pair = [cos(fv[x], fv[y]) for x in range(len(fv)) for y in range(x + 1, len(fv))]
    if pair: calib.append((sum(pair) / len(pair), sum(cos(flat_v(cen_v), flat_v(r[1])) for r in reps) / len(reps)))

print(f"\n== décodage (n>=2, {len(steps_n2)} étapes) : verdicts ==")
for key in ("candidate", "centroid", "P"):
    tot = sum(ver[key].values()); print(f"  {key:10s} " + "  ".join(f"{kk}={vv/tot:.2f}" for kk, vv in sorted(ver[key].items())) + f"   (n={tot})")
print("\n== entropie sémantique (10 tirages T=1, groupés par fait) ==")
for key in ("candidate", "centroid", "P", "ln_n"):
    print(f"  {key:10s} {sum(se[key])/len(se[key]):.3f}")
print("\n== écart au point fixe 1 - cos(z, E_ema(D(z))) ==")
for key in ("candidate", "centroid", "P"):
    print(f"  {key:10s} {sum(fp[key])/len(fp[key]):.3f}")
if calib:
    print(f"\n== calibration : cos moyen entre paraphrases d'un même fait {sum(c[0] for c in calib)/len(calib):.3f} ; cos centroïde↔candidates {sum(c[1] for c in calib)/len(calib):.3f} ==")
json.dump({"cos": out, "verdicts": {k: dict(v) for k, v in ver.items()}, "sem_entropy": {k: sum(v)/len(v) for k, v in se.items()},
           "fixed_point": {k: sum(v)/len(v) for k, v in fp.items()}, "calib": calib[:5]}, open(f"{a.run}/diag_averaging.json", "w"), indent=1)
