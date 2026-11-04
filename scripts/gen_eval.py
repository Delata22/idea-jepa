"""Free-generation reasoning check on synthetic v2: premises given, models write the deductions
(first token forced to a connector), a checker parses each sentence into a fact tuple and compares it
to the valid facts of the story (reference chain + all candidate facts along it).
usage: uv run python scripts/gen_eval.py <run> [<run> ...] --n 30 [--show 6]"""
import sys, os, re, json, argparse, collections, torch
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.eval.generate import _pick
from ideajepa.model.ideajepa import IdeaJEPA
from ideajepa.data import reasoning as R
import random
from ideajepa.data import verifier as V
from ideajepa.eval.samplers import idea_candidates
from ideajepa.model.flow_head import to_target
import torch.nn.functional as F

PAD, BOS, EOS = 0, 1, 2
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+"); ap.add_argument("--n", type=int, default=30); ap.add_argument("--show", type=int, default=6)
ap.add_argument("--force", default="so")
ap.add_argument("--zero_mem", action="store_true", help="diagnostic: silence the idea memory of P at inference")
ap.add_argument("--jsonl", default=None, help="write one line per (story, model) with every verdict")
ap.add_argument("--oracle", action="store_true", help="upper bound: replace P's idea by E_ema of a random VALID unsaid deduction")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rel_oracle", action="store_true", help="rel_cond models, diagnostic: after each generated idea, its "
                "(relation, pointer) input label is replaced by the generator rule's (antecedent in F2/F3, useful premise in F1) "
                "computed on the fact written, instead of the head's prediction")
ap.add_argument("--plan_mode", default="sample", choices=["sample", "argmax", "novelty", "novelty_far", "head_nonrep", "nonrep_uniform"],
                help="plan_cond models (notes §32.20): the plan idea o is drawn from the pointer head (T = 1) or its argmax; "
                     "novelty (§32.22): for every past idea j, Ẑ(j) = P(context, o = j); drop the j whose Ẑ(j) has cos > "
                     "--novelty_cos with an idea already in the context; draw uniformly among the others (novelty) or take the "
                     "one farthest from every idea said (novelty_far); none left: the least close")
# phase 2: where the next idea comes from, and best-of-K selection
ap.add_argument("--sampler", default="P", choices=["P", "flow", "khyp"])
ap.add_argument("--n_cand", type=int, default=1, help="flow samples per step (khyp: all K hypotheses are used)")
ap.add_argument("--select", default="first", choices=["first", "oracle", "novelty", "energy", "novelty+energy", "vote"],
                help="first: candidate 0 (P / 1st flow sample / best-scored hypothesis); oracle: first candidate whose "
                     "decoded sentence is valid (upper bound); novelty: first candidate whose idea is not cos > "
                     "--novelty_cos to an idea already said; energy: best score of the masked predictor (--energy_run); "
                     "novelty+energy: best energy among the novel candidates (all candidates if none is novel)")
ap.add_argument("--dec_temp", type=float, default=0.0,
                help="token sampling temperature of the decoder (0 = greedy); gives P distinct candidates for best-of-K")
ap.add_argument("--flow_steps", type=int, default=16)
ap.add_argument("--flow_temp", type=float, default=1.0)
ap.add_argument("--novelty_cos", type=float, default=0.9)
ap.add_argument("--log_cands", default=None, help="jsonl: per step, every candidate decoding, the greedy decoding of the "
                "same idea, the chosen one, with their verdicts given the model's own past deductions (vote analysis)")
ap.add_argument("--form", default="random", help="form_cond models: 'random' (uniform over the 6 deduction templates) "
                "or a fixed template id in 2..7; the forced first word follows the template's connector")
ap.add_argument("--snap", default="none", choices=["none", "context", "story", "valid"],
                help="DIAGNOSTIC ONLY (uses the world's facts): each candidate idea is replaced by E_ema of a real "
                     "paraphrase of its nearest fact (paraphrase-mean cosine). story = among all true deductions of the "
                     "story (valid now, already said, or not yet derivable): isolates the MODE CHOICE from landing "
                     "precision; valid = among the valid ones only (upper bound, cheating); context = HONEST: among every "
                     "well-formed fact of the family's schema built from the entities of the premises (no truth information)")
ap.add_argument("--energy_run", default=None, help="stage-A checkpoint with a masked idea predictor (e.g. runs/v2/v2h_stageA2_s0)")
a = ap.parse_args()
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w = lambda toks: " ".join(vocab[t] for t in toks)
is_end = torch.zeros(len(vocab), dtype=torch.bool, device=DEV); is_end[vocab.index(".")] = True
FORCE = vocab.index(a.force)

REL = {"mother": "parent", "father": "parent", "grandmother": "grandparent", "grandfather": "grandparent",
       "sister": "sibling", "brother": "sibling", "aunt": "pibling", "uncle": "pibling"}


def parse(sent):
    s = sent.replace(" .", "").strip()
    s = re.sub(r"^(so|then|this means that|since [^,]*,)\s+", "", s)
    m = re.match(r"the (\w+) (?:is|ends up) in the (\w+)$", s)
    if m: return ("objin", m.group(1), m.group(2))
    m = re.match(r"(\w+) is the (\w+) of (\w+)$", s)
    if m and m.group(2) in REL: return ("rel", m.group(1), REL[m.group(2)], m.group(3))
    m = re.match(r"(\w+) 's (\w+) is (\w+)$", s)
    if m and m.group(2) in REL: return ("rel", m.group(3), REL[m.group(2)], m.group(1))
    m = re.match(r"the (\w+) box can be opened by (\w+)$", s)
    if m: return ("canopen", m.group(2), m.group(1))
    m = re.match(r"(\w+) can open the (\w+) box$", s)
    if m: return ("canopen", m.group(1), m.group(2))
    m = re.match(r"(\w+) gets the (\w+) key$", s)
    if m: return ("gets", m.group(1), ("key", m.group(2)))
    m = re.match(r"the (\w+) key goes to (\w+)$", s)
    if m: return ("gets", m.group(2), ("key", m.group(1)))
    m = re.match(r"(\w+) gets the (\w+)$", s)
    if m: return ("gets", m.group(1), ("item", m.group(2)))
    m = re.match(r"the (\w+) goes to (\w+)$", s)
    if m: return ("gets", m.group(2), ("item", m.group(1)))
    return None


def flat_idea(k_, v_):
    """Flow/JEPA space [k ; v/|v|], flattened and L2-normalised, for cosines between ideas."""
    return F.normalize(to_target(k_, v_).flatten(1), dim=-1)


FORM_RNG = random.Random(a.seed + 7)


def pick_form(model):
    """form_cond models: the surface template of the next deduction (not part of the idea)."""
    if not model.cfg.get("form_cond", False):
        return None, FORCE
    t = FORM_RNG.choice(R.DEDUCTION_TEMPLATES) if a.form == "random" else int(a.form)
    first = w2i[R.CONNECT[(t - 2) // 2][0]]
    FORM_LOG.append(t)
    return torch.tensor([t], device=DEV), first


FORM_LOG, FORM_FOLLOW = [], collections.Counter()
SNAP_RNG = random.Random(a.seed + 11)
SNAP_STATS = collections.Counter()


REL_STATE = {"rel": None}   # step 2 (rel_cond models): relation predicted for the idea being decoded
REL_LOG = collections.Counter()


@torch.no_grad()
def decode_idea(model, prefix, pk, pv, m_prefix, max_len):
    form, first = pick_form(model)
    rk = model.dec_rel(REL_STATE["rel"])
    toks = [first]
    for s_ in range(1, max_len):
        win = torch.cat([prefix, torch.tensor([toks + [PAD]], device=DEV)], 1)
        e = model.O(win, m_prefix)
        n_ = torch.tensor([s_ + 1], device=DEV)
        logits, tr = (model.dec(win, n_, m_prefix, pk, pv, e, form=form, **rk) if form is not None
                      else model.dec(win, n_, m_prefix, pk, pv, e, **rk))
        if is_end[toks[-1]] or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)): break
        lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf")
        if a.dec_temp > 0:  # sampled decoding: the diversity source of best-of-K for a deterministic P
            toks.append(int(torch.multinomial(torch.softmax(lg / a.dec_temp, -1), 1, generator=DEC_GEN)))
        else:
            toks.append(int(lg.argmax()))
    return toks


def encode_ideas(model, ideas):
    L = max(len(t) for t in ideas)
    tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
    for i, t in enumerate(ideas): tok[i, :len(t)] = torch.tensor(t)
    return tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV)


_SNAP_REP = {}


@torch.no_grad()
def fact_rep(model, f):
    """Paraphrase-mean representation of a fact (flow space), for --snap."""
    key = (id(model), f)
    if key not in _SNAP_REP:
        sents = []
        for t in R.DEDUCTION_TEMPLATES:
            sents.append([w2i[x] for x in R.render_with_template(f, t).w])
        tok, lens = encode_ideas(model, sents)
        k_, v_ = model.E_ema(tok, lens)
        _SNAP_REP[key] = (F.normalize(flat_idea(k_, v_).mean(0), dim=0), (tok, lens, k_, v_))
    return _SNAP_REP[key]


@torch.no_grad()
def snap_candidates(model, CK, CV, ctx, said_txt):
    """--snap: replace every candidate by E_ema of a random real paraphrase of its nearest fact."""
    fam, prem = ctx
    said = [v["fact"] for v in V.judge(fam, prem, said_txt) if v["verdict"].startswith("valid")]
    if a.snap == "valid":
        pool = sorted(V.valid_next(fam, prem, said), key=str)
    elif a.snap == "story":
        pool = sorted(V.derivable(fam, prem), key=str)
    else:
        pool = V.schema_facts(fam, prem)
    if not pool:
        return CK, CV
    reps = torch.stack([fact_rep(model, f)[0] for f in pool])
    CK, CV = CK.clone(), CV.clone()
    for c in range(CK.shape[1]):
        f = pool[int((flat_idea(CK[:, c], CV[:, c]) @ reps.T).argmax())]
        _, (tok, lens, k_, v_) = fact_rep(model, f)
        j = SNAP_RNG.randrange(len(tok))
        CK[:, c], CV[:, c] = k_[j:j + 1], v_[j:j + 1]
        SNAP_STATS["snapped"] += 1
        SNAP_STATS["to_valid_now"] += f in V.valid_next(fam, prem, said)
    return CK, CV


@torch.no_grad()
def energy_scores(story, cands):
    """Masked-predictor energy (stage-A model): cos between its prediction for a masked idea at the end of the
    story and E_ema(candidate). Higher = the candidate fits this place better."""
    tok, lens = encode_ideas(ENERGY, story + cands)
    k, v = ENERGY.E(tok, lens); kt, vt = ENERGY.E_ema(tok, lens)
    n, C = len(story), len(cands)
    di = torch.arange(n + 1, device=DEV)[None].repeat(C, 1); di[:, n] = n + torch.arange(C, device=DEV)
    masked = torch.zeros(n + C, dtype=torch.bool, device=DEV); masked[n:] = True
    scores = []
    for c in range(C):  # one document per candidate (shared story rows would be overwritten in a joint call)
        _, pv_ = ENERGY.PM(k, v, di[c:c + 1], masked)
        pf = F.normalize(F.normalize(pv_[n + c], dim=-1).flatten(), dim=0)
        tf = F.normalize(F.normalize(vt[n + c], dim=-1).flatten(), dim=0)
        scores.append(float(pf @ tf))
    return scores


@torch.no_grad()
def gen_ideajepa(model, premises, n_ded, m_prefix=3, max_len=48, oracle=None, ctx=None):
    """oracle=(family, premise_facts): the idea fed to the decoder is E_ema(random valid unsaid deduction).
    ctx=(family, premise_facts): used by --select oracle (verifier)."""
    story = [list(p) for p in premises]
    rels = [(0, 0)] * len(premises)          # rel_cond: (relation, pointer offset) of every idea of the story
    for j in range(n_ded):
        tok, lens = encode_ideas(model, story)
        k, v = model.E(tok, lens)
        n = len(story)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        batch = {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                 "idea_pos": torch.tensor([n], device=DEV),
                 "idea_role": (torch.arange(n, device=DEV) >= len(premises)).long()}   # p_role_emb models
        if model.cfg.get("rel_cond", False):
            batch["rel_id"] = torch.tensor([x[0] for x in rels], device=DEV)
            batch["rel_off"] = torch.tensor([x[1] for x in rels], device=DEV)
            model.predict(k, v, batch)
            lr, lp = model.rel_logits()
            r_, o_ = int(lr[0].argmax()), int(lp[0].argmax())
            rels.append((r_, o_)); REL_LOG[("none", "justified_by", "sequence")[r_]] += 1
            REL_STATE["rel"] = torch.tensor([r_], device=DEV)
        hist = [BOS] + [t for idea in story for t in idea]; p = hist[-m_prefix:]
        prefix = torch.full((1, m_prefix), PAD, dtype=torch.long); prefix[0, m_prefix - len(p):] = torch.tensor(p)
        prefix = prefix.to(DEV)
        if oracle is not None:
            pk, pv = model.predict(k, v, batch)
            fam_, prem_facts = oracle
            said = [V.parse_sentence(w(x))[0] for x in story[len(premises):]]
            said = [f for f in said if f is not None]
            cands = sorted(V.valid_next(fam_, prem_facts, said), key=str)
            if cands:
                f = ORACLE_RNG.choice(cands)
                sent, _ = R.render_fact(f, ORACLE_RNG, deduction=True)
                otok = torch.tensor([[w2i[x] for x in sent.w]], device=DEV)
                with torch.no_grad():
                    pk, pv = model.E_ema(otok, torch.tensor([otok.shape[1]], device=DEV))
            story.append(decode_idea(model, prefix, pk, pv, m_prefix, max_len))
            continue
        if a.plan_mode.startswith("novelty") and model.cfg.get("plan_cond", False):
            batch["plan_next"] = torch.tensor([novelty_plan(model, k, v, batch, story)], device=DEV)
        elif a.plan_mode in ("head_nonrep", "nonrep_uniform") and model.cfg.get("plan_cond", False):
            batch["plan_next"] = torch.tensor([nonrep_plan(model, k, v, batch, story, len(premises))], device=DEV)
        CK, CV = idea_candidates(model, k, v, batch, a.sampler, a.n_cand, a.flow_steps, a.flow_temp, FLOW_GEN)
        if a.sampler == "P" and a.n_cand > 1:   # best-of-K for P: same idea, K sampled decodings (needs --dec_temp)
            CK, CV = CK.expand(-1, a.n_cand, -1, -1), CV.expand(-1, a.n_cand, -1, -1)
        if a.snap != "none" and ctx is not None:
            CK, CV = snap_candidates(model, CK, CV, ctx, [w(x) for x in story[len(premises):]])
        outs = [decode_idea(model, prefix, CK[:, c], CV[:, c], m_prefix, max_len) for c in range(CK.shape[1])]
        choice = 0
        if len(outs) > 1 and a.select == "oracle" and ctx is not None:
            said_txt = [w(x) for x in story[len(premises):]]
            ok = [V.judge(ctx[0], ctx[1], said_txt + [w(o)])[-1]["verdict"].startswith("valid") for o in outs]
            choice = ok.index(True) if any(ok) else 0
        elif len(outs) > 1 and a.select == "vote":
            # notes §34.3: drop the samples that restate a DEDUCTION already written (premises never count), then the
            # most frequent fact among the others (tie: the first sampled); none left: the first sample
            said = {V.parse_sentence(w(x))[0] for x in story[len(premises):]}
            facts = [V.parse_sentence(w(o))[0] for o in outs]
            keep = [c for c, f in enumerate(facts) if f is not None and f not in said]
            if keep:
                cnt = collections.Counter(facts[c] for c in keep)
                top = max(cnt.values())
                choice = next(c for c in keep if cnt[facts[c]] == top)
        elif len(outs) > 1 and a.select == "novelty+energy":
            keep = list(range(len(outs)))
            if len(story) > len(premises):
                ct, cl = encode_ideas(model, outs); ck, cv = model.E_ema(ct, cl)
                st, sl = encode_ideas(model, story[len(premises):]); sk, sv = model.E_ema(st, sl)
                sim = (flat_idea(ck, cv) @ flat_idea(sk, sv).T).max(1).values
                keep = [c for c in keep if float(sim[c]) <= a.novelty_cos] or keep
            sc = energy_scores(story, [outs[c] for c in keep])
            choice = keep[max(range(len(keep)), key=lambda j: sc[j])]
        elif len(outs) > 1 and a.select == "novelty" and len(story) > len(premises):
            ct, cl = encode_ideas(model, outs); ck, cv = model.E_ema(ct, cl)
            st, sl = encode_ideas(model, story[len(premises):]); sk, sv = model.E_ema(st, sl)
            sim = (flat_idea(ck, cv) @ flat_idea(sk, sv).T).max(1).values        # [C] max cos to a said idea
            fresh = (sim <= a.novelty_cos).nonzero()[:, 0]
            choice = int(fresh[0]) if len(fresh) else int(sim.argmin())
        elif len(outs) > 1 and a.select == "energy":
            sc = energy_scores(story, outs)
            choice = max(range(len(outs)), key=lambda c: sc[c])
        SELECT_STATS["steps"] += 1; SELECT_STATS["changed"] += choice != 0
        if model.cfg.get("form_cond", False) and FORM_LOG:
            # each decoded candidate appended one requested template; the chosen one is at -len(outs)+choice
            t_req = FORM_LOG[-len(outs) + choice]
            f_, _ = V.parse_sentence(w(outs[choice]))
            FORM_FOLLOW["steps"] += 1
            FORM_FOLLOW["parsed"] += f_ is not None
            FORM_FOLLOW["followed"] += f_ is not None and " ".join(R.render_with_template(f_, t_req).w) == w(outs[choice])
        if a.log_cands and ctx is not None:
            said_txt = [w(x) for x in story[len(premises):]]
            verd = lambda o: V.judge(ctx[0], ctx[1], said_txt + [w(o)])[-1]["verdict"]
            T0 = a.dec_temp
            a.dec_temp = 0.0
            greedy = decode_idea(model, prefix, CK[:, 0], CV[:, 0], m_prefix, max_len)
            a.dec_temp = T0
            CAND_LOG.append({"story_key": CUR_STORY[0], "step": len(story) - len(premises), "family": ctx[0],
                             "cands": [w(o) for o in outs], "cand_verdicts": [verd(o) for o in outs],
                             "greedy": w(greedy), "greedy_verdict": verd(greedy), "chosen": choice,
                             "said": said_txt, "premises": [w(p) for p in premises]})
        story.append(outs[choice])
        if a.rel_oracle and model.cfg.get("rel_cond", False) and ctx is not None:
            fam_, pfacts = ctx
            facts_ = list(pfacts) + [V.parse_sentence(w(x))[0] for x in story[len(premises):]]
            f_ = facts_[-1]
            ptr = None
            if f_ is not None and fam_ in ("F2", "F3"):
                an = R.antecedent(fam_, f_, [x for x in facts_[:-1] if x is not None])
                ptr = max((i for i in range(len(facts_) - 1) if facts_[i] == an), default=None) if an is not None else None
            elif f_ is not None and fam_ == "F1" and f_[0] == "objin":
                up = R.f1_useful_premises(list(pfacts))
                ptr = up.get(f_[1])
            if ptr is not None:
                rels[-1] = (1, len(story) - 1 - ptr)
    return story[len(premises):]


@torch.no_grad()
def gen_gpt(model, premises, n_ded, max_tokens=150):
    seq = torch.tensor([[BOS] + [t for p in premises for t in p] + [FORCE]], device=DEV)
    out, ideas, cur = [FORCE], [], [FORCE]
    for _ in range(max_tokens):
        logits = model.run(seq[:, -model.pos.num_embeddings:])[:, -1].float(); logits[:, [PAD, BOS, EOS]] = float("-inf")
        nxt = int(_pick(logits, 0.0, None, None)); seq = torch.cat([seq, torch.tensor([[nxt]], device=DEV)], 1); cur.append(nxt)
        if bool(is_end[nxt]):
            ideas.append(cur); cur = [FORCE]
            if len(ideas) >= n_ded: break
            seq = torch.cat([seq, torch.tensor([[FORCE]], device=DEV)], 1)
    return ideas


ORACLE_RNG = random.Random(a.seed)
FLOW_GEN = torch.Generator(device=DEV).manual_seed(a.seed)
DEC_GEN = torch.Generator(device=DEV).manual_seed(a.seed + 3)
SELECT_STATS = collections.Counter()
NOVELTY_STATS = collections.Counter()
NOVELTY_RNG = __import__("random").Random(a.seed)


@torch.no_grad()
def nonrep_plan(model, k, v, batch, story, n_prem):
    """§32.28 (corrected novelty): drop the j whose Ẑ(j) has cos > --novelty_cos with a DEDUCTION already written
    (premises do not count); head_nonrep: first j by the head's probability among the kept; nonrep_uniform: uniform."""
    n = len(story)
    ded = story[n_prem:]
    zded = None
    if ded:
        tok, lens = encode_ideas(model, ded)
        _, ve = model.E_ema(tok, lens)
        zded = F.normalize(F.normalize(ve.float(), dim=-1).flatten(1), dim=-1)
    mode0 = model.plan_mode
    model.plan_mode = "argmax"
    model.predict(k, v, {kk: vv for kk, vv in batch.items() if kk != "plan_next"})
    model.plan_mode = mode0
    order = [int(j) for j in model.plan_logits[0, n].float().argsort(descending=True).tolist() if j < n]
    keep = []
    for j in range(n):
        if zded is None:
            keep.append(j); continue
        _, pv = model.predict(k, v, {**batch, "plan_next": torch.tensor([j], device=DEV)})
        zj = F.normalize(F.normalize(pv.float(), dim=-1).flatten(1), dim=-1)[0]
        if float((zded @ zj).max()) <= a.novelty_cos:
            keep.append(j)
    NOVELTY_STATS["steps"] += 1; NOVELTY_STATS["kept"] += len(keep); NOVELTY_STATS["none_left"] += not keep
    if not keep:
        return order[0]
    if a.plan_mode == "nonrep_uniform":
        return NOVELTY_RNG.choice(keep)
    ks = set(keep)
    return next(j for j in order if j in ks)


@torch.no_grad()
def novelty_plan(model, k, v, batch, story):
    """§32.22: choice of the plan idea by novelty of the conditioned prediction (see --plan_mode)."""
    n = len(story)
    tok, lens = encode_ideas(model, story)
    _, ve = model.E_ema(tok, lens)
    zctx = F.normalize(F.normalize(ve.float(), dim=-1).flatten(1), dim=-1)          # ideas already in the context
    best = []
    for j in range(n):
        _, pv = model.predict(k, v, {**batch, "plan_next": torch.tensor([j], device=DEV)})
        zj = F.normalize(F.normalize(pv.float(), dim=-1).flatten(1), dim=-1)[0]
        best.append(float((zctx @ zj).max()))
    keep = [j for j in range(n) if best[j] <= a.novelty_cos]
    NOVELTY_STATS["steps"] += 1; NOVELTY_STATS["kept"] += len(keep); NOVELTY_STATS["none_left"] += not keep
    if not keep:
        return min(range(n), key=lambda j: best[j])
    if a.plan_mode == "novelty_far":
        return min(keep, key=lambda j: best[j])
    return NOVELTY_RNG.choice(keep)
ENERGY = load_model(f"{a.energy_run}/ckpt.pt", device=DEV)[0] if a.energy_run else None
if a.select in ("energy", "novelty+energy"):
    assert ENERGY is not None and getattr(ENERGY, "PM", None) is not None, "--select energy needs --energy_run <stage-A run>"
w2i = {x: i for i, x in enumerate(vocab)}
models = {r.rstrip("/").split("/")[-1]: load_model(f"{r}/ckpt.pt", device=DEV)[0] for r in a.runs}
for _m in models.values():
    if getattr(_m, "cfg", {}).get("plan_cond", False):
        _m.plan_mode = a.plan_mode
        _m.plan_gen = torch.Generator(device=DEV).manual_seed(a.seed)
if a.zero_mem:
    for _n, _m in models.items():
        for mem in getattr(getattr(_m, "P", None), "mems", []) or []:
            mem.forward = (lambda x, _mem=mem: torch.zeros_like(x))
    print("[zero_mem] idea memories silenced at inference")
names = list(models)
stats = {n: collections.defaultdict(lambda: [0, 0]) for n in names}  # fam -> [valid, total]
errk = {n: collections.Counter() for n in names}  # error kinds: repeat / wrong / unparsed
examples = []
per_fam = collections.Counter()
jsonl_rows = []
CAND_LOG = []
CUR_STORY = [None]
for doc in docs:
    fam = doc.get("family")
    if per_fam[fam] >= a.n: continue
    prem = [i["tokens"] for i in doc["ideas"] if i.get("depth", -1) < 0]
    ded = [i for i in doc["ideas"] if i.get("depth", -1) >= 0]
    if len(ded) < 2: continue
    per_fam[fam] += 1
    valid = {tuple(i["fact"]) if not isinstance(i["fact"], tuple) else i["fact"] for i in ded}
    for i in ded:
        for c in i.get("candidates", []): valid.add(c)
    outs = {}
    for n, m in models.items():
        REL_LOG.clear()
        CUR_STORY[0] = [docs.index(doc), n]
        ideas = (gen_ideajepa(m, prem, len(ded), oracle=((fam, V.doc_premises(doc)) if a.oracle else None),
                                ctx=(fam, V.doc_premises(doc)))
                 if isinstance(m, IdeaJEPA) else gen_gpt(m, prem, len(ded)))
        # exact verdicts from the TRUE state (src/ideajepa/data/verifier.py): valid / premature / repeat /
        # premise / false / unparsed. The old check (fact in reference chain + candidates) ignored order and
        # rejected valid facts outside the reference trajectory.
        verdicts = V.judge(fam, V.doc_premises(doc), [w(x) for x in ideas])
        if a.jsonl:
            jsonl_rows.append({"story": docs.index(doc), "family": fam, "model": n, "n_deductions": len(ded),
                               "sampler": a.sampler, "n_cand": a.n_cand, "select": a.select,
                               "n_valid": sum(v["verdict"].startswith("valid") for v in verdicts),
                               "verdicts": [v["verdict"] for v in verdicts],
                               "sentences": [v["sentence"] for v in verdicts],
                               **({"rel_pred": dict(REL_LOG)} if REL_LOG else {})})
        res = []
        for x, vd in zip(ideas, verdicts):
            ok = vd["verdict"].startswith("valid")
            if not ok: errk[n][vd["verdict"]] += 1
            res.append((w(x), ok)); stats[n][fam][0] += ok; stats[n][fam][1] += 1
        outs[n] = res
    examples.append((fam, prem, ded, outs))

print("\nTaux de déductions valides en génération libre (connecteur forcé), par famille :")
for n in names:
    row = "  ".join(f"{f}: {v}/{t} ({100*v/max(t,1):.0f}%)" for f, (v, t) in sorted(stats[n].items()))
    tot = sum(v for v, t in stats[n].values()); tt = sum(t for v, t in stats[n].values())
    print(f"  {n:24s} {row}   total {tot}/{tt} ({100*tot/max(tt,1):.0f}%)")

print("Nature des erreurs :")
for n in names: print(f"  {n:24s} {dict(errk[n])}")

# examples where the first model fails and another succeeds, and the reverse
first = names[0]; shown = 0
print(f"\n--- Cas où {first} se trompe et un autre modèle a juste ---")
for fam, prem, ded, outs in examples:
    bad = [i for i, (s, ok) in enumerate(outs[first]) if not ok]
    if not bad: continue
    other_ok = [n for n in names[1:] if all(ok for s, ok in outs[n])]
    if not other_ok: continue
    print(f"[{fam}] référence : " + " | ".join(w(i["tokens"]) for i in ded))
    for n in names: print(f"   {n:24s}: " + " | ".join(("✓ " if ok else "✗ ") + s for s, ok in outs[n]))
    shown += 1
    if shown >= a.show: break
shown = 0
print(f"\n--- Cas où {first} a juste et un autre se trompe ---")
for fam, prem, ded, outs in examples:
    if not all(ok for s, ok in outs[first]): continue
    others_bad = [n for n in names[1:] if any(not ok for s, ok in outs[n])]
    if not others_bad: continue
    print(f"[{fam}] référence : " + " | ".join(w(i["tokens"]) for i in ded))
    for n in names: print(f"   {n:24s}: " + " | ".join(("✓ " if ok else "✗ ") + s for s, ok in outs[n]))
    shown += 1
    if shown >= a.show // 2: break

if a.log_cands:
    with open(a.log_cands, "w") as fh:
        for r in CAND_LOG:
            fh.write(json.dumps(r) + "\n")
if a.jsonl:
    with open(a.jsonl, "w") as fh:
        for r in jsonl_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(jsonl_rows)} rows to {a.jsonl}")

if SELECT_STATS["steps"]:
    if NOVELTY_STATS["steps"]:  # (novelty*, head_nonrep, nonrep_uniform)
        print(f"[plan {a.plan_mode}] ideas j kept per step {NOVELTY_STATS['kept'] / NOVELTY_STATS['steps']:.2f}, "
              f"steps with none left {NOVELTY_STATS['none_left']}/{NOVELTY_STATS['steps']}")
    print(f"[select={a.select}, sampler={a.sampler}, n_cand={a.n_cand}] candidate changed at "
          f"{SELECT_STATS['changed']}/{SELECT_STATS['steps']} steps")
if FORM_FOLLOW["steps"]:
    print(f"[form={a.form}] requested template followed exactly: {FORM_FOLLOW['followed']}/{FORM_FOLLOW['steps']} "
          f"({FORM_FOLLOW['followed'] / FORM_FOLLOW['steps']:.1%}; parsed {FORM_FOLLOW['parsed']})")
if SNAP_STATS["snapped"]:
    print(f"[snap={a.snap}] {SNAP_STATS['snapped']} candidates snapped; nearest fact valid at that step: "
          f"{SNAP_STATS['to_valid_now']}/{SNAP_STATS['snapped']} ({SNAP_STATS['to_valid_now'] / SNAP_STATS['snapped']:.1%})")
