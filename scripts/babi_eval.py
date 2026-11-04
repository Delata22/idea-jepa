"""bAbI evaluation (notes §34.6): exact match of the answer word extracted from the generated deduction sentence, and the
rate of out-of-format outputs, per task, for Idea-JEPA models (P predicts the idea after the question, greedy decoding with
"so" forced) and token-level references B0 / B3 (greedy continuation of "<story> <question> so" up to ".").
Format expected (scripts/prepare_babi.py templates): qa1/qa2 "so X is in the <loc> ." ; qa3 "so the O was in the <loc> before
the L ." The answer = the word after "in the"; out of format = no "in the <word>" or the sentence does not end with ".".
usage: python scripts/babi_eval.py <run> [<run> ...] [--split test|val] [--data data/babi/qa1_1k] [--n N] [--json out.json]
(--data defaults to the run's own data.path)
Every run also gets <run>/babi_outputs_<split>.jsonl (--no_outputs to skip): one line per example with the question, the
supporting facts (sentences), the expected answer, the generated sentence, correct / error class, number of ideas.
Error classes (qa2 / qa3, answer in format but wrong; notes §35): a_initial_place (where the object was first picked up),
b_earlier_carrier_place (a place a carrier visited earlier while the object existed), c_other_actor_place (where another
actor is at the end), d_other_place_in_story, e_place_not_in_story. Out-of-format outputs are counted apart.
Length bins (ideas before the answer): "71-100" (beyond every qa2-10k training story, which stop at 70) and "101+" (qa3).
"""
import argparse
import collections
import json
import os
import sys

import torch

sys.path.insert(0, "src")
from ideajepa.model.ideajepa import IdeaJEPA
from ideajepa.train import load_model

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--split", default="test")
ap.add_argument("--data", default=None)
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--json", default=None)
ap.add_argument("--show", type=int, default=3)
ap.add_argument("--no_outputs", action="store_true")
ap.add_argument("--zero_mem", action="store_true", help="ablation: P's delta memories read a ZERO state (S = 0) at inference")
ap.add_argument("--out_suffix", default="", help="suffix of the outputs file name")
ap.add_argument("--swap_mem", action="store_true",
                help="ablation (notes §35.16): the answer block of P reads, in every memory layer, the state S of ANOTHER story "
                     "(the next one of the evaluated set, at its own answer position) instead of its own")
a = ap.parse_args()
DEV = os.environ.get("IDEAJEPA_DEVICE", "cuda")


MOVE = {"moved", "went", "journeyed", "travelled"}
TAKE = {"got", "picked", "grabbed", "took"}
DROP = {"dropped", "left", "discarded", "put"}
BINS = [(0, 15, "<=15"), (16, 30, "16-30"), (31, 50, "31-50"), (51, 70, "51-70"), (71, 100, "71-100"), (101, 10 ** 6, "101+")]


def simulate(statements):
    """bAbI world state from the statements: object -> (initial place, set of earlier places of its carriers),
    and the final place of every actor."""
    pos, holder, obj_loc, first, carrier_hist = {}, {}, {}, {}, collections.defaultdict(set)
    for s in statements:
        w = [x for x in s if x not in (".",)]
        if len(w) < 3:
            continue
        who, verb = w[0], w[1]
        if verb in MOVE:
            loc = w[-1]
            for o, h in holder.items():
                if h == who:
                    carrier_hist[o].add(pos.get(who)); obj_loc[o] = loc
            pos[who] = loc
        elif verb in TAKE:
            o = w[-1] if w[-1] != "there" else w[-2]
            holder[o] = who; obj_loc[o] = pos.get(who)
            first.setdefault(o, pos.get(who))
        elif verb in DROP:
            o = w[-1] if w[-1] != "there" else w[-2]
            holder[o] = None
    return first, carrier_hist, pos


def object_moves(statements, obj):
    """qa3 (notes §35.35): number of moves of the object = moves of a person while carrying it."""
    holder, n = None, 0
    for s in statements:
        w = [x for x in s if x not in (".", "there", "back", "up", "down")]
        if len(w) < 3:
            continue
        if w[1] in MOVE and w[0] == holder:
            n += 1
        elif w[1] in TAKE and w[-1] == obj:
            holder = w[0]
        elif w[1] in DROP and w[-1] == obj:
            holder = None
    return n


def parse(words):
    if not words or words[-1] != ".":
        return None
    for i in range(len(words) - 2):
        if words[i] == "in" and words[i + 1] == "the":
            return words[i + 2]
    return None


def run_one(run):
    model, cfg, _ = load_model(f"{run}/ckpt.pt", device=DEV)
    model.eval()
    path = a.data or cfg["data"]["path"]
    data = torch.load(f"{path}/{a.split}.pt", weights_only=False)
    docs, vocab = data["docs"], data["vocab"]
    if a.n:
        docs = docs[:a.n]
    w2i = {x: i for i, x in enumerate(vocab)}
    FORCE, DOT = w2i["so"], w2i["."]
    w = lambda t: [vocab[x] for x in t]
    is_im = isinstance(model, IdeaJEPA)
    M = cfg["model"].get("m_prefix", 3)

    def enc_tokens(ideas):
        L = max(len(t) for t in ideas)
        tok = torch.full((len(ideas), L), PAD, dtype=torch.long)
        for i, t in enumerate(ideas):
            tok[i, :len(t)] = torch.tensor(t)
        return tok.to(DEV), torch.tensor([len(t) for t in ideas], device=DEV)

    mems = list(getattr(model.P, "mems", [])) if is_im else []
    R_ = model.P.r if is_im else 0

    def mem_states(ctx):
        """Delta-rule state S [H, dv, dk] of every memory layer after the writes of the ideas before the answer block."""
        tok, lens = enc_tokens(ctx)
        k, v = model.E(tok, lens)
        n = len(ctx)
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        for m_ in mems:
            m_.capture, m_.read_override = True, None
        model.P(k, v, di, role=torch.zeros(n, dtype=torch.long, device=DEV), tokens=model._p_tokens(k))
        out = []
        for m_ in mems:
            m_.capture = False
            _, kk = m_.last_qk; vv, beta = m_.last_v, m_.last_beta
            S = torch.zeros(kk.shape[1], vv.shape[-1], kk.shape[-1], device=DEV)
            for t in range(n * R_):
                kt, vt, bt = kk[0, :, t], vv[0, :, t], beta[0, :, t, None, None]
                S = S - bt * torch.einsum("hvk,hk,hj->hvj", S, kt, kt) + bt * torch.einsum("hv,hk->hvk", vt, kt)
            out.append(S)
        return out

    last = {}

    def gen_im(ctx, max_len=24, support=(), foreign=None):
        tok, lens = enc_tokens(ctx)
        k, v = model.E(tok, lens)
        n = len(ctx)
        if a.zero_mem:
            for m_ in dict.fromkeys(getattr(model.P, "mems", [])):
                m_.read_override = [(0, 0, 10 ** 9, torch.zeros(m_.h, m_.dh, m_.dh, device=DEV))]
        if foreign is not None:
            for m_, S_ in zip(mems, foreign):
                m_.read_override = [(0, n * R_, R_, S_)]
        di = torch.arange(n + 1, device=DEV)[None].clone(); di[0, n] = 0
        pk, pv = model.predict(k, v, {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device=DEV),
                                      "idea_pos": torch.tensor([n], device=DEV),
                                      "idea_role": torch.zeros(n, dtype=torch.long, device=DEV),
                                      # ORACLE variants only (p_word_kv: support): the supporting facts of the question
                                      "idea_support": torch.tensor([j in set(support) for j in range(n)], device=DEV)})
        for m_ in mems:
            m_.read_override = None
        last["pk"], last["pv"] = pk, pv
        hist = [BOS] + [t for idea in ctx for t in idea]; p = hist[-M:]
        pre = torch.full((1, M), PAD, dtype=torch.long); pre[0, M - len(p):] = torch.tensor(p); pre = pre.to(DEV)
        toks = [FORCE]
        for s_ in range(1, max_len):
            win = torch.cat([pre, torch.tensor([toks + [PAD]], device=DEV)], 1)
            logits, tr = model.dec(win, torch.tensor([s_ + 1], device=DEV), M, pk, pv, model.O(win, M))
            if toks[-1] == DOT or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)):
                break
            lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf")
            toks.append(int(lg.argmax()))
        return toks

    def gen_gpt(ctx, max_len=24):
        seq = torch.tensor([[BOS] + [t for p in ctx for t in p] + [FORCE]], device=DEV)
        out = [FORCE]
        for _ in range(max_len):
            lg = model.run(seq[:, -model.pos.num_embeddings:])[:, -1].float()
            lg[:, [PAD, BOS, EOS]] = float("-inf")
            nxt = int(lg.argmax())
            out.append(nxt); seq = torch.cat([seq, torch.tensor([[nxt]], device=DEV)], 1)
            if nxt == DOT:
                break
        return out

    cnt = collections.defaultdict(collections.Counter)
    ex = []
    outf = None if a.no_outputs else open(f"{run}/babi_outputs_{a.split}{a.out_suffix}.jsonl", "w")
    with torch.no_grad():
        for di_, d in enumerate(docs):
            ctx = [i["tokens"] for i in d["ideas"] if i["step"] != "deduction"]
            foreign = None
            if a.swap_mem and is_im:
                d2 = docs[(di_ + 1) % len(docs)]
                foreign = mem_states([i["tokens"] for i in d2["ideas"] if i["step"] != "deduction"])
            out = gen_im(ctx, support=d.get("support", []), foreign=foreign) if is_im else gen_gpt(ctx)
            jd_ans = None
            if is_im:   # JEPA distance of P's answer idea to E_ema(answer sentence), as in the training loss
                tgt = [i["tokens"] for i in d["ideas"] if i["step"] == "deduction"][0]
                kt, vt = model.E_ema(*enc_tokens([tgt]))
                pk_, pv_ = last["pk"], last["pv"]
                jd_ans = float(((pk_ - kt).pow(2).sum(-1) + (torch.nn.functional.normalize(pv_, dim=-1)
                                                            - torch.nn.functional.normalize(vt, dim=-1)).pow(2).sum(-1)).mean())
                cnt[d["family"]]["jepa_ans_sum"] += jd_ans
            ans = parse(w(out))
            ok = ans == d["answer"]
            n_id = len(ctx)
            bin_ = next(b for lo, hi, b in BINS if lo <= n_id <= hi)
            mv_key = None
            if d["family"] == "qa3":   # errors by number of moves of the object (1 / 2 / 3+)
                qw = w(ctx[-1]); o_ = qw[qw.index("the") + 1] if "the" in qw else None
                nm = object_moves([w(t) for t in ctx[:-1]], o_)
                mv_key = d["family"] + "|moves" + ("0" if nm == 0 else "1" if nm == 1 else "2" if nm == 2 else "3+")
            for key in (d["family"], d["family"] + "|" + bin_) + ((mv_key,) if mv_key else ()):
                c = cnt[key]
                c["n"] += 1; c["correct"] += ok; c["out_of_format"] += ans is None
            kind = None
            if ans is None:
                kind = "out_of_format"
            elif not ok and d["family"] in ("qa2", "qa3"):
                q = w(ctx[-1])
                obj = q[q.index("the") + 1] if "the" in q else None
                stmts = [w(t) for t in ctx[:-1]]
                first, hist, actor_pos = simulate(stmts)
                places = {x[-2] for x in stmts if len(x) >= 2}
                kind = ("a_initial_place" if obj in first and ans == first[obj] else
                        "b_earlier_carrier_place" if obj in hist and ans in hist[obj] else
                        "c_other_actor_place" if ans in actor_pos.values() else
                        "d_other_place_in_story" if ans in places else "e_place_not_in_story")
                cnt[d["family"]]["err_" + kind] += 1
            elif not ok:
                kind = "wrong"
            if outf is not None:
                sup = [" ".join(w(ctx[j])) for j in d.get("support", []) if j < len(ctx) - 1]
                outf.write(json.dumps({"i": di_, "n_ideas": n_id, "len_bin": bin_, "question": " ".join(w(ctx[-1])),
                                       "support": sup, "support_idx": d.get("support", []), "answer": d["answer"],
                                       "generated": " ".join(w(out)), "pred": ans, "ok": ok, "err": kind,
                                       "jepa_ans": jd_ans}) + "\n")
            if len(ex) < a.show and ans != d["answer"]:
                ex.append({"question": " ".join(w(ctx[-1])), "answer": d["answer"], "generated": " ".join(w(out))})
    if outf is not None:
        outf.close()
    res = {t: {**dict(c), "acc": c["correct"] / c["n"], "err": 1 - c["correct"] / c["n"], "oof": c["out_of_format"] / c["n"]}
           for t, c in cnt.items()}
    for t, r_ in res.items():
        if "jepa_ans_sum" in r_:
            r_["L_jepa_answer"] = r_["jepa_ans_sum"] / r_["n"]
    print(f"{run} [{path} {a.split}]: " + "  ".join(f"{t} acc {r['acc']:.3f} (out of format {r['oof']:.3f}, n={r['n']}"
                                                  + (f", L_jepa answers {r['L_jepa_answer']:.3f})" if "L_jepa_answer" in r else ")")
                                                  for t, r in sorted(res.items()) if "|" not in t), flush=True)
    print("    by idea count: " + "  ".join(f"{t.split('|')[1]} {r['acc']:.2f} (n={r['n']})"
                                                    for t, r in sorted(res.items()) if "|" in t))
    errs = {k: v for t, r in res.items() if "|" not in t for k, v in r.items() if k.startswith("err_")}
    if errs:
        print("    errors: " + "  ".join(f"{k[4:]} {v}" for k, v in sorted(errs.items())))
    for e in ex:
        print(f"    ✗ {e['question']} → '{e['generated']}' (answer {e['answer']})")
    del model
    return {"run": run, "data": path, "split": a.split, "results": res, "examples": ex}


out = [run_one(r) for r in a.runs]
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
