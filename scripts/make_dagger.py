"""Offline DAgger (one iteration) for synthetic v2.
For N training stories: the model generates its own deduction chain from the premises (first token forced to a
connector). For every prefix of that chain, the generator's oracle (closure_step_*) gives the valid next facts
given what the model *actually* said; one is sampled and rendered. Each (premises + model prefix + oracle
deduction) becomes a training document where only the oracle deduction carries loss (train_w = 1) and the
model's own deductions are context only (train_w = 0).
usage: uv run python scripts/make_dagger.py <ckpt_run> --n 20000 --out data/reasoning_dagger"""
import sys, re, random, argparse, shutil, collections, torch
from pathlib import Path
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.data import reasoning as R
from ideajepa.data import verifier as V

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--out", default="data/reasoning_dagger")
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
src = Path("data/reasoning"); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
train = torch.load(src / "train.pt", weights_only=False)
vocab, docs = train["vocab"], train["docs"]
w2i = {w: i for i, w in enumerate(vocab)}
w = lambda toks: " ".join(vocab[t] for t in toks)
model, cfg, _ = load_model(f"{a.run}/ckpt.pt")
is_end = torch.zeros(len(vocab), dtype=torch.bool, device="cuda"); is_end[w2i["."]] = True
FORCE = w2i["so"]
REL = {"mother": "parent", "father": "parent", "grandmother": "grandparent", "grandfather": "grandparent",
       "sister": "sibling", "brother": "sibling", "aunt": "pibling", "uncle": "pibling"}


def parse(sent):  # single source of truth: src/ideajepa/data/verifier.py
    return V.parse_sentence(sent)[0]


@torch.no_grad()
def gen_batch(all_stories, n_steps_each, m_prefix=3, max_len=48):
    """all_stories: list of lists of idea token lists (premises). Generates n_steps_each[b] deductions for
    story b (batched over the stories still active at each step)."""
    n_prem = [len(s) for s in all_stories]
    for step in range(max(n_steps_each)):
        active = [b for b in range(len(all_stories)) if step < n_steps_each[b]]
        if not active: break
        stories = [all_stories[b] for b in active]
        N = len(stories)
        rows, di = [], torch.full((N, max(len(s) for s in stories) + 1), -1, dtype=torch.long)
        for b, s in enumerate(stories):
            for i, idea in enumerate(s):
                di[b, i] = len(rows); rows.append(idea)
            di[b, len(s)] = 0  # placeholder for the idea to predict
        L = max(len(t) for t in rows)
        tok = torch.full((len(rows), L), PAD, dtype=torch.long)
        for i, t in enumerate(rows): tok[i, :len(t)] = torch.tensor(t)
        k, v = model.E(tok.cuda(), torch.tensor([len(t) for t in rows], device="cuda"))
        batch = {"doc_index": di.cuda(), "idea_doc": torch.arange(N, device="cuda"),
                 "idea_pos": torch.tensor([len(s) for s in stories], device="cuda")}
        pk, pv = model.predict(k, v, batch)
        prefix = torch.full((N, m_prefix), PAD, dtype=torch.long)
        for b, s in enumerate(stories):
            hist = [BOS] + [t for idea in s for t in idea]; p = hist[-m_prefix:]
            prefix[b, m_prefix - len(p):] = torch.tensor(p)
        prefix = prefix.cuda()
        toks = torch.full((N, max_len), PAD, dtype=torch.long, device="cuda"); toks[:, 0] = FORCE
        done = torch.zeros(N, dtype=torch.bool, device="cuda"); length = torch.full((N,), max_len, device="cuda")
        for s_ in range(1, max_len):
            win = torch.cat([prefix, toks[:, :s_ + 1]], 1)
            e = model.O(win, m_prefix)
            logits, tr = model.dec(win, torch.full((N,), s_ + 1, device="cuda"), m_prefix, pk, pv, e)
            stop = is_end[toks[:, s_ - 1]] | (tr["stop_logit"][:, s_] > 0 if "stop_logit" in tr else False)
            newly = stop & ~done; length = torch.where(newly, torch.full_like(length, s_), length); done |= stop
            if done.all(): break
            lg = logits[:, s_].float(); lg[:, [PAD, BOS, EOS]] = float("-inf")
            nxt = lg.argmax(-1); toks[:, s_] = torch.where(done, torch.full_like(nxt, PAD), nxt)
        for b in range(N):
            stories[b].append(toks[b, :int(length[b])].tolist())
    return all_stories


def render(fact, family):
    s, _ = R.render_fact(fact, rng, deduction=True)
    return [w2i[x] for x in s.w], s.r, s.t


def closure(family, stated, n_prem):
    return V._closure(family, stated, n_prem)


def derivable(family, prem_facts):
    """All facts derivable from the TRUE premises (fixpoint of the closure)."""
    return V.derivable(family, prem_facts)


new_docs, stats = [], collections.Counter()
MAXI = cfg["data"]["max_ideas"]          # P's idea-position table: stories longer than this are skipped
sel = [d for d in docs if len(d["ideas"]) <= MAXI][:a.n]
for start in range(0, len(sel), a.bs):
    chunk = sel[start:start + a.bs]
    prem = [[i["tokens"] for i in d["ideas"] if i["step"] == "premise"] for d in chunk]
    nded = [sum(i["step"] == "deduction" for i in d["ideas"]) for d in chunk]
    gen = gen_batch([list(p) for p in prem], nded)
    for d, p, nd, g in zip(chunk, prem, nded, gen):
        fam = d["family"]; prem_ideas = [i for i in d["ideas"] if i["step"] == "premise"]
        prem_facts = [tuple(i["fact"]) if not isinstance(i["fact"], tuple) else i["fact"] for i in prem_ideas]
        gen_ideas = g[len(p):len(p) + nd]
        gen_facts = [parse(w(x)) for x in gen_ideas]
        truth = derivable(fam, prem_facts)
        for t in range(nd):
            # DAgger expert (v3), judged from the TRUE state with the strict verifier:
            #  established (derivation basis) = premises + prefix statements VALID AT THE TIME they were said;
            #    a premature statement is true but was not licensed, so nothing is derived from it
            #    (v2 derived from any true statement, which rewarded skipping steps);
            #  said (excluded from targets) = every true prefix statement (valid or premature), so a fact
            #    the model already wrote is never taught again.
            try:
                frontier, _ = V.expert_targets(fam, prem_facts, [w(x) for x in gen_ideas[:t]])
            except Exception:
                stats["oracle_error"] += 1; break
            if not frontier:
                stats["empty_frontier_skipped"] += 1; continue
            f = rng.choice(sorted(frontier, key=str))
            toks, roles, types = render(f, fam)
            ideas = []
            for i in prem_ideas:
                ideas.append({**i, "train_w": 1.0})
            for x, gf in zip(gen_ideas[:t], gen_facts[:t]):
                ideas.append({"tokens": x, "roles": [0] * len(x), "types": [0] * len(x), "idea_id": 10_000_000 + (hash(str(gf)) % 1_000_000),
                              "fact": gf, "step": "deduction", "b_eff": 0, "candidates": [], "restated": False,
                              "distractor": False, "depth": t, "train_w": 0.0})
            ideas.append({"tokens": toks, "roles": roles, "types": types, "idea_id": 20_000_000 + (hash(str(f)) % 1_000_000),
                          "fact": f, "step": "deduction", "b_eff": len(frontier), "candidates": sorted(frontier, key=str),
                          "restated": False, "distractor": False, "depth": t, "train_w": 1.0})
            new_docs.append({"family": fam, "ideas": ideas, "dagger": True})
            stats["docs"] += 1; stats["deviated" if gen_facts[:t] != [tuple(i["fact"]) for i in d["ideas"] if i["step"] == "deduction"][:t] else "on_reference"] += 1
    if (start // a.bs) % 20 == 0:
        print(f"{start + len(chunk)}/{len(sel)} stories -> {stats['docs']} dagger docs", flush=True)

print("stats:", dict(stats))
mixed = docs + new_docs
torch.save({"docs": mixed, "vocab": vocab}, out / "train.pt")
for f in ("val.pt", "test_iid.pt", "test_depth.pt", "test_entities.pt", "meta.pt"):
    if (src / f).exists(): shutil.copy(src / f, out / f)
print(f"saved {len(mixed)} docs ({len(new_docs)} dagger) to {out}")
