"""For repeated deductions in free generation: is P's predicted Z the same (P re-predicts the same idea)
or different (decoder maps different Z to the same sentence)?
usage: uv run python scripts/diag_repeats.py <ideajepa_run> --n 30"""
import sys, re, argparse, collections, statistics as st, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
sys.argv, saved = [sys.argv[0]], sys.argv  # avoid gen_eval's argparse if imported; we re-implement below
sys.argv = saved
from ideajepa.train import load_model

PAD, BOS, EOS = 0, 1, 2
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--n", type=int, default=30)
a = ap.parse_args()
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w = lambda toks: " ".join(vocab[t] for t in toks)
is_end = torch.zeros(len(vocab), dtype=torch.bool, device="cuda"); is_end[vocab.index(".")] = True
FORCE = vocab.index("so")
REL = {"mother": "parent", "father": "parent", "grandmother": "grandparent", "grandfather": "grandparent",
       "sister": "sibling", "brother": "sibling", "aunt": "pibling", "uncle": "pibling"}


def parse(sent):
    s = re.sub(r"^(so|then|this means that)\s+", "", sent.replace(" .", "").strip())
    for pat, fn in [(r"the (\w+) (?:is|ends up) in the (\w+)$", lambda m: ("objin", m[1], m[2])),
                    (r"(\w+) is the (\w+) of (\w+)$", lambda m: ("rel", m[1], REL.get(m[2]), m[3])),
                    (r"(\w+) 's (\w+) is (\w+)$", lambda m: ("rel", m[3], REL.get(m[2]), m[1])),
                    (r"the (\w+) box can be opened by (\w+)$", lambda m: ("canopen", m[2], m[1])),
                    (r"(\w+) can open the (\w+) box$", lambda m: ("canopen", m[1], m[2])),
                    (r"(\w+) gets the (\w+) key$", lambda m: ("gets", m[1], ("key", m[2]))),
                    (r"the (\w+) key goes to (\w+)$", lambda m: ("gets", m[2], ("key", m[1]))),
                    (r"(\w+) gets the (\w+)$", lambda m: ("gets", m[1], ("item", m[2]))),
                    (r"the (\w+) goes to (\w+)$", lambda m: ("gets", m[2], ("item", m[1])))]:
        m = re.match(pat, s)
        if m: return fn(m)
    return None


model = load_model(f"{a.run}/ckpt.pt")[0]


@torch.no_grad()
def gen(premises, n_ded, m_prefix=3, max_len=48):
    story = [list(p) for p in premises]; zs = []
    for j in range(n_ded):
        L = max(len(t) for t in story)
        tok = torch.full((len(story), L), PAD, dtype=torch.long)
        for i, t in enumerate(story): tok[i, :len(t)] = torch.tensor(t)
        k, v = model.E(tok.cuda(), torch.tensor([len(t) for t in story], device="cuda"))
        n = len(story); di = torch.arange(n + 1, device="cuda")[None].clone(); di[0, n] = 0
        batch = {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device="cuda"), "idea_pos": torch.tensor([n], device="cuda")}
        pk, pv = model.predict(k, v, batch); zs.append(pv[0].float().clone())
        hist = [BOS] + [t for idea in story for t in idea]; p = hist[-m_prefix:]
        prefix = torch.full((1, m_prefix), PAD, dtype=torch.long); prefix[0, m_prefix - len(p):] = torch.tensor(p); prefix = prefix.cuda()
        toks = [FORCE]
        for s_ in range(1, max_len):
            win = torch.cat([prefix, torch.tensor([toks + [PAD]], device="cuda")], 1)
            e = model.O(win, m_prefix); logits, tr = model.dec(win, torch.tensor([s_ + 1], device="cuda"), m_prefix, pk, pv, e)
            if is_end[toks[-1]] or ("stop_logit" in tr and bool(tr["stop_logit"][0, s_] > 0)): break
            lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf"); toks.append(int(lg.argmax()))
        story.append(toks)
    return story[len(premises):], zs


def zcos(z1, z2):  # slot-wise cosine weighted by norms
    n1, n2 = z1.norm(dim=-1), z2.norm(dim=-1)
    c = F.cosine_similarity(z1, z2, dim=-1)
    return float((c * n1 * n2).sum() / (n1 * n2).sum().clamp_min(1e-6))


rep, rep_dist, nonrep, first_vs_ref = [], [], [], []
ex = []
per = collections.Counter()
for doc in docs:
    fam = doc.get("family")
    if per[fam] >= a.n: continue
    prem = [i["tokens"] for i in doc["ideas"] if i.get("depth", -1) < 0]
    ded = [i for i in doc["ideas"] if i.get("depth", -1) >= 0]
    if len(ded) < 2: continue
    per[fam] += 1
    ideas, zs = gen(prem, len(ded))
    facts = [parse(w(x)) for x in ideas]
    seenat = {}
    for j, f in enumerate(facts):
        if f is None: continue
        if f in seenat:
            i = seenat[f]
            rep.append(zcos(zs[i], zs[j]))
            rep_dist.append(j - i)
            if len(ex) < 4: ex.append((fam, i, j, w(ideas[i]), w(ideas[j]), rep[-1]))
        else:
            seenat[f] = j
    for j in range(1, len(zs)):
        if facts[j] is not None and facts[j - 1] is not None and facts[j] != facts[j - 1]:
            nonrep.append(zcos(zs[j - 1], zs[j]))

print(f"répétitions : {len(rep)} ; cos(Z_i, Z_j) des doublons : moyenne {st.mean(rep):.3f} médiane {st.median(rep):.3f} min {min(rep):.3f}")
print(f"idées consécutives distinctes ({len(nonrep)}) : cos moyen {st.mean(nonrep):.3f} médiane {st.median(nonrep):.3f}")
print(f"distance entre doublons (en idées) : {collections.Counter(rep_dist)}")
for e in ex: print("  ex:", e)
