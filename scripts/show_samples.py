"""Qualitative samples on synthetic v2: give the premises of a val story, let models write the deductions.
Idea-JEPA generates idea by idea (explicit idea boundaries); GPT baselines token by token (split at '.').
usage: uv run python scripts/show_samples.py <ideajepa_run> <b3_run> <b0_run> [n_stories] [close_rule]"""
import sys, torch
sys.path.insert(0, "src")
from ideajepa.train import load_model
from ideajepa.eval.generate import generate_idea, _pick
from ideajepa.model.ideajepa import IdeaJEPA

PAD, BOS, EOS = 0, 1, 2
runs = [a for a in sys.argv[1:4]]
n_stories = int(sys.argv[4]) if len(sys.argv) > 4 else 3
close_rule = sys.argv[5] if len(sys.argv) > 5 else "stop"
FORCE = sys.argv[6] if len(sys.argv) > 6 else None   # e.g. "so": force the first token of each generated idea
data = torch.load("data/reasoning/val.pt", weights_only=False)
vocab, docs = data["vocab"], data["docs"]
w = lambda toks: " ".join(vocab[t] for t in toks)
is_end = torch.zeros(len(vocab), dtype=torch.bool, device="cuda")
is_end[vocab.index(".")] = True


@torch.no_grad()
def gen_ideajepa(model, premises, n_ded, m_prefix=3, max_len=48):
    story = [list(p) for p in premises]
    r, d = model.cfg["r"], model.cfg.get("d_slot", model.cfg["d"])
    for j in range(n_ded):
        rows = story
        L = max(len(t) for t in rows)
        tok = torch.full((len(rows), L), PAD, dtype=torch.long)
        for i, t in enumerate(rows):
            tok[i, :len(t)] = torch.tensor(t)
        k, v = model.E(tok.cuda(), torch.tensor([len(t) for t in rows], device="cuda"))
        n = len(rows)
        di = torch.arange(n + 1, device="cuda")[None].clone(); di[0, n] = 0  # placeholder for the idea to predict
        batch = {"doc_index": di, "idea_doc": torch.zeros(1, dtype=torch.long, device="cuda"),
                 "idea_pos": torch.tensor([n], device="cuda")}
        pk, pv = model.predict(k, v, batch)
        hist = [BOS] + [t for idea in story for t in idea]
        p = hist[-m_prefix:]
        prefix = torch.full((1, m_prefix), PAD, dtype=torch.long); prefix[0, m_prefix - len(p):] = torch.tensor(p)
        if FORCE is None:
            ideas, rho, length = generate_idea(model, prefix.cuda(), pk, pv, max_len, is_end, close_rule,
                                               banned=[PAD, BOS, EOS])
            story.append(ideas[0]); continue
        # forced first token, then greedy with the chosen closing rule
        toks = [vocab.index(FORCE)]
        prefix = prefix.cuda()
        for s_ in range(1, max_len):
            win = torch.cat([prefix, torch.tensor([toks + [PAD]], device="cuda")], 1)
            n = torch.tensor([s_ + 1], device="cuda")
            e = model.O(win, m_prefix)
            logits, tr = model.dec(win, n, m_prefix, pk, pv, e)
            if close_rule == "stop" and bool(tr["stop_logit"][0, s_] > 0):
                break
            if is_end[toks[-1]]:
                break
            lg = logits[0, s_].float(); lg[[PAD, BOS, EOS]] = float("-inf")
            toks.append(int(lg.argmax()))
        story.append(toks)
    return story[len(premises):]


@torch.no_grad()
def gen_gpt(model, premises, n_ded, max_tokens=120):
    seq = torch.tensor([[BOS] + [t for p in premises for t in p]], device="cuda")
    out, periods = [], 0
    if FORCE is not None:
        f = vocab.index(FORCE); seq = torch.cat([seq, torch.tensor([[f]], device="cuda")], 1); out.append(f)
    for _ in range(max_tokens):
        logits = model.run(seq[:, -model.pos.num_embeddings:])[:, -1].float()
        logits[:, [PAD, BOS, EOS]] = float("-inf")
        nxt = _pick(logits, 0.0, None, None)
        seq = torch.cat([seq, nxt[:, None]], 1); out.append(int(nxt))
        if bool(is_end[nxt]):
            periods += 1
            if periods >= n_ded:
                break
            if FORCE is not None:
                f = vocab.index(FORCE); seq = torch.cat([seq, torch.tensor([[f]], device="cuda")], 1); out.append(f)
    # split at periods
    ideas, cur = [], []
    for t in out:
        cur.append(t)
        if bool(is_end[t]):
            ideas.append(cur); cur = []
    if cur:
        ideas.append(cur)
    return ideas


models = {name: load_model(f"{run}/ckpt.pt")[0] for name, run in zip(["V-full", "B3", "B0"], runs)}
shown = {}
for doc in docs:
    fam = doc.get("family")
    if fam is None:
        continue
    if fam in shown or len(shown) >= n_stories:
        if len(shown) >= n_stories:
            break
        continue
    prem = [i["tokens"] for i in doc["ideas"] if i.get("depth", -1) < 0]
    ded = [i for i in doc["ideas"] if i.get("depth", -1) >= 0]
    if any(i.get("restated") for i in ded) or len(ded) < 3:
        continue
    shown[fam] = True
    print(f"\n=== famille {fam} ===")
    print("PRÉMISSES :")
    for p in prem:
        print("   " + w(p))
    print(f"CHAÎNE DE RÉFÉRENCE ({len(ded)} déductions, b_eff par étape {[i['b_eff'] for i in ded]}) :")
    for i in ded:
        print("   " + w(i["tokens"]))
    first_cands = i0 = ded[0].get("candidates")
    if first_cands:
        print(f"   candidats valides à l'étape 1 : {first_cands}")
    for name, m in models.items():
        if isinstance(m, IdeaJEPA):
            ideas = gen_ideajepa(m, prem, len(ded))
        else:
            ideas = gen_gpt(m, prem, len(ded))
        print(f"{name} :")
        print("   " + "  |  ".join(w(x) for x in ideas))
