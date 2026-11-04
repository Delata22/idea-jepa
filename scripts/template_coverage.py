"""Template coverage (chantier 3 bis, spec §2): for every family and role, are the templates the model meets at
EVALUATION a subset of the templates it sees in TRAINING, on each of its inputs and targets?
Template of a sentence: deduction -> connector/clause form (so/f0, so/f1, then/f0 … tmt/f1, since/f0, since/f1);
premise -> prem/f0, prem/f1.
Training view (the recipe's Collator on documents, loss switches honoured):
  P_input      rows P reads and writes into its memory: idea_tok (plain: the story's sentence; surfaces: the main role
               form; data.p_input_surface: pin_tok)
  E_input      everything E reads: idea_tok + enc_tok (loss.cross_encode) + pos_tok (loss.w_nce) + pin_tok
  dec_target   what the decoder is trained to write: main rows (+ alt rows under loss.multi_form)
Evaluation view: P_input and E_input = the story's own sentences (reference context of decode_verify / refctx; the
premises in free generation); dec_target = the "so" forms (generation forces "so").
usage: uv run python scripts/template_coverage.py [configs ...] [--docs 400]   (exit code 1 if a check fails)
"""
import argparse, collections, sys
import torch
sys.path.insert(0, "src")
from ideajepa.data import reasoning as R
from ideajepa.data.dataset import Collator
from ideajepa.utils.common import load_config

DEFAULT = ["configs/v2_pwin_delta.yaml", "configs/v3_stepA.yaml", "configs/v3_stepA_psurf.yaml",
           "configs/v3_stepB_w3.yaml", "configs/v3_stepB_w3_since.yaml"]


def template_of(words, fact, step):
    fact = tuple(tuple(x) if isinstance(x, list) else x for x in fact)
    if step == "deduction" or (words and words[0] in ("so", "then", "this", "since")):
        if words[0] == "since":
            clause = words[words.index(",") + 1:]
            for form in (0, 1):
                if clause == R.render_clause(fact, None, form)[0].w + ["."]:
                    return f"since/f{form}"
            return "since/?"
        for t in R.DEDUCTION_TEMPLATES:
            if words == R.render_with_template(fact, t).w:
                return f"{['so', 'then', 'tmt'][(t - 2) // 2]}/f{(t - 2) % 2}"
        return "ded/?"
    for form, w in enumerate(R.premise_forms(fact)):
        if words == w:
            return f"prem/f{form}"
    return "prem/?"


def views(cfg, docs, vocab, w2i):
    d = cfg["data"]; lc = cfg["loss"]
    if d.get("compound"):                                      # step 1: both views read the converted documents
        from ideajepa.data.reasoning import split_compound
        docs = [split_compound(dd, d["compound"], vocab) for dd in docs]
    coll = Collator(m_prefix=cfg["model"]["m_prefix"], surfaces=d.get("surfaces", False), w2i=w2i, seed=0,
                    p_input_surface=d.get("p_input_surface"), since_surfaces=d.get("since_surfaces"))
    train = collections.defaultdict(set); ev = collections.defaultdict(set)
    words = lambda row, n=None: [vocab[t] for t in row.tolist() if t != 0][:n]
    for s in range(0, len(docs), 16):
        chunk = docs[s:s + 16]
        ideas = [i for dd in chunk for i in dd["ideas"]]
        for rep in range(3):                                   # random surface draws
            b = coll(chunk)
            M = b["M"]
            for r, idea in enumerate(ideas):
                fam, st, f = next(dd["family"] for dd in chunk if idea in dd["ideas"]), idea["step"], idea["fact"]
                key = lambda part: (fam, st, part)
                p_row = b["pin_tok"][r] if "pin_tok" in b else b["idea_tok"][r]
                train[key("P_input")].add(template_of(words(p_row), f, st))
                e_rows = [b["idea_tok"][r], p_row]
                if lc.get("cross_encode") and "enc_tok" in b:
                    e_rows.append(b["enc_tok"][r])
                for e in e_rows:
                    train[key("E_input")].add(template_of(words(e), f, st))
                train[key("dec_target")].add(template_of(words(b["win"][r, M:]), f, st))
            if lc.get("w_nce", 0) > 0 and "pos_tok" in b:
                for j, r in enumerate(b["pos_idea"].tolist()):
                    idea = ideas[r]; fam = next(dd["family"] for dd in chunk if idea in dd["ideas"])
                    fa = idea["fact"]
                    wd = words(b["pos_tok"][j])
                    # a positive may be another fact (the position a movement implies): label it with its own parse
                    t = template_of(wd, fa, idea["step"])
                    if not t.endswith("?"):
                        train[(fam, idea["step"], "E_input")].add(t)
            if lc.get("multi_form") and "alt_win" in b:
                for j, r in enumerate(b["alt_idea"].tolist()):
                    idea = ideas[r]; fam = next(dd["family"] for dd in chunk if idea in dd["ideas"])
                    train[(fam, idea["step"], "dec_target")].add(template_of(words(b["alt_win"][j, M:]), idea["fact"], idea["step"]))
        for dd in chunk:
            for idea in dd["ideas"]:
                t = template_of([vocab[x] for x in idea["tokens"]], idea["fact"], idea["step"])
                for part in ("P_input", "E_input"):
                    ev[(dd["family"], idea["step"], part)].add(t)
                if idea["step"] == "deduction":
                    ev[(dd["family"], "deduction", "dec_target")] |= {"so/f0", "so/f1"}
    return train, ev


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("configs", nargs="*"); ap.add_argument("--docs", type=int, default=400)
    a = ap.parse_args()
    data = torch.load("data/reasoning/val.pt", weights_only=False)
    vocab = data["vocab"]; w2i = {x: i for i, x in enumerate(vocab)}
    docs = [dd for dd in data["docs"][:a.docs] if dd["family"] in ("F1", "F2", "F3")]
    failed = False
    for c in a.configs or DEFAULT:
        cfg = load_config(c)
        train, ev = views(cfg, docs, vocab, w2i)
        bad = []
        for key in sorted(ev):
            missing = sorted(t for t in ev[key] - train.get(key, set()) if not t.endswith("?"))
            if missing:
                bad.append(f"{key[0]} {key[1]:9s} {key[2]:10s} missing in training: {', '.join(missing)}")
        print(f"{c}: {'OK' if not bad else 'FAIL'}")
        for line in bad:
            print("   " + line)
        failed |= bool(bad)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
