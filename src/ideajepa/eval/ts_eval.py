"""Stage-2 (TinyStories) evaluation: samples + long-range entity coherence.

python -m ideajepa.eval.ts_eval runs/stage2/ts_v_full_s0 [...] --n 20 --tokens 320
Writes <run>/samples.json and <run>/coherence.json.

Entity coherence on long generations (>= 300 tokens), computed from surface text:
  * name_persistence : fraction of character names introduced in the first third that reappear later;
  * late_new_names   : number of distinct names first introduced in the last two thirds (lower = fewer
                       characters appearing out of nowhere);
  * pronoun_agreement: for each sentence whose subject is a gendered name, fraction of he/she pronouns in the
                       next sentence that agree with that gender (TinyStories names lexicon below).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from ..model.ideajepa import IdeaJEPA
from ..train import load_model
from ..utils.common import ROOT
from .generate import generate_gpt, generate_story_idea

FEMALE = {"lily", "sue", "mia", "anna", "sara", "sarah", "lucy", "emma", "amy", "kate", "jane", "molly", "daisy",
          "ellie", "lisa", "mary", "sophie", "zoe", "rose", "ella", "grace", "mom", "mommy", "girl", "sally",
          "jenny", "katie", "ruby", "bella", "lola", "susie", "abby", "julie", "pam"}
MALE = {"tom", "ben", "timmy", "tim", "max", "jack", "sam", "bob", "joe", "leo", "billy", "jimmy", "mike",
        "john", "dad", "daddy", "boy", "spot", "tommy", "johnny", "peter", "jake", "danny", "toby", "freddy",
        "bobby", "ted", "roger", "fred", "rex"}
NOT_NAMES = {"once", "one", "the", "a", "an", "he", "she", "they", "it", "i", "you", "we", "then", "but", "and",
             "so", "when", "after", "suddenly", "yes", "no", "oh", "every", "at", "in", "on", "his", "her",
             "their", "there", "this", "that", "what", "why", "let", "can", "look", "thank", "wow", "please",
             "come", "mom", "dad", "mommy", "daddy", "hello", "hi", "okay", "ok", "sure", "some", "all"}


def names_in(text):
    """Capitalised tokens that are not sentence-initial (approximate character names) + known names."""
    out = []
    for m in re.finditer(r"(?<![.!?\"]\s)(?<!^)\b([A-Z][a-z]+)\b", text):
        w = m.group(1).lower()
        if w not in NOT_NAMES:
            out.append((m.start(), w))
    for m in re.finditer(r"\b([A-Z][a-z]+)\b", text):
        w = m.group(1).lower()
        if (w in FEMALE or w in MALE) and w not in NOT_NAMES and (m.start(), w) not in out:
            out.append((m.start(), w))
    return sorted(set(out))


def coherence(text):
    n = len(text)
    names = names_in(text)
    first = {}
    for pos, w in names:
        first.setdefault(w, pos)
    early = [w for w, p in first.items() if p < n / 3]
    persist = [any(p > n / 3 and w2 == w for p, w2 in names) for w in early]
    late_new = sum(1 for w, p in first.items() if p >= n / 3)
    sents = re.split(r"(?<=[.!?])\s+", text)
    agree, tot = 0, 0
    for s1, s2 in zip(sents, sents[1:]):
        m = re.match(r"\W*([A-Z][a-z]+)", s1)
        if not m:
            continue
        w = m.group(1).lower()
        g = "f" if w in FEMALE else "m" if w in MALE else None
        if g is None:
            continue
        for p in re.findall(r"\b(he|she|He|She)\b", s2):
            tot += 1
            agree += (p.lower() == "she") == (g == "f")
    return {"name_persistence": float(np.mean(persist)) if persist else np.nan,
            "late_new_names": late_new, "pronoun_agreement": agree / tot if tot else np.nan,
            "n_pronoun_checks": tot}


def run(run_dir: Path, n=20, n_tokens=320, temperature=0.8, top_k=50):
    model, cfg, _ = load_model(run_dir / "ckpt.pt")
    meta = torch.load(ROOT / cfg["data"]["path"] / "meta.pt", weights_only=False)
    tok = Tokenizer.from_file(meta["tokenizer"])
    is_end = torch.tensor(meta["is_end"], device="cuda")
    mc = cfg["model"]
    if isinstance(model, IdeaJEPA):
        stories = generate_story_idea(model, n_ideas=64, is_end=is_end, max_idea_len=mc["max_idea_len"],
                                      m_prefix=mc["m_prefix"], n_stories=n, close_rule="punct",
                                      temperature=temperature, top_k=top_k, max_tokens=n_tokens)
        seqs = [[t for idea in s for t in idea][:n_tokens] for s in stories]
    else:
        seqs = generate_gpt(model, n_tokens, n_stories=n, temperature=temperature, top_k=top_k,
                            is_end=is_end, m_prefix=mc["m_prefix"], max_idea_len=mc["max_idea_len"])
    texts = [tok.decode(s) for s in seqs]
    coh = [coherence(t) for t in texts]
    agg = {k: float(np.nanmean([c[k] for c in coh])) for k in coh[0]}
    (run_dir / "samples.json").write_text(json.dumps(texts, indent=1))
    (run_dir / "coherence.json").write_text(json.dumps({"mean": agg, "per_story": coh}, indent=1))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=320)
    a = ap.parse_args()
    for r in a.runs:
        print(r, run(Path(r), a.n, a.tokens))


if __name__ == "__main__":
    main()
