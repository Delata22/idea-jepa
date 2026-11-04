"""Exact entropy floor of the synthetic generator (nats / token).

The text -> generator-choices map is a bijection once we only count choices that are visible in the text
(e.g. the verb form is invisible in the passive, NP forms are invisible under a pronoun, the word-order
choice is invisible without a location/time phrase). So -log p(story | n_sentences) is computed exactly by
summing the log-probabilities of the visible choices, marginalising over the (partly invisible) cast.
The number of sentences is excluded because models do not score an end-of-story token.

python -m ideajepa.data.synthetic_entropy  -> prints the floor for the test_iid split (seed 2).
"""
from __future__ import annotations

import itertools
import math
import random

from . import synthetic as S

LOG = math.log


def cast_logprob(pairs, held):
    """log p(sequence of (agent, patient) pairs) marginalised over casts consistent with it."""
    ents = list(S.ENTITIES)
    used = set(x for p in pairs for x in p)
    n = len(ents)
    total = 0.0
    for size in (2, 3):
        if len(used) > size:
            continue
        others = [e for e in ents if e not in used]
        extra = size - len(used)
        p_size = 0.5
        # p(cast as a set) = 1 / C(n, size) (rng.sample then set), all sets equally likely
        p_set = 1.0 / math.comb(n, size)
        for add in itertools.combinations(others, extra):
            cast = list(used) + list(add)
            valid = [(a, b) for a in cast for b in cast if a != b and (a, b) not in held]
            if not valid:
                continue
            if any(p not in valid for p in pairs):
                continue
            total += p_size * p_set * (1.0 / len(valid)) ** len(pairs)
    return LOG(total)


def sentence_logprob(idea, surf, info, prev):
    lp = LOG(1 / len(S.VERBS))
    lp += LOG(0.5) + (LOG(1 / len(S.PLACES)) if idea.loc else 0.0)
    lp += (LOG(0.25) + LOG(1 / len(S.TIMES))) if idea.time else LOG(0.75)
    f, voice, order = info["forms"], info["voice"], info["order"]
    lp += LOG(0.2 if voice == "passive" else 0.8)
    if idea.loc or idea.time:
        front = surf.words and ("," in surf.words)
        lp += LOG(1 / 3 if front else 2 / 3)
        if front and idea.loc and idea.time:
            lp += LOG(0.5)
    words = surf.words

    def np_lp(ent, role, subject):
        cls = S.ENTITIES[ent]
        eligible = ent in prev and sum(S.ENTITIES[e] == cls for e in prev) == 1
        pron = (S.SUBJ[cls] if subject else S.OBJ[cls])
        role_words = [w for w, r in zip(words, surf.roles) if r == S.R[role]]
        is_pron = role_words[:1] == [pron] and len([w for w in role_words if w in S.ENTITIES]) == 0
        out = 0.0
        if eligible:
            p = 0.5 if subject else 0.3
            out += LOG(p if is_pron else 1 - p)
        if is_pron:
            return out
        out += LOG(1 / 3)
        form = f[role]
        if form == "medium":
            out += LOG(1 / (10 * 9))
        elif form == "long":
            out += LOG(1 / len(S.OWNERS)) + LOG(1 / (10 * 9))
        return out

    if voice == "passive":
        lp += np_lp(idea.patient, "patient", True) + np_lp(idea.agent, "agent", False)
    else:
        lp += np_lp(idea.agent, "agent", True) + np_lp(idea.patient, "patient", False)
        lp += LOG(1 / 3)
        if f["verb"] == "long":
            lp += LOG(1 / len(S.ADVERBS))
    if idea.loc:
        lp += LOG(1 / 3)
        if f["loc"] == "medium":
            lp += LOG(1 / len(S.PLACE_ADJ))
        elif f["loc"] == "long":
            lp += LOG(1 / (len(S.PLACE_ADJ) * (len(S.PLACE_ADJ) - 1)))
    if idea.time:
        lp += LOG(1 / 3)
        if f["time"] in ("medium", "long"):
            lp += LOG(1 / len(S.TIME_ADJ))
    return lp


def cast_logprob_comp(pairs, held, held_at):
    """test_comp: sentence `held_at` uses a pair drawn uniformly from the held-out pairs; the cast is then
    extended with its entities; the other sentences draw non-held pairs from the (extended) cast."""
    ents = list(S.ENTITIES)
    hp = pairs[held_at]
    others = [p for i, p in enumerate(pairs) if i != held_at]
    used_before = set(x for i, p in enumerate(pairs) if i < held_at for x in p)
    n = len(ents)
    total = 0.0
    for size in (2, 3):
        pool = [e for e in ents if e not in used_before]
        need = size - len(used_before)
        if need < 0:
            continue
        for add in itertools.combinations(pool, need):
            cast = list(used_before) + list(add)
            ext = list(dict.fromkeys(cast + [hp[0], hp[1]]))
            p = 0.5 / math.comb(n, size)
            ok = True
            for i, pr in enumerate(pairs):
                if i == held_at:
                    continue
                c = cast if i < held_at else ext
                valid = [(a, b) for a in c for b in c if a != b and (a, b) not in held]
                if pr not in valid:
                    ok = False
                    break
                p *= 1.0 / len(valid)
            if ok:
                total += p
    return math.log(total) + math.log(1 / len(held)) + math.log(1 / len(pairs))


def floor(seed=2, n=5000, comp=False):
    held = S.heldout_pairs()
    rng = random.Random(seed)
    tot_lp, tot_tok = 0.0, 0
    for _ in range(n):
        sents = S.sample_story(rng, held, comp)
        prev = []
        pairs = [(i.agent, i.patient) for i, _, _ in sents]
        if comp:
            held_at = next(k for k, pr in enumerate(pairs) if pr in held)
            lp = cast_logprob_comp(pairs, held, held_at)
        else:
            lp = cast_logprob(pairs, held)
        for idea, surf, info in sents:
            lp += sentence_logprob(idea, surf, info, prev)
            prev = [idea.agent, idea.patient]
            tot_tok += len(surf.words)
        tot_lp += lp
    return -tot_lp / tot_tok


if __name__ == "__main__":
    print(f"entropy floor test_iid: {floor():.4f} nats/token")
    print(f"entropy floor test_comp: {floor(seed=3, comp=True):.4f} nats/token")
