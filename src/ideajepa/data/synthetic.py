"""Template-based synthetic story generator with full ground truth.

Each story is a list of sentences ("ideas"). Each idea is the tuple
(agent, action, patient, location|None, time|None); every surface formulation
of the same tuple (NP length, verb form, voice, word order, pronoun) shares the
same `idea_id`. Per word we store its role (agent/action/patient/loc/time/none)
and its type (function / content / modifier / pronoun).

Run as a script to build `data/synthetic/{train,val,test_iid,test_comp}.pt`.
"""
from __future__ import annotations

import argparse
import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROLES = ["none", "agent", "action", "patient", "loc", "time"]
TYPES = ["function", "content", "modifier", "pronoun"]
R = {r: i for i, r in enumerate(ROLES)}
T = {t: i for i, t in enumerate(TYPES)}
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# name -> pronoun class ("it", "he", "she")
ENTITIES = {
    "cat": "it", "dog": "it", "mouse": "it", "bird": "it", "rabbit": "it", "fox": "it",
    "horse": "it", "duck": "it", "frog": "it", "bear": "it", "owl": "it", "goat": "it",
    "boy": "he", "man": "he", "king": "he", "prince": "he",
    "girl": "she", "woman": "she", "queen": "she", "princess": "she",
}
ANIMAL_ADJ = ["small", "furry", "grey", "sleepy", "brown", "clever", "tiny", "fat", "shy", "wild"]
PERSON_ADJ = ["young", "tall", "kind", "brave", "old", "happy", "quiet", "little", "proud", "busy"]
OWNERS = ["neighbour", "baker", "sailor", "grandmother", "farmer"]
SUBJ = {"it": "it", "he": "he", "she": "she"}
OBJ = {"it": "it", "he": "him", "she": "her"}
# base -> (3sg, ing, past participle)
VERBS = {
    "chase": ("chases", "chasing", "chased"), "see": ("sees", "seeing", "seen"),
    "help": ("helps", "helping", "helped"), "follow": ("follows", "following", "followed"),
    "push": ("pushes", "pushing", "pushed"), "find": ("finds", "finding", "found"),
    "watch": ("watches", "watching", "watched"), "hug": ("hugs", "hugging", "hugged"),
    "call": ("calls", "calling", "called"), "meet": ("meets", "meeting", "met"),
    "visit": ("visits", "visiting", "visited"), "carry": ("carries", "carrying", "carried"),
    "feed": ("feeds", "feeding", "fed"), "catch": ("catches", "catching", "caught"),
    "pull": ("pulls", "pulling", "pulled"), "bite": ("bites", "biting", "bitten"),
    "kick": ("kicks", "kicking", "kicked"), "greet": ("greets", "greeting", "greeted"),
    "teach": ("teaches", "teaching", "taught"), "ignore": ("ignores", "ignoring", "ignored"),
}
ADVERBS = ["quickly", "slowly", "gently", "suddenly", "happily", "proudly"]
PLACES = ["garden", "park", "forest", "kitchen", "house", "barn", "field", "river",
          "school", "beach", "village", "castle"]
PLACE_ADJ = ["big", "quiet", "old", "sunny", "green", "dark", "busy", "empty"]
TIMES = ["morning", "evening", "night", "afternoon", "winter", "summer"]
TIME_ADJ = ["sunny", "cold", "quiet", "rainy", "warm", "long"]

FUNCTION_WORDS = ["the", "is", "by", "in", "one", "early", "late", "somewhere", ",", ".", "'s"]


def build_vocab() -> list[str]:
    words = list(FUNCTION_WORDS)
    words += list(ENTITIES) + ANIMAL_ADJ + PERSON_ADJ + OWNERS
    words += sorted(set(SUBJ.values()) | set(OBJ.values()))
    for forms in VERBS.values():
        words += list(forms)
    words += ADVERBS + PLACES + PLACE_ADJ + TIMES + TIME_ADJ
    seen, out = set(), []
    for w in SPECIALS + words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


@dataclass
class Idea:
    agent: str
    action: str
    patient: str
    loc: str | None = None
    time: str | None = None

    def key(self):
        return (self.agent, self.action, self.patient, self.loc, self.time)


@dataclass
class Surface:
    words: list[str] = field(default_factory=list)
    roles: list[int] = field(default_factory=list)
    types: list[int] = field(default_factory=list)

    def add(self, w, role, typ):
        self.words.append(w)
        self.roles.append(R[role])
        self.types.append(T[typ])

    def extend(self, other: "Surface"):
        self.words += other.words
        self.roles += other.roles
        self.types += other.types


def np_phrase(ent: str, role: str, form: str, rng: random.Random, pronoun: str | None = None) -> Surface:
    s = Surface()
    if pronoun is not None:
        s.add(pronoun, role, "pronoun")
        return s
    adjs = ANIMAL_ADJ if ENTITIES[ent] == "it" else PERSON_ADJ
    s.add("the", role, "function")
    if form == "medium":
        a1, a2 = rng.sample(adjs, 2)
        s.add(a1, role, "modifier")
        s.add(a2, role, "modifier")
    elif form == "long":
        s.add(rng.choice(OWNERS), role, "modifier")
        s.add("'s", role, "function")
        a1, a2 = rng.sample(adjs, 2)
        s.add(a1, role, "modifier")
        s.add(a2, role, "modifier")
    s.add(ent, role, "content")
    return s


def loc_phrase(place: str, form: str, rng: random.Random) -> Surface:
    s = Surface()
    if form == "long":
        s.add("somewhere", "loc", "function")
    s.add("in", "loc", "function")
    s.add("the", "loc", "function")
    if form in ("medium", "long"):
        for a in rng.sample(PLACE_ADJ, 1 if form == "medium" else 2):
            s.add(a, "loc", "modifier")
    s.add(place, "loc", "content")
    return s


def time_phrase(t: str, form: str, rng: random.Random) -> Surface:
    s = Surface()
    if form == "short":
        s.add("in", "time", "function")
        s.add("the", "time", "function")
    elif form == "medium":
        s.add("one", "time", "function")
        s.add(rng.choice(TIME_ADJ), "time", "modifier")
    else:
        s.add("late", "time", "function")
        s.add("in", "time", "function")
        s.add("the", "time", "function")
        s.add(rng.choice(TIME_ADJ), "time", "modifier")
    s.add(t, "time", "content")
    return s


FORMS = ["short", "medium", "long"]


def realize(idea: Idea, rng: random.Random, prev_ents: list[str] | None = None,
            forms: dict | None = None, voice: str | None = None, order: str | None = None,
            allow_pronoun: bool = True) -> tuple[Surface, dict]:
    """Render one idea. Returns the surface and the formulation choices."""
    prev_ents = prev_ents or []
    f = dict(forms or {})
    for k in ("agent", "patient", "verb", "loc", "time"):
        f.setdefault(k, rng.choice(FORMS))
    voice = voice or ("passive" if rng.random() < 0.2 else "active")
    order = order or rng.choice(["std", "std", "front"])

    def pron(ent, subject):
        if not allow_pronoun or ent not in prev_ents:
            return None
        cls = ENTITIES[ent]
        # only unambiguous reprises: a single previous entity of that pronoun class
        if sum(ENTITIES[e] == cls for e in prev_ents) != 1:
            return None
        if rng.random() < (0.5 if subject else 0.3):
            return SUBJ[cls] if subject else OBJ[cls]
        return None

    s3, ing, pp = VERBS[idea.action]
    verb = Surface()
    if voice == "passive":
        verb.add("is", "action", "function")
        verb.add(pp, "action", "content")
        verb.add("by", "action", "function")
        subj_ent, subj_role, obj_ent, obj_role = idea.patient, "patient", idea.agent, "agent"
    else:
        if f["verb"] == "short":
            verb.add(s3, "action", "content")
        elif f["verb"] == "medium":
            verb.add("is", "action", "function")
            verb.add(ing, "action", "content")
        else:
            verb.add(rng.choice(ADVERBS), "action", "modifier")
            verb.add(s3, "action", "content")
        subj_ent, subj_role, obj_ent, obj_role = idea.agent, "agent", idea.patient, "patient"

    subj = np_phrase(subj_ent, subj_role, f[subj_role], rng, pron(subj_ent, True))
    obj = np_phrase(obj_ent, obj_role, f[obj_role], rng, pron(obj_ent, False))
    core = Surface()
    core.extend(subj)
    core.extend(verb)
    core.extend(obj)

    loc = loc_phrase(idea.loc, f["loc"], rng) if idea.loc else None
    tim = time_phrase(idea.time, f["time"], rng) if idea.time else None

    out = Surface()
    front = []
    if order == "front" and (loc or tim):
        front = [loc] if loc and (not tim or rng.random() < 0.5) else [tim]
    for ph in front:
        out.extend(ph)
        out.add(",", "none", "function")
    out.extend(core)
    for ph in (loc, tim):
        if ph is not None and ph not in front:
            out.extend(ph)
    out.add(".", "none", "function")
    info = {"forms": f, "voice": voice, "order": order}
    return out, info


def all_pairs():
    ents = list(ENTITIES)
    return [(a, b) for a in ents for b in ents if a != b]


def heldout_pairs(seed: int = 1234, frac: float = 0.1):
    """Held-out ordered (agent, patient) pairs; the reverse pair always stays in training."""
    rng = random.Random(seed)
    pairs = all_pairs()
    rng.shuffle(pairs)
    held = set()
    target = int(frac * len(pairs))
    for a, b in pairs:
        if len(held) >= target:
            break
        if (b, a) not in held:
            held.add((a, b))
    return held


def sample_idea(rng, cast, held, force_held=False):
    for _ in range(1000):
        if force_held:
            a, p = rng.choice(sorted(held))
        else:
            a, p = rng.sample(cast, 2)
            if (a, p) in held:
                continue
        return Idea(a, rng.choice(list(VERBS)), p,
                    rng.choice(PLACES) if rng.random() < 0.5 else None,
                    rng.choice(TIMES) if rng.random() < 0.25 else None)
    raise RuntimeError("could not sample idea")


def sample_story(rng: random.Random, held: set, comp: bool = False):
    n_sent = rng.randint(3, 6)
    cast = rng.sample(list(ENTITIES), rng.randint(2, 3))
    ideas = []
    held_at = rng.randrange(n_sent) if comp else -1
    for i in range(n_sent):
        if i == held_at:
            idea = sample_idea(rng, cast, held, force_held=True)
            cast = list(dict.fromkeys(cast + [idea.agent, idea.patient]))
        else:
            idea = sample_idea(rng, cast, held)
        ideas.append(idea)
    sents, prev = [], []
    for idea in ideas:
        surf, info = realize(idea, rng, prev)
        sents.append((idea, surf, info))
        prev = [idea.agent, idea.patient]
    return sents


class IdeaIndex:
    """Stable integer ids for idea tuples."""

    def __init__(self):
        self.map = {}

    def __call__(self, idea: Idea) -> int:
        k = idea.key()
        if k not in self.map:
            self.map[k] = len(self.map)
        return self.map[k]


def encode_story(sents, w2i, idx: IdeaIndex):
    ideas = []
    for idea, surf, info in sents:
        ideas.append({
            "tokens": [w2i[w] for w in surf.words],
            "roles": surf.roles,
            "types": surf.types,
            "idea_id": idx(idea),
            "agent": idea.agent, "patient": idea.patient, "action": idea.action,
            "loc": idea.loc, "time": idea.time,
            "voice": info["voice"],
            "len_class": sum(FORMS.index(info["forms"][k]) for k in ("agent", "patient", "verb")),
        })
    return {"ideas": ideas}


def build(out_dir: Path, n_train=200_000, n_val=5_000, n_test=5_000, seed=0):
    vocab = build_vocab()
    w2i = {w: i for i, w in enumerate(vocab)}
    held = heldout_pairs()
    idx = IdeaIndex()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": (n_train, False, seed), "val": (n_val, False, seed + 1),
              "test_iid": (n_test, False, seed + 2), "test_comp": (n_test, True, seed + 3)}
    for name, (n, comp, s) in splits.items():
        rng = random.Random(s)
        docs = [encode_story(sample_story(rng, held, comp), w2i, idx) for _ in range(n)]
        torch.save({"docs": docs, "vocab": vocab}, out_dir / f"{name}.pt")
        ntok = sum(len(i["tokens"]) for d in docs for i in d["ideas"])
        print(f"{name}: {n} stories, {ntok} tokens")
    torch.save({"vocab": vocab, "heldout_pairs": sorted(held), "n_ideas": len(idx.map)},
               out_dir / "meta.pt")


def inverted_pair_probe(n: int = 2000, seed: int = 99):
    """Pairs of ideas (a, v, p) / (p, v, a) rendered with identical formulation choices."""
    rng = random.Random(seed)
    out = []
    ents = list(ENTITIES)
    for _ in range(n):
        a, p = rng.sample(ents, 2)
        v = rng.choice(list(VERBS))
        forms = {k: rng.choice(FORMS) for k in ("agent", "patient", "verb", "loc", "time")}
        voice = "passive" if rng.random() < 0.2 else "active"
        pair = []
        for idea in (Idea(a, v, p), Idea(p, v, a)):
            r2 = random.Random(rng.random())
            surf, _ = realize(idea, r2, [], forms=forms, voice=voice, order="std", allow_pronoun=False)
            pair.append((idea, surf))
        out.append(pair)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n_train", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(Path(a.out), n_train=a.n_train, seed=a.seed)
