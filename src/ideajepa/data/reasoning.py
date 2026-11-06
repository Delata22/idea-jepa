"""Synthetic v2: short deductive "reasoning" stories (one sentence = one idea = one step).

Three rule families (one per story in v2.0; label in metadata):
  F1 location tracking (bAbI-like): premises = ordered actions ("tom goes to the kitchen", "tom takes the
     ball", "tom drops the ball"); deductions = final location of each object ("so the ball is in the park").
  F2 kinship composition (CLUTRR-like): "this is about A" + chain edges "B is the brother of C" (shuffled);
     deductions compose from A: "so A is the aunt of D", with a small closed composition table.
  F3 keys and boxes (ProofWriter-like Horn clauses): "tom has the red key", "the red key opens the blue box",
     "the blue box holds the green key"; deductions "so tom can open the blue box", "so tom gets the green key".

Branching is exact: at each deduction step the valid next steps A_t are the *frontier* of facts that are
derivable from what has been stated (premises + previous deductions) and not yet stated. A_t is a
deterministic function of the visible text; the generator states one of them uniformly, so the per-step
entropy of the choice of deduction is exactly log|A_t| (b_eff = |A_t| is stored).
Premises are implicit in deductions by default; with probability p_restate the deduction restates its
antecedent ("since tom has the red key , tom can open the blue box ."), flagged per idea.
Distractors: true premises that are never used by any deduction (flagged).

python -m ideajepa.data.reasoning --out data/reasoning
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from .synthetic import R, SPECIALS, T  # same role / type ids as synthetic v1

NAMES = {"tom": "m", "ben": "m", "sam": "m", "max": "m", "leo": "m", "jack": "m",
         "anna": "f", "lily": "f", "sue": "f", "mia": "f", "eva": "f", "kate": "f"}
COLORS = ["red", "blue", "green", "yellow", "black", "white", "pink", "brown", "gold", "silver"]
ITEMS = ["coin", "ring", "apple", "letter", "map", "stone"]
LOCS = ["kitchen", "garden", "park", "school", "garage", "beach", "office", "attic"]
OBJS = ["ball", "book", "cup", "hat", "box", "doll", "kite", "lamp"]
CONNECT = [["so"], ["then"], ["this", "means", "that"]]
REL_WORDS = {"parent": ("father", "mother"), "sibling": ("brother", "sister"),
             "grandparent": ("grandfather", "grandmother"), "pibling": ("uncle", "aunt")}
# composition table: X is R1 of Y, Y is R2 of Z  ->  X is R of Z  (closed, unambiguous)
COMPOSE = {("parent", "parent"): "grandparent", ("parent", "sibling"): "parent",
           ("parent", "pibling"): "grandparent", ("sibling", "parent"): "pibling",
           ("sibling", "sibling"): "sibling", ("sibling", "pibling"): "pibling",
           ("grandparent", "sibling"): "grandparent", ("pibling", "sibling"): "pibling"}

FUNCTION = {"the", "is", "of", "so", "then", "this", "means", "that", "since", ",", ".", "can", "by", "'s",
            "to", "in", "about", "a", "an", "be", "up", "ends", "goes"}


def build_vocab():
    words = set(FUNCTION) | set(NAMES) | set(COLORS) | set(ITEMS) | set(LOCS) | set(OBJS)
    words |= {w for pair in REL_WORDS.values() for w in pair}
    words |= {"has", "opens", "holds", "open", "opened", "gets", "get", "key", "keys", "takes", "drops",
              "moves", "went", "go", "reachable", "for", "as", "picks", "leaves", "at", "put"}
    # v2.1 words are appended AFTER the sorted v2 vocabulary so that v2 token ids stay unchanged
    return SPECIALS + sorted(words) + V21_WORDS


V21_WORDS = ["let", "us", "think"]


class S:
    """Surface builder with per-word roles and types."""

    def __init__(self):
        self.w, self.r, self.t = [], [], []

    def add(self, words, role="none", typ=None):
        for x in words.split() if isinstance(words, str) else words:
            self.w.append(x)
            self.r.append(R[role])
            self.t.append(T[typ or ("function" if x in FUNCTION else "content")])
        return self

    def ext(self, other):
        self.w += other.w
        self.r += other.r
        self.t += other.t
        return self


# ------------------------------------------------------------------ rendering of facts
def np_obj(kind, color):  # "the red key"
    return f"the {color} {kind}"


def render_fact(f, rng, deduction=False, restate=None):
    """Render a fact tuple as a sentence. f = (pred, *args). Returns S and the choice labels."""
    s = S()
    choice = {}
    if deduction:
        if restate is not None:
            s.add("since")
            s.ext(render_clause(restate, rng, 0)[0])
            s.add(",")
            choice["connector"] = "since"
        else:
            c = rng.randrange(len(CONNECT))
            s.add(CONNECT[c])
            choice["connector"] = c
    # premises about initial locations use the neutral form only ("ends up in" would imply a deduction)
    if f[0] == "marker":
        s.ext(render_clause(f, rng, 0)[0]).add(".")
        return s, {"form": 0}
    form = 0 if (not deduction and f[0] in ("objin", "at")) else rng.randrange(2)
    clause, _ = render_clause(f, rng, form)
    s.ext(clause)
    s.add(".")
    choice["form"] = form
    return s, choice


def render_clause(f, rng, form):
    p = f[0]
    s = S()
    if p == "has":          # (has, X, color)
        s.add(f[1], "agent").add("has", "action").add(np_obj("key", f[2]), "patient")
    elif p == "opens":      # (opens, kcolor, bcolor)
        if form == 0:
            s.add(np_obj("key", f[1]), "agent").add("opens", "action").add(np_obj("box", f[2]), "patient")
        else:
            s.add(np_obj("box", f[2]), "patient").add("is opened by", "action").add(np_obj("key", f[1]), "agent")
    elif p == "holds":      # (holds, bcolor, item)  item = ("key", color) or ("item", name)
        what = np_obj("key", f[2][1]) if f[2][0] == "key" else f"the {f[2][1]}"
        s.add(np_obj("box", f[1]), "agent").add("holds", "action").add(what, "patient")
    elif p == "canopen":    # (canopen, X, bcolor)
        if form == 0:
            s.add(f[1], "agent").add("can open", "action").add(np_obj("box", f[2]), "patient")
        else:
            s.add(np_obj("box", f[2]), "patient").add("can be opened by", "action").add(f[1], "agent")
    elif p == "gets":       # (gets, X, item)
        what = np_obj("key", f[2][1]) if f[2][0] == "key" else f"the {f[2][1]}"
        if form == 0:
            s.add(f[1], "agent").add("gets", "action").add(what, "patient")
        else:
            s.add(what, "patient").add("goes to", "action").add(f[1], "agent")
    elif p == "marker":     # (marker,): boundary between premises and deductions (v2.1)
        s.add("let us think", "none", "function")
    elif p == "about":      # (about, A)
        s.add("this is about", "none").add(f[1], "agent")
    elif p == "rel":        # (rel, X, R, Y): X is the R of Y
        word = REL_WORDS[f[2]][NAMES[f[1]] == "f"]
        if form == 0:
            s.add(f[1], "agent").add("is the", "action").add(word, "action").add("of", "action").add(f[3], "patient")
        else:
            s.add(f[3], "patient").add("'s", "action").add(word, "action").add("is", "action").add(f[1], "agent")
    elif p == "goes":       # (goes, X, loc)
        s.add(f[1], "agent").add("goes to", "action").add(f"the {f[2]}", "loc")
    elif p == "takes":      # (takes, X, obj)
        s.add(f[1], "agent").add("takes", "action").add(f"the {f[2]}", "patient")
    elif p == "drops":      # (drops, X, obj)
        s.add(f[1], "agent").add("drops", "action").add(f"the {f[2]}", "patient")
    elif p == "objin":      # (objin, obj, loc)
        if form == 0:
            s.add(f"the {f[1]}", "patient").add("is in", "action").add(f"the {f[2]}", "loc")
        else:
            s.add(f"the {f[1]}", "patient").add("ends up in", "action").add(f"the {f[2]}", "loc")
    elif p == "at":         # (at, X, loc) initial position premise
        s.add(f[1], "agent").add("is in", "action").add(f"the {f[2]}", "loc")
    else:
        raise ValueError(p)
    return s, form


# ------------------------------------------------------------------ families
def gen_f3(rng, depth_range, n_people):
    """Keys and boxes. Returns (premises, used_flags, derive_fn, antecedent_fn, entities)."""
    for _ in range(200):
        colors_k = rng.sample(COLORS, len(COLORS))
        colors_b = rng.sample(COLORS, len(COLORS))
        people = rng.sample(sorted(NAMES), n_people)
        facts = []
        ki = bi = 0
        # frontier of keys to expand: (holder, key color)
        agenda = []
        for p in people[:max(1, n_people - 1)]:  # the last person (if >1) may be a distractor-only name
            agenda.append((p, colors_k[ki]))
            facts.append(("has", p, colors_k[ki]))
            ki += 1
        target = rng.randint(*depth_range)
        n_ded = 0
        while agenda and n_ded < target and bi < len(colors_b) and ki < len(colors_k):
            holder, kc = agenda.pop(rng.randrange(len(agenda)))
            n_boxes = 2 if rng.random() < 0.15 else 1
            for _ in range(n_boxes):
                if bi >= len(colors_b):
                    break
                bc = colors_b[bi]; bi += 1
                facts.append(("opens", kc, bc))
                n_ded += 1  # canopen
                n_in = 2 if rng.random() < 0.2 else 1
                for _ in range(n_in):
                    if rng.random() < 0.75 and ki < len(colors_k):
                        c = colors_k[ki]; ki += 1
                        facts.append(("holds", bc, ("key", c)))
                        agenda.append((holder, c))
                    else:
                        facts.append(("holds", bc, ("item", rng.choice(ITEMS))))
                    n_ded += 1  # gets
        # distractors: facts about objects nobody can reach
        n_dis = rng.randint(1, 2)
        dis = []
        for _ in range(n_dis):
            if ki < len(colors_k) and bi < len(colors_b):
                if rng.random() < 0.5:
                    dis.append(("opens", colors_k[ki], colors_b[bi]))
                    ki += 1
                else:
                    dis.append(("holds", colors_b[bi], ("item", rng.choice(ITEMS))))
                bi += 1
        if n_people > 1 and len(people) > 1 and rng.random() < 0.5 and ki < len(colors_k):
            dis.append(("has", people[-1], colors_k[ki]))  # a key that opens nothing
            ki += 1
        if depth_range[0] <= n_ded <= depth_range[1] + 2:
            break
    prem = facts + dis
    return prem, set(dis), people


def closure_step_f3(stated):
    """Derivable-and-unstated facts given the set of stated facts (F3 rules)."""
    has = {(f[1], f[2]) for f in stated if f[0] == "has"} | \
          {(f[1], f[2][1]) for f in stated if f[0] == "gets" and f[2][0] == "key"}
    opens = {(f[1], f[2]) for f in stated if f[0] == "opens"}
    canopen = {(f[1], f[2]) for f in stated if f[0] == "canopen"}
    holds = [(f[1], f[2]) for f in stated if f[0] == "holds"]
    out = set()
    for (x, k) in has:
        for (k2, b) in opens:
            if k2 == k and (x, b) not in canopen:
                out.add(("canopen", x, b))
    for (x, b) in canopen:
        for (b2, item) in holds:
            if b2 == b:
                g = ("gets", x, item)
                if g not in stated:
                    out.add(g)
    return out


def antecedent_f3(f, stated):
    """The premise that actually licenses deduction f (deterministic: sorted candidates).
    canopen(X,B): the key X holds that opens B -> restate "X has K" if it is a premise, else "K opens B".
    gets(X,O):    the box X can open that holds O (another box may hold the same item: a distractor)."""
    if f[0] == "canopen":
        held = {s[2] for s in stated if s[0] == "has" and s[1] == f[1]} | \
               {s[2][1] for s in stated if s[0] == "gets" and s[1] == f[1] and s[2][0] == "key"}
        for s in sorted((s for s in stated if s[0] == "opens" and s[2] == f[2] and s[1] in held), key=repr):
            return ("has", f[1], s[1]) if ("has", f[1], s[1]) in stated else s
    if f[0] == "gets":
        boxes = {s[2] for s in stated if s[0] == "canopen" and s[1] == f[1]}
        for s in sorted((s for s in stated if s[0] == "holds" and s[2] == f[2] and s[1] in boxes), key=repr):
            return s
    return None


def gen_f2(rng, depth_range, n_people):
    """Kinship: chain(s) starting at A; composition along the chain from A."""
    target0 = rng.randint(*depth_range)
    for _ in range(500):
        # a chain of k compositions needs k + 2 people (+1 for a second branch / distractor)
        n = min(len(NAMES), max(n_people, target0 + 2) + rng.randint(1, 2))
        people = rng.sample(sorted(NAMES), n)
        A = people[0]
        rest = people[1:]
        edges = []
        # first edge from A, then extend chains; a second branch from A with some probability
        n_branches = 2 if (len(rest) >= 3 and rng.random() < 0.35) else 1
        target = target0
        heads = []
        pool = list(rest)
        for _ in range(n_branches):
            if not pool:
                break
            y = pool.pop(0)
            r = rng.choice(list(REL_WORDS))
            edges.append(("rel", A, r, y))
            heads.append((y, r))
        n_ded = 0
        while pool and n_ded < target:
            i = rng.randrange(len(heads))
            y, r_acc = heads[i]
            options = [r2 for r2 in REL_WORDS if (r_acc, r2) in COMPOSE]
            if not options:
                heads.pop(i)
                if not heads:
                    break
                continue
            r2 = rng.choice(options)
            z = pool.pop(0)
            edges.append(("rel", y, r2, z))
            heads[i] = (z, COMPOSE[(r_acc, r2)])
            n_ded += 1
        dis = []
        if pool:
            z = pool.pop(0)
            dis.append(("rel", z, rng.choice(list(REL_WORDS)), rng.choice(rest)))
        if depth_range[0] <= n_ded <= depth_range[1]:
            return [("about", A)] + edges + dis, set(dis), people
    raise RuntimeError("f2 generation failed")


def closure_step_f2(stated):
    A = next(f[1] for f in stated if f[0] == "about")
    known = {(f[3], f[2]) for f in stated if f[0] == "rel" and f[1] == A}  # (Y, R): A is R of Y
    edges = [f for f in stated if f[0] == "rel" and f[1] != A]
    out = set()
    for (y, r1) in known:
        for e in edges:
            if e[1] == y and (r1, e[2]) in COMPOSE:
                f = ("rel", A, COMPOSE[(r1, e[2])], e[3])
                if f not in stated and e[3] != A:
                    out.add(f)
    return out


def antecedent_f2(f, stated):
    """The chain edge actually composed: y -[r2]-> z with A known to be r1 of y and COMPOSE[r1, r2] = r."""
    A, r, z = f[1], f[2], f[3]
    known = {(g[3], g[2]) for g in stated if g[0] == "rel" and g[1] == A}
    for e in sorted(stated, key=repr):
        if e[0] == "rel" and e[3] == z and e[1] != A:
            if any(y == e[1] and COMPOSE.get((r1, e[2])) == r for (y, r1) in known):
                return e
    return None


def gen_f1(rng, depth_range, n_people, dense=0):
    """Location tracking: ordered actions, deductions = final locations of the objects.
    dense > 0 (v2.1): more objects and actions per person (many facts about the same entities)."""
    people = rng.sample(sorted(NAMES), n_people)
    locs = rng.sample(LOCS, rng.randint(3, 4))
    n_obj = rng.randint(max(1, depth_range[0] // 2), max(1, min(len(OBJS), depth_range[1] // 2 + 1)))
    if dense:
        n_obj = min(len(OBJS), n_obj + dense // 2)
    objs = rng.sample(OBJS, n_obj)
    pos = {p: rng.choice(locs) for p in people}
    where = {o: rng.choice(locs) for o in objs}
    holder = {o: None for o in objs}
    prem = [("at", p, pos[p]) for p in people] + [("objin", o, where[o]) for o in objs]
    n_actions = rng.randint(depth_range[0] + 2, depth_range[1] + 4) + (2 * dense if dense else 0)
    for _ in range(n_actions):
        p = rng.choice(people)
        acts = [("goes", p, l) for l in locs if l != pos[p]]
        acts += [("takes", p, o) for o in objs if holder[o] is None and where[o] == pos[p]]
        acts += [("drops", p, o) for o in objs if holder[o] == p]
        a = rng.choice(acts)
        if a[0] == "goes":
            pos[p] = a[2]
            for o in objs:
                if holder[o] == p:
                    where[o] = a[2]
        elif a[0] == "takes":
            holder[a[2]] = p
        else:
            holder[a[2]] = None
        prem.append(a)
    return prem, set(), people


def final_state_f1(stated):
    pos, where, holder = {}, {}, {}
    for f in stated:
        if f[0] == "at":
            pos[f[1]] = f[2]
        elif f[0] == "objin" and f[1] not in where:
            where[f[1]] = f[2]
            holder[f[1]] = None
        elif f[0] == "goes":
            pos[f[1]] = f[2]
            for o, h in holder.items():
                if h == f[1]:
                    where[o] = f[2]
        elif f[0] == "takes":
            holder[f[2]] = f[1]
        elif f[0] == "drops":
            holder[f[2]] = None
    return where


def closure_step_f1(stated, n_premises):
    prem = stated[:n_premises]
    where = final_state_f1(prem)
    said = {f[1] for f in stated[n_premises:] if f[0] == "objin"}
    return {("objin", o, l) for o, l in where.items() if o not in said}


# ------------------------------------------------------------------ fixpoint helpers (shared with verifier.py)
def family_closure(family, stated, n_prem):
    """Derivable-and-unstated facts (the frontier) given the list of stated facts (premises first)."""
    if family == "F1":
        return closure_step_f1(stated, n_prem)
    if family == "F2":
        return closure_step_f2(set(stated))
    return closure_step_f3(set(stated))


def derive_all(family, premises, exclude=None):
    """Fixpoint: every fact entailed by the premises (optionally never deriving `exclude`)."""
    stated, out = list(premises), set()
    while True:
        new = family_closure(family, stated, len(premises)) - out - ({exclude} if exclude else set())
        if not new:
            return out
        out |= new
        stated += sorted(new, key=repr)


def derive_sinks(family, premises):
    """Final conclusions: derived facts that no other derived fact depends on."""
    allf = derive_all(family, premises)
    return {f for f in allf if len(derive_all(family, premises, exclude=f)) == len(allf) - 1}


def add_dense(family, prem, dis, rng, n):
    """v2.1 'dense' worlds: up to n extra TRUE-but-useless premises that mention entities/objects of the
    chain (key/name collisions). A candidate is kept only if the fixpoint of derived facts is unchanged."""
    base = derive_all(family, prem)
    used_k = {f[2] for f in prem if f[0] == "has"} | {f[1] for f in prem if f[0] == "opens"} | \
             {f[2][1] for f in prem if f[0] == "holds" and f[2][0] == "key"}
    used_b = {f[2] for f in prem if f[0] == "opens"} | {f[1] for f in prem if f[0] == "holds"}
    extra = []
    for _ in range(4 * n):
        if len(extra) >= n:
            break
        if family == "F3":
            owners = [f[1] for f in prem if f[0] == "has"]
            free_k = [c for c in COLORS if c not in used_k]
            free_b = [c for c in COLORS if c not in used_b]
            kind = rng.randrange(3)
            if kind == 0 and free_k and owners:          # the protagonist owns a key that opens nothing
                c = rng.choice(free_k); cand = ("has", rng.choice(owners), c); used_k.add(c)
            elif kind == 1 and free_k and used_b:        # an unowned key opens a box of the chain
                c = rng.choice(free_k); cand = ("opens", c, rng.choice(sorted(used_b))); used_k.add(c)
            elif kind == 2 and free_b and used_k:        # an unopenable box holds a key colour of the chain
                b = rng.choice(free_b); cand = ("holds", b, ("key", rng.choice(sorted(used_k)))); used_b.add(b)
            else:
                continue
        elif family == "F2":
            people = {x for f in prem if f[0] == "rel" for x in (f[1], f[3])} | {f[1] for f in prem if f[0] == "about"}
            free = [x for x in sorted(NAMES) if x not in people]
            if not free:
                break
            cand = ("rel", rng.choice(free), rng.choice(list(REL_WORDS)), rng.choice(sorted(people)))
        else:
            break  # F1 density is set in gen_f1 (more objects and actions)
        if derive_all(family, prem + extra + [cand]) == base:
            extra.append(cand)
    return prem + extra, dis | set(extra)


# ------------------------------------------------------------------ story generation
def generate_story(rng, family, depth_range=(3, 7), n_people=None, p_restate=0.2, mode="chain",
                   marker=False, dense=0):
    """mode: "chain" (every deduction step is written) | "skip" (only the final conclusions = sinks of the
    derivation DAG, in any order: premises -> conclusion, the intermediate steps are latent).
    marker: add the boundary sentence "let us think ." between premises and deductions.
    dense: number of extra useless premises about the chain's entities (F2/F3); F1: more objects/actions."""
    n_people = n_people or rng.randint(3, 4)
    if family == "F3":
        prem, dis, _ = gen_f3(rng, depth_range, min(n_people, 2))
        if dense:
            prem, dis = add_dense("F3", prem, dis, rng, dense)
        rng.shuffle(prem)
        closure, ante = (lambda st, n: closure_step_f3(set(st))), (lambda f, st: antecedent_f3(f, set(st)))
    elif family == "F2":
        prem, dis, _ = gen_f2(rng, depth_range, n_people)
        if dense:
            prem, dis = add_dense("F2", prem, dis, rng, dense)
        head, rest = prem[0], prem[1:]
        rng.shuffle(rest)
        prem = [head] + rest
        closure, ante = (lambda st, n: closure_step_f2(set(st))), (lambda f, st: antecedent_f2(f, set(st)))
    else:
        prem, dis, _ = gen_f1(rng, depth_range, n_people, dense=dense)
        closure, ante = (lambda st, n: closure_step_f1(st, n)), (lambda f, st: None)
    stated = list(prem)
    steps = []
    for f in prem:
        steps.append({"fact": f, "step": "premise", "distractor": f in dis})
    if marker:
        steps.append({"fact": ("marker",), "step": "marker"})
    depth = 0
    if mode == "skip":
        remaining = derive_sinks(family, prem)
        while remaining:
            cand = sorted(remaining, key=repr)
            f = cand[rng.randrange(len(cand))]
            steps.append({"fact": f, "step": "deduction", "b_eff": len(cand), "candidates": cand,
                          "restate": None, "restate_possible": False, "depth": depth, "skip": True})
            remaining.discard(f)
            depth += 1
        return steps
    while True:
        cand = sorted(closure(stated, len(prem)), key=repr)
        if not cand:
            break
        f = cand[rng.randrange(len(cand))]
        possible = ante(f, stated)
        a = possible if (possible is not None and rng.random() < p_restate) else None
        steps.append({"fact": f, "step": "deduction", "b_eff": len(cand), "candidates": cand,
                      "restate": a, "restate_possible": possible is not None, "depth": depth})
        stated.append(f)
        depth += 1
    return steps


def realise(steps, rng, w2i, idx, distinct_final_ids=False):
    """distinct_final_ids (v2.1): in F1 a final location may equal the initial-position premise (same fact
    tuple); give the deduction its own idea_id so that premise and conclusion are distinct ideas."""
    ideas = []
    for st in steps:
        key = st["fact"]
        if distinct_final_ids and st["step"] == "deduction" and key[0] == "objin":
            key = ("objin_final",) + tuple(key[1:])
        surf, choice = render_fact(st["fact"], rng, deduction=st["step"] == "deduction", restate=st.get("restate"))
        ideas.append({
            "tokens": [w2i[w] for w in surf.w], "roles": surf.r, "types": surf.t,
            "idea_id": idx(key), "fact": st["fact"], "step": st["step"],
            "b_eff": st.get("b_eff", 0), "candidates": st.get("candidates"),
            "restated": st.get("restate") is not None, "restate_possible": st.get("restate_possible", False),
            "distractor": st.get("distractor", False),
            "depth": st.get("depth", -1), "connector": choice.get("connector"), "form": choice["form"],
            # compatibility with the v1 diagnostics
            "agent": None, "patient": None, "voice": "active", "len_class": 0,
        })
    return {"ideas": ideas}


N_TEMPLATES = 10
DEDUCTION_TEMPLATES = list(range(2, 8))  # the non-restated deduction templates (drawn at random in generation)


def template_id(idea):
    """Surface template of an idea ("separate form" variant): 0-1 premise/marker form; 2 + 2*connector + form
    for a deduction (3 connectors x 2 forms); 8 + form for a restated deduction ("since ..., ...")."""
    form = int(idea.get("form") or 0)
    if idea.get("step") != "deduction":
        return form
    if idea.get("restated"):
        return 8 + form
    return 2 + 2 * int(idea.get("connector") or 0) + form


def render_with_template(fact, t, rng=None):
    """Render a deduction with a given template t in DEDUCTION_TEMPLATES (inverse of template_id)."""
    conn, form = divmod(t - 2, 2)
    s = S().add(CONNECT[conn])
    s.ext(render_clause(fact, rng, form)[0]).add(".")
    return s


def deduction_logprob(idea):
    """Exact log-probability of a deduction idea given the visible past (choice of step + paraphrase)."""
    lp = -math.log(idea["b_eff"])
    if idea["restated"]:
        lp += math.log(0.2)
    else:
        lp += (math.log(0.8) if idea["restate_possible"] else 0.0) - math.log(len(CONNECT))
    return lp - math.log(2)  # form (deductions always choose among 2 forms)


class IdeaIndex:
    def __init__(self):
        self.map = {}

    def __call__(self, f):
        if f not in self.map:
            self.map[f] = len(self.map)
        return self.map[f]


def build(out: Path, n_train=200_000, n_eval=4_000, seed=0, mode="chain", marker=False, dense=0):
    vocab = build_vocab()
    w2i = {w: i for i, w in enumerate(vocab)}
    idx = IdeaIndex()
    out.mkdir(parents=True, exist_ok=True)
    fams = ["F1", "F2", "F3"]
    splits = {
        "train": dict(n=n_train, depth=(3, 7), people=None, seed=seed),
        "val": dict(n=n_eval, depth=(3, 7), people=None, seed=seed + 1),
        "test_iid": dict(n=n_eval, depth=(3, 7), people=None, seed=seed + 2),
        "test_depth": dict(n=n_eval, depth=(8, 10), people=None, seed=seed + 3),
        "test_entities": dict(n=n_eval, depth=(3, 7), people=5, seed=seed + 4),
    }
    stats = {}
    for name, sp in splits.items():
        rng = random.Random(sp["seed"])
        docs = []
        for i in range(sp["n"]):
            fam = fams[i % 3]
            steps = generate_story(rng, fam, sp["depth"], sp["people"], mode=mode, marker=marker, dense=dense)
            d = realise(steps, rng, w2i, idx, distinct_final_ids=(mode != "chain" or marker or dense > 0))
            d["family"] = fam
            docs.append(d)
        ded = [i for d in docs for i in d["ideas"] if i["step"] == "deduction"]
        ntok = sum(len(i["tokens"]) for d in docs for i in d["ideas"])
        ntok_ded = sum(len(i["tokens"]) for i in ded)
        floor = -sum(deduction_logprob(i) for i in ded) / max(1, ntok_ded)
        from collections import Counter
        b = Counter(min(i["b_eff"], 3) for i in ded)
        stats[name] = {"docs": len(docs), "tokens": ntok, "deduction_tokens": ntok_ded,
                       "deduction_floor_nats_per_token": floor,
                       "b_eff_share": {k: v / len(ded) for k, v in sorted(b.items())},
                       "mean_deductions": len(ded) / len(docs)}
        torch.save({"docs": docs, "vocab": vocab}, out / f"{name}.pt")
        print(name, stats[name], flush=True)
    torch.save({"vocab": vocab, "stats": stats, "n_ideas": len(idx.map),
                "options": {"mode": mode, "marker": marker, "dense": dense}}, out / "meta.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reasoning")
    ap.add_argument("--n_train", type=int, default=200_000)
    ap.add_argument("--mode", default="chain", choices=["chain", "skip"])
    ap.add_argument("--marker", action="store_true")
    ap.add_argument("--dense", type=int, default=0)
    a = ap.parse_args()
    build(Path(a.out), n_train=a.n_train, mode=a.mode, marker=a.marker, dense=a.dense)


# ------------------------------------------------------------------ surfaces of a fact (stage A)
DEDUCIBLE = {"objin", "rel", "canopen", "gets"}                     # predicates that can be written as deductions
PREMISE_PREDS = {"objin", "at", "rel", "has", "opens", "holds", "about", "goes", "takes", "drops"}
SO_TEMPLATES = [2, 3]                                                # "so" + clause form 0 / 1


def premise_forms(fact):
    """Surface(s) of a fact written as a premise (no connector): 2 clause forms when they differ, else 1.
    Initial locations ("objin"/"at") use the neutral form only, as the generator does."""
    forms = (0,) if fact[0] in ("objin", "at") else (0, 1)
    out = []
    for form in forms:
        s = S().ext(render_clause(fact, None, form)[0]).add(".")
        if s.w not in out:
            out.append(s.w)
    return out


def role_forms(fact, step):
    """What the DECODER is trained on for an idea of this role: the two "so" forms of a deduction, the
    premise form(s) of a premise. Returns lists of words."""
    if fact[0] == "marker":
        return [["let", "us", "think", "."]]
    if step == "deduction":
        out = []
        for t in SO_TEMPLATES:
            w = render_with_template(fact, t).w
            if w not in out:
                out.append(w)
        return out
    return premise_forms(fact)


def fact_surfaces(fact):
    """Every surface of the fact tuple, whatever the role (what the ENCODER may read under cross-encoding):
    the 6 deduction templates if the predicate is deducible + its premise form(s)."""
    if fact[0] == "marker":
        return [["let", "us", "think", "."]]
    out = []
    if fact[0] in DEDUCIBLE:
        for t in DEDUCTION_TEMPLATES:
            w = render_with_template(fact, t).w
            if w not in out:
                out.append(w)
    if fact[0] in PREMISE_PREDS:
        for w in premise_forms(fact):
            if w not in out:
                out.append(w)
    return out


def antecedent(family, f, stated):
    """The antecedent a restated deduction f would carry given the facts already stated (the generator's rule;
    None when f cannot be restated, e.g. F1)."""
    if family == "F3":
        return antecedent_f3(f, set(stated))
    if family == "F2":
        return antecedent_f2(f, set(stated))
    return None


def since_forms(fact, ante):
    """Restated surfaces of deduction `fact` with antecedent `ante`: "since" + ante (clause form 0, as generated) + ","
    + fact in each clause form + ".". Returns lists of words."""
    out = []
    for form in (0, 1):
        s = S().add("since").ext(render_clause(ante, None, 0)[0]).add(",").ext(render_clause(fact, None, form)[0]).add(".")
        if s.w not in out:
            out.append(s.w)
    return out


def f1_useful_premises(prem_facts):
    """F1 (no generator antecedent): object -> index among the premises of the LAST premise that fixed its location
    (its initial "the o is in L", or the last "X goes to L" of the person carrying it then). Its location is the final
    one. Shared by the step-2 rule arm and scripts/diag_query_oracle.py."""
    src, holder = {}, {}
    for i, f in enumerate(prem_facts):
        if f[0] == "objin" and f[1] not in src:
            src[f[1]] = i; holder[f[1]] = None
        elif f[0] == "goes":
            for o, h in holder.items():
                if h == f[1]:
                    src[o] = i
        elif f[0] == "takes":
            holder[f[2]] = f[1]
        elif f[0] == "drops":
            holder[f[2]] = None
    return src


def split_compound(doc, mode, vocab, rel_source="connector", rel_f1="useful"):
    """Chantier 3 bis, step 1: the idea unit becomes the proposition. A restated deduction "since A , B ." becomes
      mode "conclusion": the proposition B alone ("so" + B in the clause form written), with the relation label
                         rel = "justified_by" and rel_ptr = index of the idea stating A, already in the story (no copy);
      mode "split":      two ideas, A ("so" + A, clause form 0; context only: train_w = 0, split_context = True) then B
                         (rel = "justified_by", rel_ptr = the index of that A).
    Every other idea gets rel = "sequence" (connector "then") or "none", rel_ptr = -1. Labels are stored, not used.
    rel_source (step 2): "connector" = the labels above (the relation exists only where the text gives it);
    "rule" = CEILING arm: every deduction whose licensing premise is given by the generator's rule (F2, F3; F1 has
    none) gets rel = "justified_by" with the pointer to the last idea stating it; F1 (rel_f1 = "useful"): pointer to
    the useful premise of the object (f1_useful_premises), rel_f1 = "none": connector labels kept.
    Returns a new document; the input is left untouched."""
    from . import verifier as V
    as_fact = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f) if f is not None else None
    w2i = {w: i for i, w in enumerate(vocab)}
    out = []
    for idea in doc["ideas"]:
        f = as_fact(idea.get("fact"))
        if idea.get("step") == "deduction" and idea.get("restated"):
            A = V.parse_sentence(" ".join(vocab[t] for t in idea["tokens"]))[1]
            B_tok = [w2i[x] for x in render_with_template(f, 2 + int(idea.get("form") or 0)).w]
            base = {**idea, "tokens": B_tok, "restated": False, "connector": 0, "rel": "justified_by",
                    "roles": None, "types": None}
            prev = [j for j, x in enumerate(out) if as_fact(x.get("fact")) == A]
            if mode == "split":   # the copy of A keeps the idea_id of the idea that stated A (not B's)
                out.append({**idea, "idea_id": out[prev[-1]]["idea_id"] if prev else idea.get("idea_id"), "fact": A, "tokens": [w2i[x] for x in render_with_template(A, 2).w],
                            "restated": False, "connector": 0, "form": 0, "rel": "none", "rel_ptr": -1,
                            "train_w": 0.0, "split_context": True, "candidates": None, "roles": None, "types": None})
                base["rel_ptr"] = len(out) - 1
            else:
                base["rel_ptr"] = prev[-1] if prev else -1
            out.append({k: v for k, v in base.items() if v is not None or k not in ("roles", "types")})
        else:
            rel = "sequence" if idea.get("step") == "deduction" and idea.get("connector") == 1 else "none"
            out.append({**idea, "rel": rel, "rel_ptr": -1})
    if rel_source == "rule":
        prem_idx = [n for n, x in enumerate(out) if x.get("step") == "premise"]
        useful = (f1_useful_premises([as_fact(out[n]["fact"]) for n in prem_idx])
                  if doc.get("family") == "F1" and rel_f1 == "useful" else {})
        for n, idea in enumerate(out):
            f_ = as_fact(idea.get("fact"))
            if idea.get("step") == "deduction" and useful and f_ is not None and f_[0] == "objin" and f_[1] in useful:
                out[n] = {**idea, "rel": "justified_by", "rel_ptr": prem_idx[useful[f_[1]]]}
                continue
            if idea.get("step") != "deduction" or idea.get("split_context"):
                continue
            stated = [as_fact(x.get("fact")) for x in out[:n] if x.get("fact") is not None]
            a = antecedent(doc["family"], as_fact(idea["fact"]), stated) if doc.get("family") in ("F2", "F3") else None
            ptr = [m for m in range(n) if as_fact(out[m].get("fact")) == a] if a is not None else []
            if ptr:
                out[n] = {**idea, "rel": "justified_by", "rel_ptr": ptr[-1]}
    return {**doc, "ideas": out}
