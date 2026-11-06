"""Exact verifier for synthetic v2 (reasoning): text -> facts -> verdicts, from the TRUE state.

    parse_sentence("so tom can open the red box .")      -> ("canopen", "tom", "red")
    valid_next(family, premises, said)                   -> set of facts that are valid next deductions
    judge(family, premises, sentences)                   -> one verdict per generated sentence

The state is always rebuilt from the true premises plus the model's *valid* statements only: a false
statement neither blocks the correct fact nor seeds false derivations (the DAgger-oracle bug of 26/09).

Verdicts:
  valid       the fact is in the exact frontier at that point (derivable now, not yet stated)
  premature   true (in the fixpoint closure of the premises) but not derivable yet from what was stated
  repeat      already stated (as a premise or as a previous valid deduction)
  premise     restates a premise (true, but not a deduction)
  false       parsed, but not entailed by the premises
  unparsed    not a well-formed sentence of the grammar
In "skip" stories (only final conclusions are written, see reasoning.py) pass skip=True: the valid set is the
not-yet-stated sinks of the derivation DAG.
"""
from __future__ import annotations

import re

from . import reasoning as R

REL = {"mother": "parent", "father": "parent", "grandmother": "grandparent", "grandfather": "grandparent",
       "sister": "sibling", "brother": "sibling", "aunt": "pibling", "uncle": "pibling"}
W = r"([a-z]+)"
PATTERNS = [
    # deductions
    (rf"the {W} (?:is|ends up) in the {W}", lambda m: ("objin", m[1], m[2])),
    (rf"{W} is the {W} of {W}", lambda m: ("rel", m[1], REL[m[2]], m[3]) if m[2] in REL else None),
    (rf"{W} 's {W} is {W}", lambda m: ("rel", m[3], REL[m[2]], m[1]) if m[2] in REL else None),
    (rf"the {W} box can be opened by {W}", lambda m: ("canopen", m[2], m[1])),
    (rf"{W} can open the {W} box", lambda m: ("canopen", m[1], m[2])),
    (rf"{W} gets the {W} key", lambda m: ("gets", m[1], ("key", m[2]))),
    (rf"the {W} key goes to {W}", lambda m: ("gets", m[2], ("key", m[1]))),
    (rf"{W} gets the {W}", lambda m: ("gets", m[1], ("item", m[2]))),
    (rf"the {W} goes to {W}", lambda m: ("gets", m[2], ("item", m[1]))),
    # premises
    (rf"{W} has the {W} key", lambda m: ("has", m[1], m[2])),
    (rf"the {W} key opens the {W} box", lambda m: ("opens", m[1], m[2])),
    (rf"the {W} box is opened by the {W} key", lambda m: ("opens", m[2], m[1])),
    (rf"the {W} box holds the {W} key", lambda m: ("holds", m[1], ("key", m[2]))),
    (rf"the {W} box holds the {W}", lambda m: ("holds", m[1], ("item", m[2]))),
    (rf"this is about {W}", lambda m: ("about", m[1])),
    (r"let us think", lambda m: ("marker",)),
    (rf"{W} goes to the {W}", lambda m: ("goes", m[1], m[2])),
    (rf"{W} takes the {W}", lambda m: ("takes", m[1], m[2])),
    (rf"{W} drops the {W}", lambda m: ("drops", m[1], m[2])),
    (rf"{W} is in the {W}", lambda m: ("at", m[1], m[2])),
]
PATTERNS = [(re.compile(p + "$"), f) for p, f in PATTERNS]
CONNECTOR = re.compile(r"^(?:so|then|this means that)\s+")


def parse_clause(s: str):
    s = s.strip()
    for pat, build in PATTERNS:
        m = pat.match(s)
        if m:
            f = build(m)
            if f is not None:
                return _normalise(f)
    return None


def _normalise(f):
    """Disambiguate forms shared by several predicates (object vs person, item names)."""
    if f[0] == "at" and f[1] in R.OBJS:          # "the ball is in the park" parsed by the objin pattern first
        return ("objin", f[1], f[2])
    if f[0] == "objin" and f[1] in R.NAMES:
        return ("at", f[1], f[2])
    return f


def parse_sentence(sent: str):
    """Returns (fact, restated_antecedent) ; fact is None if unparsed."""
    s = sent.strip()
    if s.endswith("."):
        s = s[:-1].strip()
    ante = None
    if s.startswith("since "):
        if "," not in s:
            return None, None
        a, s = s[len("since "):].split(",", 1)
        ante = parse_clause(a)
    else:
        s = CONNECTOR.sub("", s)
    return parse_clause(s), ante


def _closure(family, stated, n_prem):
    return R.family_closure(family, stated, n_prem)


def derivable(family, premises):
    """Fixpoint: every fact entailed by the premises under the family's rules."""
    return R.derive_all(family, premises)


def sinks(family, premises):
    """Derived facts that are never used as an antecedent of another derived fact (final conclusions)."""
    return R.derive_sinks(family, premises)


def valid_next(family, premises, said, skip=False):
    """Exact set of valid next deductions given the true premises and the list of facts already said.
    Only valid statements count as stated (they must be entailed)."""
    truth = derivable(family, premises)
    stated = list(premises) + [f for f in said if f in truth]
    if skip:
        return sinks(family, premises) - set(said)
    return _closure(family, stated, len(premises))


def judge(family, premises, sentences, skip=False, context_mask=None):
    """context_mask[i] = True: sentence i is context (an antecedent A re-stated by a proposition split):
    verdict "context", neither scored nor a repeat, and it does not enter the said facts."""
    premises = [tuple(p) if isinstance(p, list) else p for p in premises]
    truth = derivable(family, premises)
    prem_set = set(premises)
    said, out = [], []
    for si, sent in enumerate(sentences):
        f, ante = parse_sentence(sent)
        if context_mask is not None and context_mask[si]:
            out.append({"sentence": sent, "fact": f, "verdict": "context", "n_valid": 0})
            continue
        frontier = valid_next(family, premises, said, skip)
        if f is None:
            v = "unparsed"
        elif f in said:
            v = "repeat"
        elif f in frontier:  # before "premise": in F1 a final location may equal the initial-position premise
            v = "valid"
        elif f in prem_set:
            v = "premise"
        elif f in truth:
            v = "premature"
        else:
            v = "false"
        if ante is not None and ante not in prem_set and ante not in said and v == "valid":
            v = "valid_bad_antecedent"  # right conclusion, restated antecedent is not a known fact
        out.append({"sentence": sent, "fact": f, "verdict": v, "n_valid": len(frontier)})
        if v in ("valid", "valid_bad_antecedent"):
            said.append(f)
    return out


def expert_targets(family, premises, prefix_sentences):
    """DAgger expert (v3): the valid next deductions after a model-written prefix, judged strictly.
    established = premises + prefix statements valid at the time they were said (nothing is derived from a
    premature statement: it was true but not licensed); said = every true prefix statement (valid or
    premature), never taught again. Returns (targets, verdicts of the prefix)."""
    premises = [tuple(p) if isinstance(p, list) else p for p in premises]
    verdicts = judge(family, premises, prefix_sentences)
    established = list(premises) + [v["fact"] for v in verdicts if v["verdict"].startswith("valid")]
    said = {v["fact"] for v in verdicts if v["verdict"].startswith("valid") or v["verdict"] == "premature"}
    return _closure(family, established, len(premises)) - said, verdicts


def schema_facts(family, premises):
    """Every well-formed deduction of the family's schema over the entities named in the premises. Carries no
    truth information (includes false, premature and already-said facts): the honest pool for projecting a
    generated idea onto well-formed ideas about the context (gen_eval --snap context)."""
    premises = [tuple(p) if isinstance(p, list) else p for p in premises]
    ents = {x for f in premises for x in f[1:] if isinstance(x, str)}
    people = sorted(e for e in ents if e in R.NAMES)
    if family == "F1":
        objs = sorted(e for e in ents if e in R.OBJS)
        locs = sorted(e for e in ents if e in R.LOCS)
        return [("objin", o, l) for o in objs for l in locs]
    if family == "F2":
        A = next(f[1] for f in premises if f[0] == "about")
        return [("rel", A, r, y) for r in R.REL_WORDS for y in people if y != A]
    boxes = sorted({f[2] for f in premises if f[0] == "opens"} | {f[1] for f in premises if f[0] == "holds"})
    items = sorted({f[2] for f in premises if f[0] == "holds"} | {("key", f[1]) for f in premises if f[0] == "opens"}
                   | {("key", f[2]) for f in premises if f[0] == "has"}, key=repr)
    return ([("canopen", x, b) for x in people for b in boxes]
            + [("gets", x, i) for x in people for i in items])


def doc_premises(doc):
    return [tuple(i["fact"]) if isinstance(i["fact"], list) else i["fact"]
            for i in doc["ideas"] if i["step"] == "premise"]
