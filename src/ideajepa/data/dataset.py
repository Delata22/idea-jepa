"""Common document format and collation.

A document is {"ideas": [{"tokens": [...], optional "roles", "types", "idea_id", ...}, ...]}.
The collate function produces, from the same documents:
  * per-idea decoder windows `win` = [M prefix tokens | idea tokens], PAD-padded;
  * the document token sequence `seq` = BOS + all tokens (for B0);
  * `doc_index` [B, Nmax] mapping to idea rows (for the idea predictor P).
"""
from __future__ import annotations

import random

import numpy as np
import torch

PAD, BOS, EOS = 0, 1, 2


def split_long_idea(tokens, max_len):
    return [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)] or [[]]


def chunk_docs(docs, max_ideas, max_idea_len, max_tokens=None):
    """Cap idea length (split at the cap) and split documents into chunks of <= max_ideas ideas."""
    out = []
    for d in docs:
        ideas = []
        for idea in d["ideas"]:
            if len(idea["tokens"]) <= max_idea_len:
                ideas.append(idea)
            else:
                for i in range(0, len(idea["tokens"]), max_idea_len):
                    sub = {k: (v[i:i + max_idea_len] if isinstance(v, list) else v) for k, v in idea.items()}
                    ideas.append(sub)
        cur, ntok = [], 0
        for idea in ideas:
            n = len(idea["tokens"])
            if cur and (len(cur) >= max_ideas or (max_tokens and ntok + n > max_tokens)):
                out.append({"ideas": cur})
                cur, ntok = [], 0
            cur.append(idea)
            ntok += n
        if cur:
            out.append({"ideas": cur})
            # bAbI supporting facts (ORACLE variants only): kept when the document is whole (one chunk, no idea split)
            if "support" in d and len(cur) == len(d["ideas"]) and len(ideas) == len(d["ideas"]):
                out[-1]["support"] = d["support"]
    return out


class Collator:
    def __init__(self, m_prefix=3, full_prefix=False, max_prefix=None, surfaces=False, w2i=None, seed=0,
                 p_input_surface=None, since_surfaces=None, nce_positives="generator"):
        """surfaces=True (stage A, synthetic v2): the decoder is trained on the forms of each idea's ROLE
        ("so" forms for a deduction, premise forms for a premise) and the text stream / prefixes are built from
        the first role form (canonical stream, what the model reads back in generation); the encoder may read any
        surface of the fact tuple (cross-encoding). DAgger docs and ideas without a fact keep their reference.
        p_input_surface ("random" / "mix", surfaces only): P's input ideas are E(a surface the story could show for
        that idea: one of the 6 deduction templates for a deduction, a premise form for a premise), drawn per idea and
        per call ("mix": the main form with p = 0.5) -> batch["pin_tok"], batch["pin_len"] (training only).
        since_surfaces ("nce" / "all", surfaces only): the restated forms "since A , B ." of a deduction B (A = its
        antecedent in the story: parsed from a restated sentence, else the generator's rule on the facts stated
        before) join B's InfoNCE positives ("nce"), and also the cross-encoding pool ("all"). A "since" positive
        carries A as a third key (batch["pos_key2"]) so that it is never a negative of A's or B's rows."""
        self.m = m_prefix
        self.full = full_prefix
        self.max_prefix = max_prefix
        self.surfaces = surfaces
        self.w2i = w2i
        self.rng = random.Random(seed)
        self._cache = {}
        self.p_input_surface = p_input_surface
        self.since_surfaces = since_surfaces
        # stage-B InfoNCE positives: "generator" (other surface of the same fact, from the v2 generator) | "dropout"
        # (label-free, SimCSE: the SAME sentence, re-encoded under another dropout mask; false negatives = identical
        # sentences only)
        self.nce_positives = nce_positives
        self._story_cache = {}

    def _antecedents(self, d):
        """{idea index: (deduction fact B, antecedent A)} for the deductions of doc d that have (or could have) a
        restated form: parsed from a restated sentence, else the generator's rule on the facts stated before."""
        from . import reasoning as R
        from . import verifier as V
        vocab = getattr(self, "_i2w", None)
        if vocab is None:
            vocab = self._i2w = {i: w for w, i in self.w2i.items()}
        fam = d.get("family")
        facts = [tuple(tuple(x) if isinstance(x, list) else x for x in i["fact"]) if i.get("fact") is not None else None
                 for i in d["ideas"]]
        out = {}
        for ii, idea in enumerate(d["ideas"]):
            f = facts[ii]
            if f is None or idea.get("step") != "deduction" or idea.get("train_w") == 0.0:
                continue
            a = None
            if idea.get("restated"):
                a = V.parse_sentence(" ".join(vocab[t] for t in idea["tokens"]))[1]
            elif idea.get("restate_possible") and fam in ("F2", "F3"):
                a = R.antecedent(fam, f, [x for x in facts[:ii] if x is not None])
            if a is not None:
                out[ii] = (f, a)
        return out

    def _story_forms(self, idea, dagger):
        """Surfaces a story can show for this idea in its role (deduction: the 6 templates; premise: premise forms)."""
        f = idea.get("fact")
        if dagger or f is None or idea.get("train_w") == 0.0:
            return [idea["tokens"]]
        f = tuple(tuple(x) if isinstance(x, list) else x for x in f)
        key = (f, idea.get("step"))
        if key not in self._story_cache:
            from . import reasoning as R
            try:
                if idea.get("step") == "deduction" and f[0] in R.DEDUCIBLE:
                    ws = list(dict.fromkeys(tuple(R.render_with_template(f, t).w) for t in R.DEDUCTION_TEMPLATES))
                else:
                    ws = [tuple(w) for w in R.role_forms(f, idea.get("step"))]
                self._story_cache[key] = [[self.w2i[w] for w in x] for x in ws]
            except (KeyError, IndexError, TypeError):
                self._story_cache[key] = [idea["tokens"]]
        return self._story_cache[key]

    def _forms(self, idea, dagger):
        """(role forms as token lists, all surfaces as token lists) for one idea."""
        f = idea.get("fact")
        if dagger or f is None or idea.get("train_w") == 0.0:
            return [idea["tokens"]], [idea["tokens"]]
        f = tuple(tuple(x) if isinstance(x, list) else x for x in f)
        key = (f, idea.get("step"))
        if key not in self._cache:
            from . import reasoning as R
            try:
                role = [[self.w2i[w] for w in ws] for ws in R.role_forms(f, idea.get("step"))]
                surf = [[self.w2i[w] for w in ws] for ws in R.fact_surfaces(f)]
            except (KeyError, IndexError, TypeError):
                role, surf = [idea["tokens"]], [idea["tokens"]]
            self._cache[key] = (role, surf)
        return self._cache[key]

    def __call__(self, docs):
        if self.surfaces:
            return self._call_surfaces(docs)
        return self._call_plain(docs)

    def _call_surfaces(self, docs):
        """Main rows = 1st role form (stream + prefixes from it); alt rows = other role forms (same prefix);
        enc = a random surface of the fact; ref rows = the original sentence with the ORIGINAL stream's prefix."""
        main_docs, alts, encs, refs, poss, fkeys, pkeys, pins, pkeys2, tkeys = [], [], [], [], [], [], [], [], [], []
        tmap = {}
        keymap = {}
        kid = lambda f: keymap.setdefault(f, len(keymap)) if f is not None else -1
        for d in docs:
            dagger = bool(d.get("dagger"))
            ideas = []
            antes = self._antecedents(d) if (self.since_surfaces and not dagger) else {}
            for ii, idea in enumerate(d["ideas"]):
                role, surf = self._forms(idea, dagger)
                since = []
                if ii in antes:
                    from . import reasoning as R
                    try:
                        since = [[self.w2i[w] for w in ws] for ws in R.since_forms(antes[ii][0], antes[ii][1])]
                    except KeyError:
                        since = []
                ideas.append({**{k: v for k, v in idea.items() if k not in ("roles", "types")}, "tokens": role[0]})
                alts.append(role[1:])
                encs.append(self.rng.choice(surf + since if self.since_surfaces == "all" else surf))
                if self.p_input_surface:
                    mix = self.p_input_surface == "mix" and self.rng.random() < 0.5
                    pins.append(role[0] if mix else self.rng.choice(self._story_forms(idea, dagger)))
                refs.append(idea["tokens"])
                # stage B positives: another surface of the same fact tuple (premise <-> deduction included); for a
                # movement "X goes to L", the position it implies, "X is in L"
                f = idea.get("fact")
                f = tuple(tuple(x) if isinstance(x, list) else x for x in f) if (f is not None and not dagger
                                                                                  and idea.get("train_w") != 0.0) else None
                pos, pkey, pkey2 = None, None, None
                if self.nce_positives == "dropout":
                    if not dagger and idea.get("train_w") != 0.0:
                        pos = role[0]
                        f = pkey = ("surface", tuple(role[0]))          # identity of the sentence, no generator label
                elif f is not None and f[0] == "goes" and self.w2i is not None:
                    from . import reasoning as R
                    pkey = ("at", f[1], f[2])
                    pos = [self.w2i[w] for w in R.premise_forms(pkey)[0]]
                elif f is not None:
                    others = [t for t in surf + since if t != role[0]]
                    if others:
                        pos, pkey = self.rng.choice(others), f
                        if since and pos in since:
                            pkey2 = antes[ii][1]
                poss.append(pos)
                fkeys.append(kid(f)); pkeys.append(kid(pkey)); pkeys2.append(kid(pkey2))
                # template = (predicate, role): same-template facts about different entities (loss.w_memsep)
                tkeys.append(tmap.setdefault((f[0], idea.get("step")), len(tmap)) if f is not None else -1)
            main_docs.append({**d, "ideas": ideas})
        batch = self._call_plain(main_docs)
        M = batch["M"]
        # prefixes of the main rows (canonical stream), reused by the alt rows of the same idea
        prefix = batch["win"][:, :M]
        alt_idea = [i for i, a in enumerate(alts) for _ in a]
        alt_rows = [t for a in alts for t in a]
        if alt_rows:
            La = max(map(len, alt_rows))
            aw = torch.full((len(alt_rows), M + La), PAD, dtype=torch.long)
            aw[:, :M] = prefix[torch.tensor(alt_idea)]
            for r, t in enumerate(alt_rows):
                aw[r, M:M + len(t)] = torch.tensor(t)
            batch["alt_win"], batch["alt_idea"] = aw, torch.tensor(alt_idea)
            batch["alt_len"] = torch.tensor([len(t) for t in alt_rows])
        Le = max(map(len, encs))
        et = torch.full((len(encs), Le), PAD, dtype=torch.long)
        for r, t in enumerate(encs):
            et[r, :len(t)] = torch.tensor(t)
        batch["enc_tok"], batch["enc_len"] = et, torch.tensor([len(t) for t in encs])
        if pins:
            pt_ = torch.full((len(pins), max(map(len, pins))), PAD, dtype=torch.long)
            for r_, t in enumerate(pins):
                pt_[r_, :len(t)] = torch.tensor(t)
            batch["pin_tok"], batch["pin_len"] = pt_, torch.tensor([len(t) for t in pins])
        # reference rows with the prefix of the ORIGINAL text stream (reference CE, eval only)
        ref_prefix, r = [], 0
        for d in docs:
            hist = [BOS]
            for idea in d["ideas"]:
                ref_prefix.append(hist[-self.m:]); hist = hist + idea["tokens"]; r += 1
        Lr = max(map(len, refs))
        rw = torch.full((len(refs), M + Lr), PAD, dtype=torch.long)
        for i, (p, t) in enumerate(zip(ref_prefix, refs)):
            rw[i, M - len(p):M] = torch.tensor(p)
            rw[i, M:M + len(t)] = torch.tensor(t)
        batch["ref_win"], batch["ref_len"] = rw, torch.tensor([len(t) for t in refs])
        # stage B (InfoNCE): positive surfaces, and fact keys to mask false negatives (same fact / implied position)
        has = [i for i, p_ in enumerate(poss) if p_ is not None]
        if has:
            Lp = max(len(poss[i]) for i in has)
            pt = torch.full((len(has), Lp), PAD, dtype=torch.long)
            for r_, i in enumerate(has):
                pt[r_, :len(poss[i])] = torch.tensor(poss[i])
            batch["pos_tok"], batch["pos_len"] = pt, torch.tensor([len(poss[i]) for i in has])
            batch["pos_idea"] = torch.tensor(has)
        batch["fact_key"], batch["pos_key"] = torch.tensor(fkeys), torch.tensor(pkeys)
        batch["tmpl_key"] = torch.tensor(tkeys)
        if self.since_surfaces:
            batch["pos_key2"] = torch.tensor(pkeys2)
        batch.pop("form_id", None)
        return batch

    def _call_plain(self, docs):
        rows, prefixes, doc_of, pos_of, extra = [], [], [], [], {"roles": [], "types": [], "idea_id": [], "step": [], "train_w": [], "b_eff": [], "candidates": []}
        seqs, seq_deds = [], []
        B = len(docs)
        Nmax = max(len(d["ideas"]) for d in docs)
        doc_index = torch.full((B, Nmax), -1, dtype=torch.long)
        for b, d in enumerate(docs):
            history = [BOS]
            hist_ded = [False]
            for j, idea in enumerate(d["ideas"]):
                doc_index[b, j] = len(rows)
                rows.append(idea["tokens"])
                if self.full:
                    pre = history if self.max_prefix is None else history[-self.max_prefix:]
                else:
                    pre = history[-self.m:]
                prefixes.append(pre)
                doc_of.append(b)
                pos_of.append(j)
                for k in extra:
                    extra[k].append(idea.get(k))
                history = history + idea["tokens"]
                hist_ded = hist_ded + [idea.get("step") == "deduction"] * len(idea["tokens"])
            seqs.append(history)
            seq_deds.append(hist_ded)
        L = max(len(t) for t in rows)
        M = self.m if not self.full else max(len(p) for p in prefixes)
        NI = len(rows)
        win = torch.full((NI, M + L), PAD, dtype=torch.long)
        for i, (p, t) in enumerate(zip(prefixes, rows)):
            win[i, M - len(p):M] = torch.tensor(p)
            win[i, M:M + len(t)] = torch.tensor(t)
        T = max(len(s) for s in seqs)
        seq = torch.full((B, T), PAD, dtype=torch.long)
        for b, s in enumerate(seqs):
            seq[b, :len(s)] = torch.tensor(s)
        seq_ded = torch.zeros((B, T), dtype=torch.bool)
        for b, sd in enumerate(seq_deds):
            seq_ded[b, :len(sd)] = torch.tensor(sd)
        batch = {
            "seq_ded": seq_ded,
            "win": win, "M": M, "idea_len": torch.tensor([len(t) for t in rows]),
            "idea_tok": win[:, M:].clone(), "idea_doc": torch.tensor(doc_of), "idea_pos": torch.tensor(pos_of),
            "doc_index": doc_index, "seq": seq,
        }
        for k in ("roles", "types"):
            if extra[k][0] is not None:
                t = torch.full((NI, L), -1, dtype=torch.long)
                for i, r in enumerate(extra[k]):
                    t[i, :len(r)] = torch.tensor(r)
                batch[k] = t
        if extra["idea_id"][0] is not None:
            batch["idea_id"] = torch.tensor(extra["idea_id"])
        if any(i.get("fact") is not None for d in docs for i in d["ideas"]):   # loss.w_qoracle
            batch["qo_ptr"] = torch.tensor([p for d in docs for p in qoracle_ptr(d["ideas"])])
        if any("rel" in i for d in docs for i in d["ideas"]):   # step 2 (chantier 3 bis): relation labels of each idea
            REL = {None: 0, "none": 0, "justified_by": 1, "sequence": 2}
            batch["rel_id"] = torch.tensor([REL[i.get("rel")] for d in docs for i in d["ideas"]])
            batch["rel_off"] = torch.tensor([j - i["rel_ptr"] if i.get("rel_ptr", -1) >= 0 else 0
                                             for d in docs for j, i in enumerate(d["ideas"])])
        if any("support" in d for d in docs):   # bAbI: supporting-fact annotation (ORACLE variants of P only)
            batch["idea_support"] = torch.tensor([j in set(d.get("support", [])) for d in docs
                                                  for j in range(len(d["ideas"]))])
        if any(x is not None for x in extra["b_eff"]):  # synthetic v2: number of valid next steps (0 = premise)
            batch["b_eff"] = torch.tensor([int(x or 0) for x in extra["b_eff"]])
        if any(x for x in extra["candidates"]):  # synthetic v2: exact set of valid next deductions per idea
            batch["cands"] = [[tuple(tuple(y) if isinstance(y, list) else y for y in c) for c in (x or [])] for x in extra["candidates"]]
        if extra["step"][0] is not None and "form" in docs[0]["ideas"][0]:  # "separate form": surface template
            from .reasoning import template_id
            batch["form_id"] = torch.tensor([template_id(i) for d in docs for i in d["ideas"]])
        if extra["step"][0] is not None:  # synthetic v2: premises are unpredictable by construction
            batch["is_deduction"] = torch.tensor([st == "deduction" for st in extra["step"]])
        if any(x is not None for x in extra["train_w"]):  # DAgger docs: model's own deductions are context only
            batch["idea_w"] = torch.tensor([1.0 if x is None else float(x) for x in extra["train_w"]])
        return batch


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class DocLoader:
    """Shuffling batch iterator over chunked documents, batching by token budget."""

    def __init__(self, docs, collate, batch_tokens, shuffle=True, seed=0, max_docs=None):
        self.docs = docs
        self.collate = collate
        self.batch_tokens = batch_tokens
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.max_docs = max_docs
        self.lens = docs.lens if hasattr(docs, "lens") else [sum(len(i["tokens"]) for i in d["ideas"]) for d in docs]

    def batches(self):
        if self.shuffle:
            idx = np.random.default_rng(self.rng.randrange(2**31)).permutation(len(self.docs)).tolist()
        else:
            idx = list(range(len(self.docs)))
        cur, ntok = [], 0
        for i in idx:
            cur.append(i)
            ntok += self.lens[i]
            if ntok >= self.batch_tokens or (self.max_docs and len(cur) >= self.max_docs):
                yield self.collate([self.docs[j] for j in cur])
                cur, ntok = [], 0
        if cur and not self.shuffle:
            yield self.collate([self.docs[j] for j in cur])

    def __iter__(self):
        while True:
            yield from self.batches()


class Prefetcher:
    """Collates batches in a background thread (the collation is pure Python and otherwise sits on the
    critical path of CPU-bound small-model training)."""

    def __init__(self, iterable, depth=4):
        import queue
        import threading
        self.q = queue.Queue(maxsize=depth)
        self.it = iter(iterable)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        pin = torch.cuda.is_available()
        for b in self.it:
            self.q.put({k: (v.pin_memory() if torch.is_tensor(v) and pin else v) for k, v in b.items()})

    def __iter__(self):
        return self

    def __next__(self):
        return self.q.get()


class _ListDS(torch.utils.data.Dataset):
    def __init__(self, docs):
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        return self.docs[i]


def _ignore_sigint(_worker_id):
    """Data workers ignore Ctrl-C: the training process saves last.pt and exits cleanly (EXIT_INTERRUPTED)."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def collate_seed(seed, epoch, worker_id):
    """Seed of the collator's RNG in data worker `worker_id` during `epoch` (distinct per seed, epoch and worker)."""
    return (int(seed) * 1_000_003 + int(epoch)) * 1_009 + int(worker_id) + 1


class _WorkerInit:
    """Worker init: ignore Ctrl-C and (reseed=True, from 30/09) reseed the forked collator's RNG with
    collate_seed(seed, epoch, worker). Before 30/09 every worker restarted each epoch from the SAME collator RNG state
    (forked from the main process, which never draws), so the surface draws were replayed across workers and epochs."""

    def __init__(self, collate, seed, epoch, reseed):
        self.collate, self.seed, self.epoch, self.reseed = collate, seed, epoch, reseed

    def __call__(self, worker_id):
        _ignore_sigint(worker_id)
        rng = getattr(self.collate, "rng", None)
        if self.reseed and rng is not None:
            rng.seed(collate_seed(self.seed, self.epoch, worker_id))


def epoch_batches(loader: "DocLoader", seed, epoch):
    """The token-budget batches (lists of doc indices) of one epoch of worker_iterator."""
    rng = random.Random(seed + epoch)
    idx = list(range(len(loader.docs)))
    rng.shuffle(idx)
    batches, cur, ntok = [], [], 0
    for i in idx:
        cur.append(i)
        ntok += loader.lens[i]
        if ntok >= loader.batch_tokens:
            batches.append(cur)
            cur, ntok = [], 0
    return batches


def worker_iterator(loader: "DocLoader", num_workers=3, seed=0, start_batch=0, restore_rng=None, reseed=True):
    """Endless iterator of collated batches built in worker *processes* (no GIL contention with the
    training loop). Batches are the same token-budget batches as DocLoader, re-shuffled every epoch.
    Resume (start_batch > 0): whole epochs are skipped without collation; the current epoch is replayed from its
    start and its first batches discarded, so every worker makes the same random draws as in an uninterrupted run
    (same data stream). restore_rng() is called where the uninterrupted run's global RNG state was saved: after the
    current epoch's DataLoader iterator is created (it draws a worker seed from the global torch RNG) when resuming
    mid-epoch, before it at an epoch boundary."""
    epoch, skip = 0, start_batch
    while True:
        batches = epoch_batches(loader, seed, epoch)
        if skip >= len(batches) and skip > 0:
            skip -= len(batches); epoch += 1
            continue
        if restore_rng is not None and skip == 0:
            restore_rng(); restore_rng = None
        dl = torch.utils.data.DataLoader(_ListDS(loader.docs), batch_sampler=batches, collate_fn=loader.collate,
                                         num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                         prefetch_factor=4 if num_workers > 0 else None,
                                         multiprocessing_context="fork" if num_workers > 0 else None,
                                         persistent_workers=False,
                                         worker_init_fn=_WorkerInit(loader.collate, seed, epoch, reseed) if num_workers > 0 else None)
        itr = iter(dl)
        for _ in range(skip):
            next(itr)
        skip = 0
        if restore_rng is not None:
            restore_rng(); restore_rng = None
        yield from itr
        epoch += 1


def qoracle_ptr(ideas):
    """Per idea: position in the story of the premise that states the object's position, for an F1 deduction about
    that object (fact ("objin", obj, ...)); -1 otherwise (the last such premise before the deduction)."""
    as_f = lambda f: tuple(tuple(x) if isinstance(x, list) else x for x in f) if f is not None else None
    out = []
    for j, idea in enumerate(ideas):
        f, p = as_f(idea.get("fact")), -1
        if idea.get("step") == "deduction" and f is not None and f[0] == "objin":
            for i in range(j):
                g = as_f(ideas[i].get("fact"))
                if ideas[i].get("step") == "premise" and g is not None and g[0] == "objin" and g[1] == f[1]:
                    p = i
        out.append(p)
    return out
