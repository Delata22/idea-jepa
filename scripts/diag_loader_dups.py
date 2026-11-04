"""Data-loading duplicate check (notes §34.30, step 0): does collation in worker processes (train.num_workers 3)
show the model duplicated stories or duplicated random draws, compared with in-process collation (num_workers 0)?

No training. For each path, the first N training batches exactly as train.py builds them (worker_iterator for
num_workers > 0, DocLoader for 0). Per batch: the stories (fingerprint = hash of all their tokens) and a fingerprint of
the collator's Python RNG state when the batch is collated (the only randomness in collation: surface draws,
p_input_surface, ...). Reported: distinct stories / story slots, stories seen twice inside one epoch, identical
batches, and batches whose collation started from the SAME RNG state as another batch (random draws replayed).

  IDEAJEPA_DEVICE=cpu .venv/bin/python scripts/diag_loader_dups.py --config configs/babi_im.yaml \
      --set data.path=data/babi/qa2_10k --n 200 --out runs/diag/loader_dups_babi_qa2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ideajepa.data.dataset import epoch_batches, worker_iterator
from ideajepa.data.loaders import make_loader
from ideajepa.utils.common import load_config


def _h(obj) -> str:
    return hashlib.md5(repr(obj).encode()).hexdigest()[:16]


class Tagged:
    """Wraps a collator: adds the story fingerprints and the RNG-state fingerprint to each batch (strings only)."""

    def __init__(self, coll):
        self.coll = coll

    @property
    def rng(self):          # the collator's RNG (reseeded per worker and epoch by worker_iterator)
        return self.coll.rng

    def __call__(self, docs):
        rng = getattr(self.coll, "rng", None)
        st = _h(rng.getstate()) if rng is not None else "-"
        b = self.coll(docs)
        b["_docs"] = "|".join(_h([i["tokens"] for i in d["ideas"]]) for d in docs)
        b["_rng"] = st
        return b


def run(cfg, nw, n, seed):
    loader = make_loader(cfg, "train", shuffle=True, seed=seed)
    loader.collate = Tagged(loader.collate)
    it = worker_iterator(loader, nw, seed) if nw > 0 else iter(loader)
    per_epoch = len(epoch_batches(loader, seed, 0))
    out = []
    for _ in range(n):
        b = next(it)
        out.append((b["_docs"].split("|"), b["_rng"]))
    del it
    return out, per_epoch, len(loader.docs)


def stats(rows, per_epoch):
    slots = [d for docs, _ in rows for d in docs]
    ep = [rows[i:i + per_epoch] for i in range(0, len(rows), per_epoch)]
    dup_in_epoch = sum(sum(c - 1 for c in Counter(d for docs, _ in e for d in docs).values()) for e in ep)
    bkeys = Counter(tuple(docs) for docs, _ in rows)
    rngs = Counter(r for _, r in rows)
    return {"batches": len(rows), "story_slots": len(slots), "distinct_stories": len(set(slots)),
            "stories_repeated_within_an_epoch": dup_in_epoch,
            "identical_batches": sum(c - 1 for c in bkeys.values() if c > 1),
            "distinct_rng_states": len(rngs),
            "batches_sharing_rng_state": sum(c for c in rngs.values() if c > 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    res = {"config": a.config, "set": a.set, "n": a.n, "seed": a.seed}
    rows = {}
    for nw in (a.workers, 0):
        r, per_epoch, ndocs = run(cfg, nw, a.n, a.seed)
        rows[nw] = r
        res[f"nw{nw}"] = {**stats(r, per_epoch), "batches_per_epoch": per_epoch, "train_stories": ndocs}
    # same stories in the same order on both paths? (they use different shuffles by construction)
    s3 = [d for docs, _ in rows[a.workers] for d in docs]
    s0 = [d for docs, _ in rows[0] for d in docs]
    res["overlap_story_sets"] = len(set(s3) & set(s0))
    res["same_first_batch"] = rows[a.workers][0][0] == rows[0][0][0]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
