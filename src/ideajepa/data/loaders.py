from __future__ import annotations

from pathlib import Path

import torch

from ..utils.common import ROOT
from .dataset import Collator, DocLoader, chunk_docs


def load_split(dcfg, split):
    kind = dcfg["kind"]
    path = ROOT / dcfg["path"]
    if kind in ("synthetic", "tinystories"):
        obj = torch.load(path / f"{split}.pt", weights_only=False)
        docs, vocab = obj["docs"], obj.get("vocab")
        n = dcfg.get(f"max_docs_{split}")
        if n:
            docs = docs[:n]
        return docs, vocab
    raise ValueError(kind)


def vocab_size(dcfg):
    path = ROOT / dcfg["path"]
    meta = torch.load(path / "meta.pt", weights_only=False)
    return meta.get("vocab_size", len(meta.get("vocab", [])))


def make_loader(cfg, split, shuffle, seed=0):
    dcfg = cfg["data"]
    if dcfg["kind"] == "tinystories":  # pre-chunked flat arrays (see data/tinystories.py)
        from .tinystories import ArrayDocs
        docs = ArrayDocs(ROOT / dcfg["path"], dcfg.get(f"file_{split}", split), dcfg.get(f"max_docs_{split}"))
    else:
        docs, vocab_ = load_split(dcfg, split)
        if dcfg.get("compound"):   # Step 1: "conclusion" (B + pointer to A) | "split" (A then B)
            from .reasoning import split_compound
            if vocab_ is None:
                vocab_ = torch.load(ROOT / dcfg["path"] / "meta.pt", weights_only=False)["vocab"]
            docs = [split_compound(d, dcfg["compound"], vocab_, dcfg.get("rel_source", "connector"),
                                   dcfg.get("rel_f1", "useful")) for d in docs]
        docs = chunk_docs(docs, dcfg["max_ideas"], cfg["model"]["max_idea_len"], dcfg.get("max_tokens"))
    m = cfg["model"]
    w2i = None
    if dcfg.get("surfaces", False):  # stage A: role forms / fact surfaces rendered from the vocabulary
        vocab = torch.load(ROOT / dcfg["path"] / "meta.pt", weights_only=False)["vocab"]
        w2i = {w: i for i, w in enumerate(vocab)}
    coll = Collator(m_prefix=m["m_prefix"], full_prefix=m.get("full_prefix", False),
                    max_prefix=m.get("max_prefix"), surfaces=dcfg.get("surfaces", False), w2i=w2i, seed=seed,
                    p_input_surface=dcfg.get("p_input_surface"), since_surfaces=dcfg.get("since_surfaces"),
                    nce_positives=dcfg.get("nce_positives", "generator"))
    return DocLoader(docs, coll, cfg["train"]["batch_tokens"], shuffle=shuffle, seed=seed)
