"""TinyStories preprocessing: byte-level BPE (8k), causal S1 segmentation, compact array storage.

python -m ideajepa.data.tinystories --out data/tinystories

S1 rule (causal, see REPORT ambiguity 8): an idea ends right after any token whose text, with trailing
spaces and quote characters removed, ends with '.', '!' or '?'. Ideas are capped at 48 tokens.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
QUOTES = "\"'”’“‘)"


def end_token_mask(tok):
    V = tok.get_vocab_size()
    mask = np.zeros(V, dtype=bool)
    for i in range(V):
        s = tok.decode([i]).rstrip().rstrip(QUOTES).rstrip()
        if s.endswith((".", "!", "?")):
            mask[i] = True
    return mask


def segment(ids, is_end, cap):
    ideas, cur = [], []
    for t in ids:
        cur.append(t)
        if is_end[t] or len(cur) >= cap:
            ideas.append(cur)
            cur = []
    if cur:
        ideas.append(cur)
    return ideas


class ArrayDocs:
    """Lazy view of pre-chunked documents stored as flat arrays."""

    def __init__(self, path: Path, split: str, max_docs=None):
        z = np.load(path / f"{split}.npz")
        self.tokens = z["tokens"]
        self.idea_off = z["idea_off"]
        self.doc_off = z["doc_off"]
        n = len(self.doc_off) - 1
        self.n = min(n, max_docs) if max_docs else n
        tok_start = self.idea_off[self.doc_off[:-1]]
        tok_end = self.idea_off[self.doc_off[1:]]
        self.lens = (tok_end - tok_start)[:self.n]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        a, b = self.doc_off[i], self.doc_off[i + 1]
        offs = self.idea_off[a:b + 1]
        toks = self.tokens[offs[0]:offs[-1]].astype(np.int64)
        base = offs[0]
        return {"ideas": [{"tokens": toks[offs[j] - base:offs[j + 1] - base].tolist()} for j in range(b - a)]}


def build(out: Path, vocab_size=8000, cap=48, max_ideas=32, max_tokens=512, n_tok_train=300_000,
          segmenter=None):
    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer, Tokenizer

    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")
    tok_path = out / "tokenizer.json"
    if tok_path.exists():
        tok = Tokenizer.from_file(str(tok_path))
    else:
        tok = ByteLevelBPETokenizer()
        sample = ds["train"].shuffle(seed=0).select(range(n_tok_train))["text"]
        tok.train_from_iterator(sample, vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIALS)
        tok.save(str(tok_path))
        tok = Tokenizer.from_file(str(tok_path))
    is_end = end_token_mask(tok)
    print("vocab", tok.get_vocab_size(), "end tokens", int(is_end.sum()))
    for split, name in (("train", "train"), ("validation", "val")):
        texts = [t.strip() for t in ds[split]["text"] if t and t.strip()]
        tokens, idea_off, doc_off = [], [0], [0]
        ntok = 0
        B = 20000
        for i in range(0, len(texts), B):
            enc = tok.encode_batch(texts[i:i + B])
            for e in enc:
                ideas = segmenter(e.ids) if segmenter else segment(e.ids, is_end, cap)
                cur_i, cur_t = 0, 0
                for idea in ideas:
                    if cur_i and (cur_i >= max_ideas or cur_t + len(idea) > max_tokens):
                        doc_off.append(len(idea_off) - 1)
                        cur_i, cur_t = 0, 0
                    tokens.append(np.asarray(idea, dtype=np.uint16))
                    ntok += len(idea)
                    idea_off.append(ntok)
                    cur_i += 1
                    cur_t += len(idea)
                doc_off.append(len(idea_off) - 1)
            print(split, i + B, ntok, flush=True)
        np.savez(out / f"{name}.npz", tokens=np.concatenate(tokens), idea_off=np.asarray(idea_off, np.int64),
                 doc_off=np.asarray(doc_off, np.int64))
        n_ideas = len(idea_off) - 1
        print(f"{name}: {len(doc_off)-1} chunks, {n_ideas} ideas, {ntok} tokens, {ntok/n_ideas:.1f} tok/idea")
    torch.save({"vocab_size": tok.get_vocab_size(), "is_end": is_end, "tokenizer": str(tok_path)},
               out / "meta.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tinystories")
    a = ap.parse_args()
    build(Path(a.out))
