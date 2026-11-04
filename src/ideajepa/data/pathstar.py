"""Path-Star graphs (Bachmann & Nagarajan 2024, "The pitfalls of next-token prediction"), reproducing the original
generator (github.com/gregorbachmann/Next-Token-Failures, data/graphs.py::star_graph) and its numeral tokenisation:
    sequence = a b | c d | ... | y z / s g = p_1 p_2 ... p_l
edges shuffled, start s = centre, goal g = end of one arm, path = l nodes from s to g (s and g included);
node values in {0, ..., N-1} (the original code; NextLat's text says 1..100 for N = 100, same distribution).
Idea segmentation (Idea-JEPA only; B0/B3 read the concatenation, identical to the original format):
    premises  = one idea per edge ("a b |", the last edge without "|") + one idea "/ s g ="
    deduction = ONE idea: the whole path "p_1 ... p_l" (the verbalisation of a single decision: which arm)
usage: python -m ideajepa.data.pathstar --deg 7 --path 7 --nodes 100 --out data/pathstar_d7_l7"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
SEPS = ["|", "/", "="]


def build_vocab(n_nodes):
    return SPECIALS + [str(i) for i in range(n_nodes)] + SEPS


def star_graph(deg, path_len, n_nodes, rng):
    """Faithful port of the original star_graph (numpy RNG replaced by random.Random)."""
    source = rng.randrange(n_nodes)
    goal = rng.randrange(n_nodes)
    while goal == source:
        goal = rng.randrange(n_nodes)
    path = [source]
    edges = []
    for _ in range(path_len - 2):
        node = rng.randrange(n_nodes)
        while node in path or node == goal:
            node = rng.randrange(n_nodes)
        path.append(node)
    path.append(goal)
    for i in range(len(path) - 1):
        edges.append([path[i], path[i + 1]])
    deg_nodes = set()
    for _ in range(deg - 1):
        node = source
        nxt = rng.randrange(n_nodes)
        length = 1
        while length < path_len:
            if nxt not in deg_nodes and nxt not in path:
                edges.append([node, nxt])
                deg_nodes.add(nxt)
                node = nxt
                length += 1
            nxt = rng.randrange(n_nodes)
    rng.shuffle(edges)
    return path, edges, source, goal


def arms(edges, source):
    """All arms from the source (lists of nodes starting at source), for diagnostics."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    out = []
    for first in adj.get(source, []):
        arm = [source, first]
        while arm[-1] in adj:
            arm.append(adj[arm[-1]][0])
        out.append(arm)
    return out


def to_doc(path, edges, source, goal, w2i):
    ideas = []
    for i, (a, b) in enumerate(edges):
        toks = [str(a), str(b)] + (["|"] if i < len(edges) - 1 else [])
        ideas.append({"tokens": [w2i[t] for t in toks], "step": "premise", "fact": ("edge", a, b)})
    ideas.append({"tokens": [w2i[t] for t in ["/", str(source), str(goal), "="]], "step": "premise",
                  "fact": ("query", source, goal)})
    ideas.append({"tokens": [w2i[str(n)] for n in path], "step": "deduction", "fact": ("path",) + tuple(path)})
    return {"ideas": ideas, "family": "pathstar", "graph": {"edges": edges, "source": source, "goal": goal, "path": path}}


def build(out: Path, deg, path_len, n_nodes, n_train=200_000, n_val=2_000, n_test=20_000, seed=0):
    vocab = build_vocab(n_nodes); w2i = {w: i for i, w in enumerate(vocab)}
    out.mkdir(parents=True, exist_ok=True)
    for split, n, s in (("train", n_train, seed), ("val", n_val, seed + 1), ("test", n_test, seed + 2)):
        rng = random.Random(s)
        docs = [to_doc(*star_graph(deg, path_len, n_nodes, rng), w2i) for _ in range(n)]
        torch.save({"docs": docs, "vocab": vocab}, out / f"{split}.pt")
        ntok = sum(len(i["tokens"]) for d in docs for i in d["ideas"])
        print(f"{split}: {n} graphs, {ntok} tokens, {ntok / n:.1f} tokens/graph, "
              f"{len(docs[0]['ideas']) - 2} edges/graph, path {len(docs[0]['graph']['path'])} nodes")
    torch.save({"vocab": vocab, "deg": deg, "path_len": path_len, "n_nodes": n_nodes}, out / "meta.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=int, required=True); ap.add_argument("--path", type=int, required=True)
    ap.add_argument("--nodes", type=int, default=100); ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=200_000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(Path(a.out), a.deg, a.path, a.nodes, n_train=a.n_train, seed=a.seed)
