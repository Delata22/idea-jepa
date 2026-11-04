"""Training entry point:  python -m ideajepa.train --config configs/syn_v_full.yaml [--set key=val ...]"""
from __future__ import annotations

import argparse
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

import torch

from .data.dataset import Prefetcher, to_device, worker_iterator
from .data.loaders import make_loader, vocab_size
from .model.baseline import GPT
from .model.ideajepa import IdeaJEPA
from .model.objective import effective_rank, idea_losses
from .utils.common import ROOT, Logger, load_config, set_seed


def model_config(cfg):
    m = dict(cfg["model"])
    m["vocab_size"] = vocab_size(cfg["data"])
    m.setdefault("max_ideas", cfg["data"]["max_ideas"] + 1)
    L = m["max_idea_len"]
    if m.get("full_prefix"):
        m["max_window"] = cfg["data"]["max_ideas"] * L + 2
    else:
        m["max_window"] = m["m_prefix"] + L
    m["max_seq"] = cfg["data"].get("max_tokens") or cfg["data"]["max_ideas"] * L + 1
    m["max_seq"] += 1
    if m.get("p_id_keys"):   # entity word lists (config, in clear) -> word ids of the data's vocabulary
        vocab = torch.load(ROOT / cfg["data"]["path"] / "meta.pt", weights_only=False)["vocab"]
        w2i = {w: i for i, w in enumerate(vocab)}
        m["p_id_name_ids"] = [w2i[w] for w in m["p_id_keys"]["names"]]
        m["p_id_obj_ids"] = [w2i[w] for w in m["p_id_keys"]["objects"]]
    return m


def build_model(cfg):
    m = model_config(cfg)
    if m["model"] == "ideajepa":
        return IdeaJEPA(m)
    return GPT(m)


def compute_losses(model, batch, cfg, p_gen=0.0, force_path=None, train=True, step=None):
    if isinstance(model, IdeaJEPA):
        return idea_losses(model, batch, cfg["loss"], p_gen=p_gen, force_path=force_path, train=train, step=step)
    return model.losses(batch)


@torch.no_grad()
def evaluate(model, loader, cfg, device, max_batches=None):
    model.eval()
    agg, n = {}, 0
    ranks = []
    for i, batch in enumerate(loader.batches()):
        if max_batches and i >= max_batches:
            break
        batch = to_device(batch, device)
        ntok = int(batch["idea_len"].sum())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if isinstance(model, IdeaJEPA):
                res = {}
                for path in ("gen", "recon"):
                    out, aux = compute_losses(model, batch, cfg, force_path=path, train=False)
                    res[f"ce_{path}"] = out["L_tok"].item()
                    for k in ("ce_min_role", "ce_main", "ce_ref"):   # stage A measures, both paths
                        if k in out:
                            res[f"{k}_{path}"] = float(out[k])
                    if path == "recon":
                        for k in ("L_jepa", "L_vic", "L_exh", "L_sparse"):
                            res[k] = out[k].item()
                        for k, v in aux["stats"].items():
                            res[k] = v
                        if len(ranks) < 4:
                            ranks.append(effective_rank(aux["v"].reshape(-1, aux["v"].shape[-1])))
                        if cfg["loss"].get("w_transcr", 0.0) > 0:   # stage C: verdicts of P's decoded predictions
                            from .model.objective import transcr_eval_stats
                            res.update(transcr_eval_stats(model, batch, _vocab(cfg)))
                    else:
                        res["rho_end_gen"] = aux["stats"]["rho_end"]
                        res["frac_exhausted_gen"] = aux["stats"]["frac_exhausted"]
            else:
                out, _ = compute_losses(model, batch, cfg, train=False)
                res = {"ce_gen": out["L_tok"].item()}
        for k, v in res.items():
            agg[k] = agg.get(k, 0.0) + v * ntok
        n += ntok
    model.train()
    out = {k: v / n for k, v in agg.items()}
    out["ppl_gen"] = math.exp(out["ce_gen"])
    if "ce_recon" in out:
        out["ppl_recon"] = math.exp(out["ce_recon"])
    if ranks:
        out["eff_rank_v"] = sum(ranks) / len(ranks)
    return out


_VOCAB = {}


def _vocab(cfg):
    path = cfg["data"]["path"]
    if path not in _VOCAB:
        _VOCAB[path] = torch.load(ROOT / path / "meta.pt", weights_only=False)["vocab"]
    return _VOCAB[path]


def lr_at(step, tc):
    warm, total = tc["warmup"], tc["steps"]
    if step < warm:
        return tc["lr"] * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return tc["lr"] * (tc.get("min_lr_frac", 0.1) + (1 - tc.get("min_lr_frac", 0.1)) * 0.5 * (1 + math.cos(math.pi * prog)))


EXIT_INTERRUPTED = 75   # SIGTERM / SIGINT: last.pt saved, the run resumes when relaunched (scripts/_sweep.sh: not a failure)


def rng_state():
    st = {"torch": torch.get_rng_state(), "random": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    # last.pt is loaded with map_location=device: the RNG byte tensors must come back to the CPU
    torch.set_rng_state(st["torch"].cpu()); random.setstate(st["random"]); np.random.set_state(st["numpy"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in st["cuda"]])


def save_last(path, model, opt, next_step, cfg, tokens_seen, elapsed, val_loader):
    """Atomic resume checkpoint (write then rename): model (E_ema and buffers included) + optimizer + next step +
    tokens + global RNG states + the validation collator's RNG (random surfaces of the eval batches). The
    learning-rate schedule and p_gen are functions of the step; the training data stream is replayed (worker_iterator)."""
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": next_step, "cfg": cfg,
                "tokens_seen": tokens_seen, "elapsed": elapsed, "rng": rng_state(),
                "val_rng": getattr(val_loader.collate, "rng", random.Random()).getstate()}, tmp)
    os.replace(tmp, path)


def save_ckpt(path, model, opt, step, cfg, tokens_seen):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt else None, "step": step,
                "cfg": cfg, "tokens_seen": tokens_seen}, path)


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["cfg"]).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [m for m in missing if m != "e_slot_norm"]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {missing} unexpected {unexpected}")
    model.eval()
    return model, ck["cfg"], ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.seed is not None:
        cfg["train"]["seed"] = a.seed
    tc = cfg["train"]
    seed = tc.get("seed", 0)
    name = a.name or f"{cfg['name']}_s{seed}"
    run_dir = ROOT / "runs" / cfg.get("group", "misc") / name
    set_seed(seed)
    device = tc.get("device", "cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    last_path = run_dir / "last.pt"
    resume = last_path.exists() and not (run_dir / "final.json").exists()

    # train.data_seed (notes §35.26): seed of the training data stream (order of the stories, collator draws), separate from
    # train.seed (weight init, dropout, gen-path draws); default = train.seed (unchanged behaviour)
    data_seed = tc.get("data_seed", seed)
    train_loader = make_loader(cfg, "train", shuffle=True, seed=data_seed)
    val_loader = make_loader(cfg, "val", shuffle=False)
    model = build_model(cfg).to(device)
    if tc.get("init_from"):  # stage-B training: start from a stage-A checkpoint
        ck0 = torch.load(ROOT / tc["init_from"], map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck0["model"], strict=False)
        print(f"[init_from] {tc['init_from']}: missing {len(missing)} unexpected {len(unexpected)}", flush=True)
        if hasattr(model, "E_ema"):
            model.E_ema.load_state_dict(model.E.state_dict())
    for mod_name in tc.get("freeze", []) or []:
        for p_ in getattr(model, mod_name).parameters():
            p_.requires_grad_(False)
        print(f"[freeze] {mod_name}", flush=True)
    nparam = model.n_params()
    ck_last = torch.load(last_path, map_location=device, weights_only=False) if resume else None
    start_step = ck_last["step"] if resume else 0
    log = Logger(run_dir, resume_step=start_step if resume else None)
    if resume:
        model.load_state_dict(ck_last["model"])
        print(f"[{name}] resuming from {last_path} at step {start_step}", flush=True)
    else:
        log.json("config.json", {**cfg, "n_params": nparam})
    print(f"[{name}] params: {nparam/1e6:.2f}M  run_dir={run_dir}", flush=True)

    trainable = [p for p in model.trainable_parameters() if p.requires_grad]
    decay = [p for p in trainable if p.dim() >= 2]
    no_decay = [p for p in trainable if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.get("wd", 0.1)},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["lr"], betas=(0.9, 0.95), fused=device == "cuda")
    is_idea = isinstance(model, IdeaJEPA)
    if cfg["loss"].get("transcr_guard", "none") == "oracle":   # the oracle guard parses decoded sentences
        model.vocab = _vocab(cfg)
    t0 = time.time()
    tokens_seen = 0
    restore = None
    if resume:
        opt.load_state_dict(ck_last["opt"])
        tokens_seen = ck_last["tokens_seen"]
        t0 -= ck_last.get("elapsed", 0.0)
        if hasattr(val_loader.collate, "rng"):
            val_loader.collate.rng.setstate(ck_last["val_rng"])
        rng_saved = ck_last["rng"]
        restore = lambda: set_rng_state(rng_saved)
        del ck_last
    nw = tc.get("num_workers", 3)
    accum = int(tc.get("grad_accum", 1))             # micro-batches per optimisation step (memory; same effective batch)
    # train.curriculum_steps = N with data.curriculum_max_ideas = m (notes §35.26): the first N steps draw only from the stories
    # of <= m ideas (their own stream, data seed + 7919), then the full stream starts from its beginning. Resume inside the
    # curriculum replays the short stream; the RNG restore targets the full stream's position only (approximate there).
    cur_n = int(tc.get("curriculum_steps", 0))
    cur_it = None
    if cur_n:
        from .data.dataset import DocLoader
        m_ = int(cfg["data"]["curriculum_max_ideas"])
        short = [d for d in train_loader.docs if len(d["ideas"]) <= m_]
        cur_loader = DocLoader(short, train_loader.collate, train_loader.batch_tokens, shuffle=True, seed=data_seed + 7919)
        if start_step < cur_n:
            cur_it = worker_iterator(cur_loader, max(nw, 1), data_seed + 7919, start_batch=start_step * accum,
                                     reseed=tc.get("collate_reseed", True))
        print(f"[{name}] curriculum: {cur_n} steps on {len(short)} / {len(train_loader.docs)} stories (<= {m_} ideas)", flush=True)
    main_start = max(0, start_step - cur_n) * accum
    if nw > 0:
        it = worker_iterator(train_loader, nw, data_seed, start_batch=main_start, restore_rng=restore,
                             reseed=tc.get("collate_reseed", True))   # False = stream of the runs before 30/09
    else:   # in-process collation (DocLoader order); resume = replay the batches already seen
        src_it = iter(train_loader)
        for _ in range(main_start):
            next(src_it)
        if restore is not None:
            restore()
        it = Prefetcher(src_it)
    if tc.get("compile"):
        model.compile_parts(tc["compile"])
    # SIGTERM / SIGINT: finish the current step, save last.pt, exit with EXIT_INTERRUPTED
    stop = {"sig": None}
    def _on_signal(signum, frame):
        stop["sig"] = signum
    for s_ in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s_, _on_signal)
    ckpt_every = tc.get("ckpt_every", 1000)          # 0 = no resume checkpoints
    stop_at = tc.get("stop_at")                      # tests: behave as if interrupted after this many steps
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    step = start_step
    last_log = t0
    ev = {}
    for step in range(start_step, tc["steps"]):
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        p_gen = tc.get("p_gen_max", 0.5) * min(1.0, step / max(1, tc.get("p_gen_ramp", 0.5) * tc["steps"]))
        # train.grad_accum = k: k micro-batches of train.batch_tokens per optimisation step (same effective batch as one
        # batch of k x batch_tokens); losses averaged; statistics from the last micro-batch
        lin_n = int(tc.get("p_linear_steps", 0))   # notes §35.27: P's MLP activations linear for the first N steps
        if lin_n and step in (start_step, lin_n):
            for b_ in model.P.blocks:
                b_.mlp.linear = step < lin_n
        opt.zero_grad(set_to_none=True)
        for _micro in range(accum):
            batch = to_device(next(cur_it if (cur_it is not None and step < cur_n) else it), device)
            with autocast:
                out, aux = compute_losses(model, batch, cfg, p_gen=p_gen, train=True, step=step)
            (out["loss"] / accum).backward()
            if _micro < accum - 1:
                tokens_seen += int(batch["idea_len"].sum())
        gn = torch.nn.utils.clip_grad_norm_(trainable, tc.get("grad_clip", 1.0))
        opt.step()
        if is_idea:
            model.ema_update(cfg["loss"].get("ema_tau", 0.996) if cfg["loss"].get("ema_tau", 0.996) > 0 else 0.0)
        tokens_seen += int(batch["idea_len"].sum())
        if step % tc.get("log_every", 50) == 0:
            now = time.time()
            row = {k: v.item() if torch.is_tensor(v) else v for k, v in out.items()}
            row.update(aux.get("stats", {}) if isinstance(aux, dict) else {})
            row.update({"lr": lr, "p_gen": p_gen, "grad_norm": gn.item(), "tokens": tokens_seen,
                        "time": now - t0})
            log.log("train", step, row)
            if step % (tc.get("log_every", 50) * 10) == 0:
                print(f"[{name}] step {step} loss {row['loss']:.3f} tok {row['L_tok']:.3f} "
                      f"{(now - t0):.0f}s", flush=True)
        if (step + 1) % tc["eval_every"] == 0 or step + 1 == tc["steps"]:
            ev = evaluate(model, val_loader, cfg, device, tc.get("eval_batches"))
            ev.update({"tokens": tokens_seen, "time": time.time() - t0})
            log.log("val", step + 1, ev)
            print(f"[{name}] eval step {step+1}: " + " ".join(f"{k}={v:.4g}" for k, v in ev.items()
                                                              if isinstance(v, float)), flush=True)
        if step + 1 in (tc.get("snapshot_at") or []) and step + 1 < tc["steps"]:
            # intermediate model-only checkpoint for evaluation, in a sibling run dir "<name>@<step>" (ckpt.pt + config.json)
            snap = run_dir.parent / f"{name}@{step + 1}"
            snap.mkdir(parents=True, exist_ok=True)
            save_ckpt(snap / "ckpt.pt", model, None, step + 1, cfg, tokens_seen)
            (snap / "config.json").write_text((run_dir / "config.json").read_text())
        interrupted = stop["sig"] is not None or (stop_at is not None and step + 1 == stop_at)
        if step + 1 < tc["steps"] and (interrupted or (ckpt_every and (step + 1) % ckpt_every == 0)):
            save_last(last_path, model, opt, step + 1, cfg, tokens_seen, time.time() - t0, val_loader)
        if interrupted and step + 1 < tc["steps"]:
            log.close()
            print(f"[{name}] interrupted at step {step + 1} (signal {stop['sig']}): {last_path} saved", flush=True)
            sys.exit(EXIT_INTERRUPTED)
        if tc.get("time_limit") and time.time() - t0 > tc["time_limit"]:
            print(f"[{name}] time limit reached at step {step}", flush=True)
            ev = evaluate(model, val_loader, cfg, device, tc.get("eval_batches"))
            ev.update({"tokens": tokens_seen, "time": time.time() - t0})
            log.log("val", step + 1, ev)
            break
    save_ckpt(run_dir / "ckpt.pt", model, None, step + 1, cfg, tokens_seen)
    log.json("final.json", {"step": step + 1, "tokens_seen": tokens_seen, "n_params": nparam,
                            "time": time.time() - t0, **ev})
    log.close()
    print(f"[{name}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
