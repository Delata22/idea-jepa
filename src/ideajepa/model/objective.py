"""Training objective of Idea-JEPA (all loss terms of spec section 2.7)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .ideajepa import IdeaJEPA, frob_norm


def vicreg(x, eps=1e-4):
    """x: [n, d]. Variance hinge (std >= 1) + off-diagonal covariance."""
    x = x - x.mean(0)
    std = torch.sqrt(x.var(0) + eps)
    var_loss = F.relu(1 - std).mean()
    n, d = x.shape
    cov = (x.T @ x) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss, off.pow(2).sum() / d


@torch.no_grad()
def effective_rank(x):
    s = torch.linalg.svdvals(x.float() - x.float().mean(0))
    p = s / s.sum().clamp_min(1e-12)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()


def per_idea_ce(logits, win, M, length):
    tgt = win[:, M:]
    ce = F.cross_entropy(logits.float().transpose(1, 2), tgt, reduction="none")
    m = (torch.arange(tgt.shape[1], device=tgt.device)[None] < length[:, None]).float()
    return ce, m


def idea_losses(model: IdeaJEPA, batch, lc, p_gen=0.0, force_path=None, train=True,
                gen_rng: torch.Generator | None = None, step=None):
    """lc: loss config. force_path: None | "recon" | "gen". Returns (loss dict, stats dict)."""
    d = model.cfg["d"]
    if train and lc.get("flow_only", False):
        return flow_only_step(model, batch, lc)
    memsep = bool(train and lc.get("w_memsep", 0.0) > 0 and "tmpl_key" in batch and getattr(model.P, "mems", None))
    qoracle = bool(train and lc.get("w_qoracle", 0.0) > 0 and "qo_ptr" in batch and getattr(model.P, "mems", None))
    if getattr(model.P, "mems", None):
        model.P.keep_mem = memsep
        for m_ in model.P.mems:
            m_.keep = memsep
            m_.keep_q = qoracle
    k, v = model.encode(batch)
    with torch.no_grad():
        if lc.get("ema_tau", 0.996) > 0:
            kt, vt = model.E_ema(batch["idea_tok"], batch["idea_len"])
        else:  # A8: no EMA target, stop-gradient of the online encoder
            kt, vt = k.detach(), v.detach()
    if model.cfg.get("plan_cond", False):   # plan variable: target o from E_ema, no gradient, no generator label
        batch["plan_o"] = plan_targets(vt, batch["doc_index"])
    # loss.e_warmup_steps = N (notes §35.6): for the first N training steps E is trained by the reconstruction only: P
    # reads detached slots (no gradient of the prediction losses into E) and InfoNCE is off (unless e_warmup_keep_nce)
    e_warm = bool(train and step is not None and step < lc.get("e_warmup_steps", 0))
    # loss.p_input_noise = sigma (notes §35.28, idea bottleneck): Gaussian noise on the idea slots P READS (keys and values,
    # each in units of its batch standard deviation), training only; E's targets and the decoder are untouched
    sig = lc.get("p_input_noise", 0.0) if train else 0.0
    noisy = lambda t: t + sig * t.detach().float().std().to(t.dtype) * torch.randn_like(t) if sig > 0 else t
    if train and "pin_tok" in batch:   # data.p_input_surface: P reads E(a random story surface) of each idea
        k_in, v_in = model.E(batch["pin_tok"], batch["pin_len"])
        if e_warm:
            k_in, v_in = k_in.detach(), v_in.detach()
        pk, pv = model.predict(noisy(k_in), noisy(v_in), batch)
    elif e_warm:
        pk, pv = model.predict(noisy(k.detach()), noisy(v.detach()), batch)
    else:
        pk, pv = model.predict(noisy(k), noisy(v), batch)
    NI = k.shape[0]
    L_memsep = torch.zeros((), device=k.device)
    memsep_stats = None
    if memsep:   # right after the main P pass (the transcr pass re-runs P)
        L_memsep, memsep_stats = _memsep_loss(model.P.mem_rows, batch, lc)
    L_qoracle = torch.zeros((), device=k.device)
    qoracle_stats = None
    L_plan = torch.zeros((), device=k.device)
    plan_stats = None
    if model.cfg.get("plan_cond", False):
        L_plan, plan_stats = _plan_loss(model, batch)
    L_rel = torch.zeros((), device=k.device)
    rel_stats = None
    if model.cfg.get("rel_cond", False) and "rel_id" in batch:   # step 2: P predicts r_{t+1} (class + pointer)
        L_rel, rel_stats = _rel_loss(model, batch)
    if qoracle:  # same place: the query / keys of the main P pass
        L_qoracle, qoracle_stats = _qoracle_loss(model.P, batch, lc)
        for m_ in model.P.mems:
            m_.keep_q = False
    dev = k.device

    L_score = torch.zeros((), device=k.device)
    khyp = getattr(model.P, "K", 1) > 1
    if khyp:  # multiple-choice learning: K hypotheses, only the closest one (relaxed) is pulled to the target
        pos = (batch["idea_doc"], batch["idea_pos"])
        pk_all, pv_all, scores = (x[pos] for x in model.P.last_hyp)       # [NI, K, r, ds], [NI, K]
        dist = ((pk_all - kt[:, None]).pow(2).sum(-1)
                + (F.normalize(pv_all, dim=-1) - F.normalize(vt[:, None], dim=-1)).pow(2).sum(-1)).mean(-1)
        winner = dist.detach().argmin(-1)                                 # [NI]
        eps = lc.get("khyp_eps", 0.05)                                    # keeps losing hypotheses alive
        jd = (1 - eps) * dist.gather(1, winner[:, None])[:, 0] + eps * dist.mean(-1)
        sd = F.cross_entropy(scores, winner, reduction="none")            # the score learns to pick the winner
        if lc.get("khyp_select", "oracle") == "oracle" and train:
            # train the decoder on the hypothesis that matches the target (in-distribution Z); at eval and
            # generation the best-SCORED hypothesis is used (pk, pv as returned by P)
            idx = winner[:, None, None, None].expand(-1, 1, *pk_all.shape[2:])
            pk, pv = pk_all.gather(1, idx)[:, 0], pv_all.gather(1, idx)[:, 0]
    else:
        jd = ((pk - kt).pow(2).sum(-1) + (F.normalize(pv, dim=-1) - F.normalize(vt, dim=-1)).pow(2).sum(-1)).mean(-1)
    if "is_deduction" in batch:  # down-weight unpredictable ideas (premises of synthetic v2)
        wj = torch.where(batch["is_deduction"], 1.0, lc.get("jepa_premise_weight", 0.1)).float()
        if "idea_w" in batch:
            wj = wj * batch["idea_w"]
        L_jepa = (jd * wj).sum() / wj.sum().clamp_min(1e-6)
        if khyp:
            L_score = (sd * wj).sum() / wj.sum().clamp_min(1e-6)
    else:
        wj = None
        L_jepa = jd.mean()
        if khyp:
            L_score = sd.mean()

    # flow head (phase 2b: trained jointly; 2a uses flow_only_step)
    L_flow = torch.zeros((), device=k.device)
    if model.Fh is not None and lc.get("w_flow", 0.0) > 0:
        L_flow = _flow_loss(model, batch, lc, kt, vt, pk, pv)

    # L_para: paraphrase invariance of E at the idea level. Pairs = ideas of the batch sharing an idea_id
    # (same fact, different formulation and/or story); one random partner per idea; pull normalised slots
    # together (VICReg keeps the space from collapsing).
    L_para = torch.zeros((), device=dev)
    if lc.get("w_para", 0.0) > 0 and "idea_id" in batch:
        ids = batch["idea_id"]
        same = (ids[:, None] == ids[None, :]) & ~torch.eye(NI, dtype=torch.bool, device=dev)
        has = same.any(1)
        if has.any():
            j = (torch.rand(NI, NI, device=dev) * same).argmax(1)
            vn = F.normalize(v, dim=-1)
            L_para = (vn[has] - vn[j[has]]).pow(2).sum(-1).mean(-1).mean()
            if lc.get("para_keys", False):   # pulling keys together collapsed them (26/09): off by default
                L_para = L_para + (k[has] - k[j[has]]).pow(2).sum(-1).mean(-1).mean()

    # L_mask: masked-idea prediction (stage A). One random deduction per document is masked; the bidirectional
    # predictor must recover its E_ema slots from the rest of the document.
    # L_nce (stage B, §30): InfoNCE on E's slots, [k ; v/|v|] concatenated and L2-normalised (never keys alone).
    # Positives: another surface of the same fact tuple, or the position a movement implies; negatives: the rest of
    # the batch (whole stories, so same-template facts on other objects of the same story are in it), with false
    # negatives (same fact / same implied position) masked.
    L_nce = torch.zeros((), device=dev)
    if lc.get("w_nce", 0.0) > 0 and "pos_tok" in batch and not (e_warm and not lc.get("e_warmup_keep_nce", False)):
        L_nce = _info_nce(model, batch, k, v, lc.get("nce_temp", 0.1), lc.get("nce_neg", "batch"))

    # L_transcr (stage C, §30): re-encoding of P's prediction. Greedy decoding of Ẑ (no gradient), E_ema of the decoded
    # sentence as a stop-gradient target, Ẑ pulled to it (JEPA distance). The gradient does not cross the decoding.
    L_transcr = torch.zeros((), device=dev)
    transcr_stats = None
    if train and lc.get("w_transcr", 0.0) > 0 and (step is None or step >= lc.get("transcr_start_step", 2000)):
        L_transcr, transcr_stats = _transcr_loss(model, batch, lc, k, v, pk, pv)

    L_mask = torch.zeros((), device=dev)
    if lc.get("w_mask", 0.0) > 0 and getattr(model, "PM", None) is not None:
        ded_m = batch["is_deduction"] if "is_deduction" in batch else torch.ones(NI, dtype=torch.bool, device=dev)
        docs_ = batch["idea_doc"]
        score = torch.rand(NI, device=dev) * ded_m
        # argmax of a random score per document -> one masked deduction per doc (docs without deduction: none)
        best = torch.zeros(int(docs_.max()) + 1, device=dev).scatter_reduce(0, docs_, score, reduce="amax")
        masked = (score == best[docs_]) & ded_m & (score > 0)
        if masked.any():
            mk, mv = model.PM(k, v, batch["doc_index"], masked)
            L_mask = ((mk[masked] - kt[masked]).pow(2).sum(-1)
                      + (F.normalize(mv[masked], dim=-1) - F.normalize(vt[masked], dim=-1)).pow(2).sum(-1)).mean()

    # L_rep: repulsion of P's prediction from ideas already stated earlier in the same document
    # (anti-repetition, method 1 of 26/09). cos over flattened normalised slots; hinge above a margin.
    L_rep = torch.zeros((), device=dev)
    if lc.get("w_rep", 0.0) > 0:
        ded_m = batch["is_deduction"] if "is_deduction" in batch else torch.ones(NI, dtype=torch.bool, device=dev)
        earlier = ((batch["idea_doc"][:, None] == batch["idea_doc"][None, :])
                   & (batch["idea_pos"][None, :] < batch["idea_pos"][:, None]) & ded_m[None, :])
        pf = F.normalize(F.normalize(pv, dim=-1).reshape(NI, -1), dim=-1)
        tf = F.normalize(F.normalize(vt, dim=-1).reshape(NI, -1), dim=-1)
        cos = pf @ tf.T                                             # [NI (pred i), NI (past j)]
        viol = F.relu(cos - lc.get("rep_margin", 0.8)) * earlier
        row = viol.max(1).values
        L_rep = row[ded_m].mean() if ded_m.any() else row.mean()

    if force_path == "recon":
        gen = torch.zeros(NI, dtype=torch.bool, device=dev)
    elif force_path == "gen":
        gen = torch.ones(NI, dtype=torch.bool, device=dev)
    else:
        gen = torch.rand(NI, device=dev, generator=gen_rng) < p_gen
    if lc.get("gen_detach_P", False):  # A7b: P trained by L_jepa only (no token gradient)
        pk_d, pv_d = pk.detach(), pv.detach()
    else:
        pk_d, pv_d = pk, pv
    k_rec, v_rec = k, v
    if train and lc.get("cross_encode", False) and "enc_tok" in batch:
        # stage A: the reconstruction Z of the current idea comes from a RANDOM surface of its fact; P's inputs
        # (past ideas) stay E(main row), as in generation; JEPA target stays E_ema(main row)
        k_rec, v_rec = model.E(batch["enc_tok"], batch["enc_len"])
    K0 = torch.where(gen[:, None, None], pk_d, k_rec)
    V0 = torch.where(gen[:, None, None], pv_d, v_rec)
    if model.cfg.get("z_norm", "none") == "idea":
        # rescale each idea's values to unit RMS slot norm (keeps relative slot norms): the decoder can no
        # longer tell the generation path (P) from the reconstruction path (E) by the scale of Z
        V0 = V0 / V0.pow(2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()[:, None, None]

    if train and lc.get("dec_noise", 0.0) > 0:
        # loss.dec_noise = sigma (LDLM recipe, notes §35.6): Gaussian noise on the slot values the decoder reads (both
        # paths), sigma in units of the batch standard deviation of those values (scalar, no gradient)
        V0 = V0 + lc["dec_noise"] * V0.detach().float().std().to(V0.dtype) * torch.randn_like(V0)
    logits, tr = model.decode(batch, K0, V0)
    M, length, win = batch["M"], batch["idea_len"], batch["win"]
    ce, m = per_idea_ce(logits, win, M, length)
    ntok = m.sum()
    if "idea_w" in batch:  # DAgger: no token loss on the model's own (context-only) deductions
        mw = m * batch["idea_w"][:, None]
        L_tok = (ce * mw).sum() / mw.sum().clamp_min(1.0)
    else:
        L_tok = (ce * m).sum() / ntok
    ce_idea = (ce * m).sum(1) / length.float()
    multi = None
    if lc.get("multi_form", False) and "alt_win" in batch:
        multi = _multi_form_tok(model, batch, K0, V0, ce, m, length)
        L_tok = multi["L_tok"]
    rho_end = tr["rho"][torch.arange(NI, device=dev), length]
    if lc.get("exh_hinge", False):  # only penalise residual above tau (ratio -> 0 would reward storing less)
        L_exh = F.relu(rho_end - lc.get("tau", 0.1)).pow(2).mean()
    else:
        L_exh = rho_end.pow(2).mean()

    L_contrib = torch.zeros((), device=dev)
    contrib_gap = None
    if train and lc.get("w_contrib", 0) > 0 and torch.rand(()).item() < lc.get("p_contrib", 0.15):
        sel = torch.randperm(NI, device=dev)[: max(1, int(NI * lc.get("contrib_frac", 0.25)))]
        lg0, _ = model.decode(batch, K0[sel], torch.zeros_like(V0[sel]), idx=sel)
        ce0, m0 = per_idea_ce(lg0, win[sel], M, length[sel])
        ce0_idea = (ce0 * m0).sum(1) / length[sel].float()
        if lc.get("contrib_detach_noz", True):
            ce0_idea = ce0_idea.detach()
        gap = ce0_idea - ce_idea[sel]
        hinge = F.relu(lc.get("contrib_margin", 1.0) - gap)
        L_contrib = (hinge * wj[sel]).sum() / wj[sel].sum() if wj is not None else hinge.mean()
        contrib_gap = gap.mean().item()

    # scale-invariant sparsity: L1/L2 ratio of slot norms, in [1, sqrt(r)] (1 = a single active slot)
    sn = v.norm(dim=-1)
    L_sparse = (sn.sum(-1) / sn.pow(2).sum(-1).sqrt().clamp_min(1e-6)).mean()
    ds = v.shape[-1]
    vpool = v.reshape(-1, ds) * (ds ** 0.5)
    var_l, cov_l = vicreg(vpool)
    L_vic = lc.get("vic_var", 1.0) * var_l + lc.get("vic_cov", 0.04) * cov_l

    L_stop = torch.zeros((), device=dev)
    if "stop_logit" in tr:
        Ls = tr["stop_logit"].shape[1]
        steps = torch.arange(Ls, device=dev)[None]
        tgt_stop = (steps == length[:, None]).float()
        msk_stop = (steps <= length[:, None]).float()
        bce = F.binary_cross_entropy_with_logits(tr["stop_logit"], tgt_stop, reduction="none")
        L_stop = (bce * msk_stop).sum() / msk_stop.sum()

    # remove constant per-slot offsets (they inflate ||Z_0|| and make rho measure the offset, not content)
    L_center = v.mean(0).pow(2).sum(-1).mean()
    tokm_f = m
    L_gate = (tr["g"] * tokm_f).sum() / ntok  # optional "reading cost" (0 in the spec)

    loss = (L_tok + lc.get("w_stop", 1.0) * L_stop + lc.get("w_center", 0.0) * L_center + lc.get("w_gate", 0.0) * L_gate + lc.get("w_jepa", 1.0) * L_jepa + lc.get("w_vic", 1.0) * L_vic
            + lc.get("w_exh", 1.0) * L_exh + lc.get("w_contrib", 0.0) * L_contrib
            + lc.get("w_sparse", 0.0) * L_sparse
            + lc.get("w_para", 0.0) * L_para + lc.get("w_rep", 0.0) * L_rep + lc.get("w_mask", 0.0) * L_mask
            + lc.get("w_score", 0.1) * L_score + lc.get("w_flow", 0.0) * L_flow
            + lc.get("w_nce", 0.0) * L_nce + lc.get("w_transcr", 0.0) * L_transcr
            + lc.get("w_memsep", 0.0) * L_memsep + lc.get("w_qoracle", 0.0) * L_qoracle
            + lc.get("w_rel", 1.0) * L_rel + lc.get("w_plan", 1.0) * L_plan)

    out = {"loss": loss, "L_tok": L_tok.detach(), "L_jepa": L_jepa.detach(), "L_vic": L_vic.detach(),
           "L_exh": L_exh.detach(), "L_contrib": L_contrib.detach(), "L_sparse": L_sparse.detach(),
           "L_center": L_center.detach(), "L_gate": L_gate.detach(),
           "L_stop": L_stop.detach(), "L_para": L_para.detach(), "L_rep": L_rep.detach(), "L_mask": L_mask.detach(),
           "L_score": L_score.detach(), "L_flow": L_flow.detach(), "L_nce": L_nce.detach(),
           "L_transcr": L_transcr.detach(), "L_memsep": L_memsep.detach(),
           "L_qoracle": L_qoracle.detach(), "L_rel": L_rel.detach(), "L_plan": L_plan.detach()}
    if multi is not None:
        out["ce_min_role"] = multi["ce_min"]; out["ce_main"] = multi["ce_main"]
    if not train and "ref_win" in batch:
        out["ce_ref"] = _ref_ce(model, batch, K0, V0)
    with torch.no_grad():
        tokm = m.bool()
        if gen.any():
            out["ce_gen"] = (ce * m)[gen].sum() / m[gen].sum()
        if (~gen).any():
            out["ce_recon"] = (ce * m)[~gen].sum() / m[~gen].sum()
        stats = {
            "rho_end": rho_end.mean().item(),
            "frac_exhausted": (rho_end < lc.get("tau", 0.1)).float().mean().item(),
            "g_mean": tr["g"][tokm].mean().item(),
            "a_max_mean": tr["a"].max(-1).values[tokm].mean().item(),
            "v_std": (v.reshape(-1, ds)).std(0).mean().item(),
            "slot_norm_mean": v.norm(dim=-1).mean().item(),
            "active_slots": (v.norm(dim=-1) > 0.1 * v.norm(dim=-1).max(-1, keepdim=True).values).float().sum(-1).mean().item(),
            "z0_norm_mean": tr["n0"].mean().item(),
            "z0_norm_std": tr["n0"].std().item(),
            "key_gram_dev": (k[0] @ k[0].T - torch.eye(k.shape[1], device=dev)).norm().item(),
            "key_var": k.var(0).sum(-1).mean().item(),
            "frob_rho_end": (frob_norm(K0, tr["V_end"]) / frob_norm(K0, V0).clamp_min(1e-6)).mean().item(),
        }
        if contrib_gap is not None:
            stats["contrib_gap"] = contrib_gap
        if transcr_stats is not None:
            stats.update(transcr_stats)
        if memsep_stats is not None:
            stats.update(memsep_stats)
        if qoracle_stats is not None:
            stats.update(qoracle_stats)
        if rel_stats is not None:
            stats.update(rel_stats)
        if plan_stats is not None:
            stats.update(plan_stats)
        if khyp:
            stats["khyp_winner_entropy"] = torch.distributions.Categorical(
                probs=torch.bincount(winner, minlength=model.P.K).float() / len(winner)).entropy().item()
            stats["khyp_score_acc"] = (scores.argmax(-1) == winner).float().mean().item()
        if model.Fh is not None and not train:
            stats.update(flow_eval_stats(model, batch, lc, kt, vt))
    return out, {"stats": stats, "trace": tr, "gen": gen, "logits": logits, "k": k, "v": v,
                 "pk": pk, "pv": pv}



def _deduction_rows(batch):
    rows = batch["is_deduction"] if "is_deduction" in batch else torch.ones(batch["idea_len"].shape[0], dtype=torch.bool, device=batch["idea_len"].device)
    if "idea_w" in batch:  # DAgger docs: the model's own prefix sentences are context only (may be false)
        rows = rows & (batch["idea_w"] > 0)
    return rows


def _flow_loss(model, batch, lc, kt, vt, pk, pv):
    """Rectified-flow loss of the flow head on deduction ideas: target = E_ema slots [k ; v/|v|], context =
    P's slots (detached by default: the flow learns to sample around P's prediction without moving P)."""
    from .flow_head import to_target
    rows = _deduction_rows(batch)
    if not rows.any():
        return torch.zeros((), device=kt.device)
    c = model.flow_context(pk, pv)
    if lc.get("flow_detach_ctx", True):
        c = c.detach()
    if hasattr(model, "e_slot_norm") and model.training:
        with torch.no_grad():
            cur = vt[rows].norm(dim=-1).mean(0)
            model.e_slot_norm.mul_(0.99).add_(0.01 * cur) if float(model.e_slot_norm.sum()) > 0 else model.e_slot_norm.copy_(cur)
    z1 = model.flow_target(kt, vt)
    if lc.get("flow_target", "reference") == "candidates" and "cands" in batch:
        # upper-bound signal (v2 only): the target is a random VALID next deduction in a random paraphrase,
        # encoded by E_ema on the fly, instead of the single reference sentence.
        z1 = _candidate_targets(model, batch, lc, z1)
    return model.Fh.loss(z1[rows].detach(), c[rows])


_CAND_W2I = {}
_CAND_PARA = {}          # fact -> list of token lists (all connectors x forms), computed once
CAND_STATS = {"fallback": 0, "used": 0}


def _candidate_targets(model, batch, lc, z1):
    import random
    from .flow_head import to_target
    from ..data import reasoning as R
    from ..utils.common import ROOT
    meta = lc.get("flow_cand_meta", "data/reasoning/meta.pt")
    if meta not in _CAND_W2I:
        vocab = torch.load(ROOT / meta, weights_only=False)["vocab"]
        _CAND_W2I[meta] = {w: i for i, w in enumerate(vocab)}
    w2i = _CAND_W2I[meta]
    rng = random.Random(int(torch.initial_seed() % (2 ** 31)) + int(torch.randint(0, 2 ** 30, (1,))))
    idx, toks = [], []
    for i, cands in enumerate(batch["cands"]):
        if not cands: continue
        f = rng.choice(cands)
        if f not in _CAND_PARA:
            paras = []
            try:
                for c in range(len(R.CONNECT)):
                    for form in (0, 1):
                        s_ = R.S(); s_.add(R.CONNECT[c]); cl, _ = R.render_clause(f, rng, form); s_.ext(cl); s_.add(".")
                        t = [w2i[x] for x in s_.w]
                        if t not in paras: paras.append(t)
            except Exception:
                paras = []
            _CAND_PARA[f] = paras
        paras = _CAND_PARA[f]
        if not paras:
            CAND_STATS["fallback"] += 1; continue
        toks.append(rng.choice(paras)); idx.append(i); CAND_STATS["used"] += 1
    if not idx:
        return z1
    L = max(len(t) for t in toks)
    tok = torch.full((len(toks), L), 0, dtype=torch.long, device=z1.device)
    for r_, t in enumerate(toks): tok[r_, :len(t)] = torch.tensor(t, device=z1.device)
    with torch.no_grad():
        k2, v2 = model.E_ema(tok, torch.tensor([len(t) for t in toks], device=z1.device))
    z1 = z1.clone(); z1[torch.tensor(idx, device=z1.device)] = model.flow_target(k2, v2).to(z1.dtype)
    return z1


def flow_only_step(model, batch, lc):
    """Phase 2a: E, E_ema, P and the decoder are frozen; only the flow head is trained (fast path: no
    decoder pass, no other loss)."""
    with torch.no_grad():
        k, v = model.encode(batch)
        kt, vt = model.E_ema(batch["idea_tok"], batch["idea_len"])
        pk, pv = model.predict(k, v, batch)
    L_flow = _flow_loss(model, batch, {**lc, "flow_detach_ctx": True}, kt, vt, pk, pv)
    zero = torch.zeros((), device=k.device)
    return {"loss": L_flow, "L_tok": zero, "L_flow": L_flow.detach()}, {"stats": {"flow_cand_used": CAND_STATS["used"], "flow_cand_fallback": CAND_STATS["fallback"]}}


@torch.no_grad()
def flow_eval_stats(model, batch, lc, kt, vt):
    """Token CE on deductions when the decoder reads the flow's sample instead of P's prediction, and the
    cosine of each to the true idea (flow space)."""
    from .flow_head import to_target
    rows = _deduction_rows(batch)
    if not rows.any():
        return {}
    k, v = model.encode(batch)
    pk, pv = model.predict(k, v, batch)
    kf, vf = model.predict_flow(k, v, batch, n_steps=lc.get("flow_eval_steps", 8))
    out, z1 = {}, to_target(kt, vt)
    for name, (K_, V_) in (("P", (pk, pv)), ("flow", (kf, vf))):
        V0 = V_
        if model.cfg.get("z_norm", "none") == "idea":
            V0 = V0 / V0.pow(2).sum(-1).mean(-1).clamp_min(1e-8).sqrt()[:, None, None]
        lg, _ = model.decode(batch, K_, V0)
        ce, m = per_idea_ce(lg, batch["win"], batch["M"], batch["idea_len"])
        out[f"ce_ded_{name}"] = ((ce * m)[rows].sum() / m[rows].sum()).item()
        if "b_eff" in batch:
            # CE on the REFERENCE sentence: fair only where the next step is unique (n = 1). At n >= 2 a sampler
            # that commits to another valid deduction is penalised, a regression that hedges is not.
            for tag, sel in (("n1", rows & (batch["b_eff"] == 1)), ("n2+", rows & (batch["b_eff"] >= 2))):
                if sel.any():
                    out[f"ce_ded_{name}_{tag}"] = ((ce * m)[sel].sum() / m[sel].sum()).item()
        out[f"cos_ded_{name}"] = F.cosine_similarity(to_target(K_, V_)[rows].flatten(1),
                                                     z1[rows].flatten(1), dim=-1).mean().item()
    return out



def _scatter_logsumexp(vals, idx, n):
    mx = torch.full((n,), float("-inf"), device=vals.device).scatter_reduce(0, idx, vals, "amax", include_self=True)
    ex = torch.zeros(n, device=vals.device).scatter_add(0, idx, torch.exp(vals - mx[idx]))
    return mx + torch.log(ex)


def _decode_rows(model, win, length, M, K, V, rel=None):
    """Decoder pass on explicit windows (alt / reference rows), same Z handling as the main rows."""
    e = model.O(win, M)
    return model.dec(win, length, M, K, V, e, **model.dec_rel(rel))


def _multi_form_tok(model, batch, K0, V0, ce, m, length):
    """Stage A: L_tok = -log sum_p exp(-CE_p) over the forms of each idea's role (the main row + its alt rows),
    CE_p = SUM of token CEs of form p; averaged over ideas after dividing by the main row's length (per-token
    scale of the old tables). DAgger prefix ideas (idea_w = 0) carry no loss."""
    NI = K0.shape[0]
    M = batch["M"]
    ce_main = (ce * m).sum(1)                                              # [NI] summed CE of the main row
    ai = batch["alt_idea"]
    rel = batch["rel_id"][ai] if "rel_id" in batch else None
    lg, _ = _decode_rows(model, batch["alt_win"], batch["alt_len"], M, K0[ai], V0[ai], rel=rel)
    ce_a, m_a = per_idea_ce(lg, batch["alt_win"], M, batch["alt_len"])
    ce_alt = (ce_a * m_a).sum(1)                                           # [A]
    lse = _scatter_logsumexp(torch.cat([-ce_main, -ce_alt]), torch.cat([torch.arange(NI, device=ai.device), ai]), NI)
    per = -lse / length.float()
    w = batch["idea_w"].float() if "idea_w" in batch else torch.ones_like(per)
    L_tok = (per * w).sum() / w.sum().clamp_min(1e-6)
    with torch.no_grad():
        best = torch.full((NI,), float("inf"), device=ai.device).scatter_reduce(
            0, ai, ce_alt / batch["alt_len"].float(), "amin", include_self=True)
        ce_min = torch.minimum(ce_main / length.float(), best)
        ce_min = (ce_min * w).sum() / w.sum().clamp_min(1e-6)
        ce_m = ((ce_main / length.float()) * w).sum() / w.sum().clamp_min(1e-6)
    return {"L_tok": L_tok, "ce_min": ce_min, "ce_main": ce_m}


@torch.no_grad()
def _ref_ce(model, batch, K0, V0):
    """Per-token CE of the ORIGINAL sentence (original stream's prefix), same Z: comparable to old tables only
    loosely (the decoder is trained on the role forms, the reference is often "then"/"this means that")."""
    M = batch["M"]
    lg, _ = _decode_rows(model, batch["ref_win"], batch["ref_len"], M, K0, V0, rel=batch.get("rel_id"))
    ce, m = per_idea_ce(lg, batch["ref_win"], M, batch["ref_len"])
    if "idea_w" in batch:
        m = m * batch["idea_w"][:, None]
    return (ce * m).sum() / m.sum().clamp_min(1.0)



def nce_embed(k, v):
    """[k ; v/|v|] over the slots, flattened and L2-normalised (the stage-B contrastive space)."""
    return F.normalize(torch.cat([k, F.normalize(v, dim=-1)], -1).flatten(1).float(), dim=-1)


def _info_nce(model, batch, k, v, temp, nce_neg="batch"):
    ia = batch["pos_idea"]
    za = nce_embed(k[ia], v[ia])                                               # anchors: E(main rows)
    kp, vp = model.E(batch["pos_tok"], batch["pos_len"])
    zp = nce_embed(kp, vp)                                                     # positives: E(other surface)
    P = za.shape[0]
    fk, pk = batch["fact_key"][ia], batch["pos_key"][ia]
    eye = torch.eye(P, dtype=torch.bool, device=za.device)

    def masked_pairs(keys):
        same = (keys[:, None, :, None] == keys[None, :, None, :]) & (keys[:, None, :, None] >= 0)
        return same.flatten(2).any(-1) & ~eye                                  # [P, P]
    false_neg = masked_pairs(torch.stack([fk, pk], -1))                        # [P, 2] equivalent facts per pair
    if nce_neg == "other_docs":   # label-free: negatives only from the OTHER stories of the batch (story identity is known)
        doc = batch["idea_doc"][ia]
        false_neg = false_neg | ((doc[:, None] == doc[None, :]) & ~eye)
    # "since A , B" positives (data.since_surfaces) also contain A: never a negative of A's rows either. Only the
    # comparisons that involve a positive use this third key; anchor-anchor pairs keep the (fact, pos) keys.
    fn_pos = masked_pairs(torch.stack([fk, pk, batch["pos_key2"][ia]], -1)) if "pos_key2" in batch else false_neg
    if nce_neg == "other_docs":
        fn_pos = fn_pos | false_neg

    def one_side(q, pos, other, m_pos, m_other):
        logits = torch.cat([q @ pos.T, q @ other.T], 1) / temp                 # [P, 2P]: positives then the others
        logits = logits.masked_fill(torch.cat([m_pos, m_other | eye], 1), float("-inf"))
        return F.cross_entropy(logits, torch.arange(P, device=q.device))
    return 0.5 * (one_side(za, zp, za, fn_pos, false_neg) + one_side(zp, za, zp, fn_pos, fn_pos))


@torch.no_grad()
def greedy_decode_rows(model, win, M, K, V, max_len=24, rel=None):
    """Batched greedy decoding of ideas (K, V) [N, r, *] after the prefix win[:, :M]; the first token is forced to the
    main row's first token (the connector "so" for a deduction), as in generation. Stops at the stop head (if any) or
    after the end token (the last token of the main rows). Decoder in eval mode (no word dropout).
    Returns (tokens [N, max_len] PAD-filled, lengths [N])."""
    N, dev = win.shape[0], win.device
    first = win[:, M]
    lens_ref = (win[:, M:] != 0).sum(1)
    end_id = int(torch.mode(win[torch.arange(N, device=dev), M + lens_ref - 1]).values)
    was = model.dec.training
    model.dec.eval()
    toks = torch.zeros(N, max_len, dtype=torch.long, device=dev)
    toks[:, 0] = first
    length = torch.ones(N, dtype=torch.long, device=dev)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    try:
        for s_ in range(1, max_len):
            w_ = torch.cat([win[:, :M], toks[:, :s_], torch.zeros(N, 1, dtype=torch.long, device=dev)], 1)
            e = model.O(w_, M)
            logits, tr = model.dec(w_, torch.full((N,), s_ + 1, device=dev), M, K, V, e, **model.dec_rel(rel))
            stop = toks[:, s_ - 1] == end_id
            if "stop_logit" in tr:
                stop = stop | (tr["stop_logit"][:, s_] > 0)
            done = done | stop
            if bool(done.all()):
                break
            lg = logits[:, s_].float()
            lg[:, :3] = float("-inf")                                    # PAD, BOS, EOS
            nxt = lg.argmax(-1)
            toks[:, s_] = torch.where(done, torch.zeros_like(nxt), nxt)
            length = length + (~done).long()
    finally:
        model.dec.train(was)
    return toks, length


def _transcr_loss(model, batch, lc, k, v, pk, pv):
    """Stage C: L_transcr = || norm(Ẑ) - sg(norm(E_ema(D_greedy(Ẑ)))) ||^2 on a random fraction of the deduction rows.
    transcr_detach_E (default true): P re-run on detached E slots, so the gradient reaches P only (spec)."""
    dev = k.device
    NI = k.shape[0]
    ded = batch["is_deduction"].bool() if "is_deduction" in batch else torch.ones(NI, dtype=torch.bool, device=dev)
    if "idea_w" in batch:
        ded = ded & (batch["idea_w"] > 0)
    sel = ded.nonzero()[:, 0]
    frac = lc.get("transcr_frac", 0.25)
    if len(sel) == 0:
        return torch.zeros((), device=dev), None
    sel = sel[torch.randperm(len(sel), device=dev)[:max(1, int(round(len(sel) * frac)))]]
    if lc.get("transcr_detach_E", True):
        pk, pv = model.predict(k.detach(), v.detach(), batch)
    pk_s, pv_s = pk[sel], pv[sel]
    toks, lens = greedy_decode_rows(model, batch["win"][sel], batch["M"], pk_s.detach(), pv_s.detach(),
                                    lc.get("transcr_max_len", 24),
                                    **({"rel": batch["rel_id"][sel]} if "rel_id" in batch else {}))
    with torch.no_grad():
        kt, vt = model.E_ema(toks, lens)
    d = ((pk_s - kt).pow(2).sum(-1) + (F.normalize(pv_s, dim=-1) - F.normalize(vt, dim=-1)).pow(2).sum(-1)).mean(-1)
    guard_rate = None
    if lc.get("transcr_guard", "none") == "oracle":
        # oracle guard (ceiling): pull only the rows whose decoded sentence states a valid next deduction (batch["cands"])
        from ..data import verifier as V
        vocab = model.vocab
        keep = torch.tensor([V.parse_sentence(" ".join(vocab[t] for t in toks[r, :int(lens[r])].tolist()))[0]
                             in set(batch["cands"][int(i)]) for r, i in enumerate(sel.tolist())], device=dev)
        guard_rate = keep.float().mean().item()
        d = d * keep.float()
        if not keep.any():
            return d.sum(), {"transcr_guard_kept": 0.0, "transcr_rows": float(len(sel))}
        d = d[keep]
    with torch.no_grad():   # share of decoded sentences identical to the main row (the reference fact, main form)
        W = toks.shape[1]
        ref = batch["win"][sel, batch["M"]:batch["M"] + W]
        ref = torch.cat([ref, torch.zeros(len(sel), W - ref.shape[1], dtype=torch.long, device=dev)], 1)
        same = ((toks == ref) | (torch.arange(W, device=dev)[None] >= lens[:, None])).all(1) & (lens == batch["idea_len"][sel])
        stats = {"transcr_same_as_ref": same.float().mean().item(), "transcr_len": lens.float().mean().item(),
                 "transcr_rows": float(len(sel))}
        if guard_rate is not None:
            stats["transcr_guard_kept"] = guard_rate
    return d.mean(), stats


@torch.no_grad()
def transcr_eval_stats(model, batch, vocab, max_rows=256):
    """Eval only (stage C): greedy decoding of P's prediction on the deduction rows, verdict against the exact set of
    valid next deductions (batch["cands"]): share valid, share valid but != the reference fact (main row), share of the
    reference fact. Tells whether L_transcr pulls P towards wrong facts (self-confirmation)."""
    from ..data import verifier as V
    if "cands" not in batch or "is_deduction" not in batch:
        return {}
    k, v = model.encode(batch)
    pk, pv = model.predict(k, v, batch)
    sel = batch["is_deduction"].nonzero()[:, 0][:max_rows]
    if len(sel) == 0:
        return {}
    toks, lens = greedy_decode_rows(model, batch["win"][sel], batch["M"], pk[sel], pv[sel],
                                    **({"rel": batch["rel_id"][sel]} if "rel_id" in batch else {}))
    M = batch["M"]
    n_valid = n_other = n_ref = 0
    for r, i in enumerate(sel.tolist()):
        f = V.parse_sentence(" ".join(vocab[t] for t in toks[r, :int(lens[r])].tolist()))[0]
        ref = V.parse_sentence(" ".join(vocab[t] for t in batch["win"][i, M:M + int(batch["idea_len"][i])].tolist()))[0]
        ok = f is not None and f in set(batch["cands"][i])
        n_valid += ok; n_ref += f is not None and f == ref; n_other += ok and f != ref
    n = len(sel)
    return {"transcr_eval_valid": n_valid / n, "transcr_eval_valid_other": n_other / n, "transcr_eval_ref": n_ref / n}


def _memsep_loss(rows, batch, lc):
    """F1 (§30): separate what P's memory WRITES for same-template facts about different entities (the negatives of
    stage B: same predicate and role, same story, different fact tuple). Per memory layer, centred cosine (batch mean
    of the written rows removed) between such pairs; L = mean over pairs and layers of relu(cos - tau)^2.
    memsep_on: "values" (default) | "keys" | "both"."""
    valid = rows["valid"]
    tk, fk, doc = batch["tmpl_key"], batch["fact_key"], batch["idea_doc"]
    pair = ((doc[:, None] == doc[None, :]) & (tk[:, None] == tk[None, :]) & (tk[:, None] >= 0)
            & (fk[:, None] != fk[None, :]) & valid[:, None] & valid[None, :])
    pair = torch.triu(pair, diagonal=1)
    if not pair.any():
        return torch.zeros((), device=valid.device), {"memsep_pairs": 0.0}
    tau = lc.get("memsep_tau", 0.2)
    parts = {"values": ["v"], "keys": ["k"], "both": ["k", "v"]}[lc.get("memsep_on", "values")]
    losses, cos_means = [], []
    for part in parts:
        x = rows[part].float()                                              # [NI, L, D]
        x = x - x[valid].mean(0, keepdim=True)
        x = F.normalize(x, dim=-1)
        for li in range(x.shape[1]):
            cp = (x[:, li] @ x[:, li].T)[pair]
            losses.append(F.relu(cp - tau).pow(2).mean())
            cos_means.append(cp.mean().detach())
    return torch.stack(losses).mean(), {"memsep_pairs": float(pair.sum()),
                                        "memsep_cos": float(torch.stack(cos_means).mean())}


def _qoracle_loss(P, batch, lc):
    """Chantier 3 bis §5, oracle test of the query (a CEILING, strong supervision as in Memory Networks): when P predicts
    an F1 deduction "obj ... L", pull the query of its memory towards the key written for the premise that states obj's
    position (the useful antecedent, known to the generator; batch["qo_ptr"] = its position in the story, -1 = no target).
    Score of a past idea p for the query block n (which predicts idea n) = mean over (query slot, key slot, head) of
    cos(q, k) with k from block p + 1 (which holds idea p), as in scripts/diag_query_keys.py; per memory layer, softmax
    cross-entropy over the ideas before n at temperature loss.qoracle_temp (0.1). L = mean over targets and layers."""
    di, ptr = batch["doc_index"], batch["qo_ptr"]
    B, Nm = di.shape
    tgt = torch.full((B, Nm), -1, dtype=torch.long, device=di.device)
    ok = di >= 0
    tgt[ok] = ptr[di[ok]]
    sel = tgt >= 0
    if not sel.any():
        return torch.zeros((), device=di.device), {"qoracle_n": 0.0}
    r = P.r
    n_idx = torch.arange(Nm, device=di.device)
    cand = (n_idx[None, :] >= 1) & (n_idx[None, :] <= n_idx[:, None])        # [n, j]: key block j holds idea j - 1 < n
    losses, accs = [], []
    for m in P.mems:
        q, k = m.kept_qk                                                     # [B, H, T, dh], T = Nm * r
        H, dh = q.shape[1], q.shape[-1]
        qb, kb = q.reshape(B, H, Nm, r, dh), k.reshape(B, H, Nm, r, dh)
        sim = torch.einsum("bhid,bhjd->bij", qb.mean(3), kb.mean(3)) / H   # = mean over slot pairs and heads
        logits = (sim / lc.get("qoracle_temp", 0.1)).masked_fill(~cand[None], float("-inf"))
        lg, t = logits[sel], tgt[sel] + 1
        losses.append(F.cross_entropy(lg, t))
        accs.append((lg.argmax(-1) == t).float().mean().detach())
        m.kept_qk = None
    return torch.stack(losses).mean(), {"qoracle_n": float(sel.sum()), "qoracle_acc": float(torch.stack(accs).mean())}


def _rel_loss(model, batch):
    """Step 2 (chantier 3 bis §4): cross-entropy of P's relation head (none / justified_by / sequence) and pointer
    head (relative offset to the justifying idea, 0 = none) on the deductions (DAgger context rows excluded)."""
    lr, lp = model.rel_logits()
    NI = lr.shape[0]
    rows = batch["is_deduction"].bool() if "is_deduction" in batch else torch.ones(NI, dtype=torch.bool, device=lr.device)
    if "idea_w" in batch:
        rows = rows & (batch["idea_w"] > 0)
    if not rows.any():
        return torch.zeros((), device=lr.device), None
    tr, tp = batch["rel_id"][rows].long(), batch["rel_off"][rows].long().clamp(0, lp.shape[1] - 1)
    L = F.cross_entropy(lr[rows], tr) + F.cross_entropy(lp[rows], tp)
    with torch.no_grad():
        acc_r = (lr[rows].argmax(-1) == tr).float().mean()
        acc_p = (lp[rows].argmax(-1) == tp).float().mean()
        jb = tr == 1
        acc_jb = ((lr[rows].argmax(-1) == tr) & (lp[rows].argmax(-1) == tp))[jb].float().mean() if jb.any() else acc_r * 0
    return L, {"rel_acc": float(acc_r), "rel_ptr_acc": float(acc_p), "rel_jb_acc": float(acc_jb),
               "rel_jb_frac": float(jb.float().mean())}


@torch.no_grad()
def plan_targets(vt, doc_index):
    """Plan variable (notes §32.20): per idea row, o = argmax over the past ideas j < n of the story of
    cos(Ē_ema(idea n), Ē_ema(idea j)), Ē = values per-slot normalised, flattened, L2 (diag_plan_var space); -1 for the
    first idea. Returns [NI] story indices."""
    z = F.normalize(F.normalize(vt.float(), dim=-1).flatten(1), dim=-1)
    B, Nm = doc_index.shape
    zb = z[doc_index.clamp_min(0)]                                          # [B, Nm, D]
    sim = zb @ zb.transpose(1, 2)
    n_idx = torch.arange(Nm, device=z.device)
    ok = (n_idx[None, :] < n_idx[:, None])[None] & (doc_index >= 0)[:, None, :]
    o = sim.masked_fill(~ok, float("-inf")).argmax(-1).masked_fill(~ok.any(-1), -1)
    out = torch.full((z.shape[0],), -1, dtype=torch.long, device=z.device)
    out[doc_index[doc_index >= 0]] = o[doc_index >= 0]
    return out


def _plan_loss(model, batch):
    """Cross-entropy of the plan pointer head (computed WITHOUT the plan) to o, on the deduction rows."""
    lg, di = model.plan_logits, batch["doc_index"]
    B, Nm = di.shape
    tgt = torch.full((B, Nm), -1, dtype=torch.long, device=di.device)
    tgt[di >= 0] = batch["plan_o"][di[di >= 0]]
    rows = tgt >= 0
    if "is_deduction" in batch:
        ded = torch.zeros((B, Nm), dtype=torch.bool, device=di.device)
        ded[di >= 0] = batch["is_deduction"].bool()[di[di >= 0]]
        rows = rows & ded
    if "idea_w" in batch:
        wr = torch.ones((B, Nm), device=di.device)
        wr[di >= 0] = batch["idea_w"].float()[di[di >= 0]]
        rows = rows & (wr > 0)
    if not rows.any():
        return torch.zeros((), device=di.device), None
    L = F.cross_entropy(lg[rows], tgt[rows])
    with torch.no_grad():
        p = torch.softmax(lg[rows], -1)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()
        acc = (lg[rows].argmax(-1) == tgt[rows]).float().mean()
    return L, {"plan_acc": float(acc), "plan_entropy": float(ent)}
