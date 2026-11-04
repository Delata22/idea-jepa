"""Aggregate stage-1 diagnostics over seeds: tables (markdown) + figures.

python -m ideajepa.eval.stage1_report runs/stage1 --out reports/stage1_tables.md
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.common import ROOT

FIG = ROOT / "reports" / "figures"

# metric name -> path into diag.json
METRICS = {
    "CE gen (iid)": ("iid", "ce_gen"),
    "CE gen (comp)": ("comp", "ce_gen"),
    "CE recon (iid)": ("iid", "ce_recon"),
    "CE recon (comp)": ("comp", "ce_recon"),
    "bind 1st ent.": ("binding", "bind_first_entity_acc"),
    "bind pair": ("binding", "bind_pair_acc"),
    "probe agent": ("z0_probe", "probe_agent"),
    "probe patient": ("z0_probe", "probe_patient"),
    "swap": ("swap", "swap_success"),
    "passive: agent": ("z0_probe", "passive_probe_meaning_agent"),
    "passive: 1st NP": ("z0_probe", "passive_probe_surface_first_np"),
    "ρ_end recon": ("exhaustion_recon", "rho_end_mean"),
    "ρ<τ recon": ("exhaustion_recon", "frac_below_tau"),
    "ρ<τ short": ("exhaustion_recon", "frac_below_tau_short"),
    "ρ<τ long": ("exhaustion_recon", "frac_below_tau_long"),
    "premature (recon)": ("exhaustion_recon", "frac_premature_cross"),
    "ρ<τ gen": ("exhaustion_gen", "frac_below_tau"),
    "stop exact (recon)": ("stop_recon", "stop_exact"),
    "stop exact short": ("stop_recon", "stop_exact_short"),
    "stop exact long": ("stop_recon", "stop_exact_long"),
    "stop exact (gen)": ("stop_gen", "stop_exact"),
    "premature (gen)": ("exhaustion_gen", "frac_premature_cross"),
    "g function": ("gate_recon", "function"),
    "g content": ("gate_recon", "content"),
    "gate AUC c>f": ("gate_recon", "auc_content_vs_function"),
    "key probe": ("key_probe_role_acc",),
    "a_max content": ("a_max_content",),
    "patching": ("patching", "patch_success"),
    "para AUC": ("paraphrase", "auc_same_vs_diff"),
    "‖P‖/‖E‖": ("p_shrink", "norm_ratio_mean"),
    "held-out ent. (recon)": ("heldout_entity_acc_recon",),
}


def get(d, path):
    for p in path:
        if d is None or p not in d:
            return np.nan
        d = d[p]
    return np.nan if d is None else float(d)


def variant_of(name):
    return re.sub(r"_s\d+$", "", name)


def load_runs(root: Path):
    runs = defaultdict(list)
    for p in sorted(root.glob("*/diag.json")):
        d = json.loads(p.read_text())
        runs[variant_of(p.parent.name)].append((p.parent, d))
    return runs


def fmt(vals):
    vals = np.array([v for v in vals if not np.isnan(v)])
    if len(vals) == 0:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}"


def table(runs, metrics, order=None):
    names = order or sorted(runs)
    rows = []
    for v in names:
        if v not in runs:
            continue
        row = {"variant": f"{v} (n={len(runs[v])})"}
        for m in metrics:
            row[m] = fmt([get(d, METRICS[m]) for _, d in runs[v]])
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def fig_collapse(runs, variants):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for v in variants:
        for rd, _ in runs.get(v, [])[:1]:
            df = pd.read_csv(rd / "train.csv")
            if "v_std" not in df:
                continue
            axes[0].plot(df["step"], df["v_std"] * 16, label=v)
            val = pd.read_csv(rd / "val.csv")
            if "eff_rank_v" in val:
                axes[1].plot(val["step"], val["eff_rank_v"], marker="o", label=v)
    axes[0].set_title("per-dim std of slot values (×√d)")
    axes[1].set_title("effective rank of pooled slot values (val)")
    for a in axes:
        a.set_xlabel("step")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "stage1_collapse.png", dpi=130)
    plt.close(fig)


def fig_rho(runs, variant, tau=0.1):
    rd = runs[variant][0][0]
    z = np.load(rd / "diag_arrays.npz")
    lc = z["len_class"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, msk in (("short", lc <= 1), ("medium", (lc >= 2) & (lc <= 4)), ("long", lc >= 5)):
        axes[0].hist(z["rho_end_recon"][msk], bins=40, range=(0, 0.5), alpha=0.5, label=name, density=True)
        traj = z["rho_traj"][msk]
        lens = z["lens"][msk]
        # rho vs normalised position (0 = start, 1 = end of idea)
        grid = np.linspace(0, 1, 21)
        curves = [np.interp(grid, np.arange(n + 1) / n, t[:n + 1]) for t, n in zip(traj, lens)]
        axes[1].plot(grid, np.mean(curves, 0), label=f"{name} (mean len {lens.mean():.1f})")
    axes[0].axvline(tau, color="k", ls="--")
    axes[0].set_title(f"ρ at idea end (recon) — {variant}")
    axes[1].axhline(tau, color="k", ls="--")
    axes[1].set_title("mean ρ vs relative position in idea")
    for a in axes:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"stage1_rho_{variant}.png", dpi=130)
    plt.close(fig)


def fig_role_slot(runs, variant):
    d = runs[variant][0][1]
    M = np.array(d["role_slot_matrix"]["M"])
    roles = d["role_slot_matrix"]["roles"]
    abl = np.array(d["slot_ablation"]["drop"]).T
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, mat, t in ((axes[0], M, "mean read attention a_i (content words)"),
                       (axes[1], abl, "accuracy drop when v_i := 0")):
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(roles)))
        ax.set_yticklabels(roles)
        ax.set_xlabel("slot")
        ax.set_title(t, fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="w" if mat[i, j] < 0.5 else "k")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(FIG / f"stage1_roleslot_{variant}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_gate(runs, variants):
    types = ["function", "content", "modifier", "pronoun"]
    fig, ax = plt.subplots(figsize=(7, 3))
    w = 0.8 / max(1, len(variants))
    for i, v in enumerate(variants):
        if v not in runs:
            continue
        vals = np.array([[get(d, ("gate_recon", t)) for t in types] for _, d in runs[v]])
        ax.bar(np.arange(4) + i * w, vals.mean(0), w, yerr=vals.std(0) if len(vals) > 1 else None, label=v)
    ax.set_xticks(np.arange(4) + w * (len(variants) - 1) / 2)
    ax.set_xticklabels(types)
    ax.set_ylabel("mean gate g")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "stage1_gate.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=str(ROOT / "reports" / "stage1_tables.md"))
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(a.root))
    order = [v for v in ["v_full", "v_full_gc", "b0", "b1", "b1prime", "a1", "a2", "a3", "a4", "a5", "a6",
                         "a7", "a7b", "a8", "a9", "a10", "a11", "a12", "a6b", "a15", "v_spec", "v_full_deep"] if v in runs]
    order += [v for v in sorted(runs) if v not in order]
    parts = ["## Perplexité / généralisation\n", table(runs, ["CE gen (iid)", "CE gen (comp)", "CE recon (iid)",
                                                             "CE recon (comp)", "held-out ent. (recon)"], order),
             "\n\n## Liaison\n", table(runs, ["bind 1st ent.", "bind pair", "probe agent", "probe patient", "passive: agent",
                                                  "passive: 1st NP", "swap", "patching"], order),
             "\n\n## Épuisement\n", table(runs, ["ρ_end recon", "ρ<τ recon", "ρ<τ short", "ρ<τ long", "premature (recon)",
                                                  "ρ<τ gen", "premature (gen)", "stop exact (recon)",
                                                  "stop exact short", "stop exact long", "stop exact (gen)"], order),
             "\n\n## Porte et rôles\n", table(runs, ["g function", "g content", "gate AUC c>f", "key probe", "a_max content", "para AUC",
                                                  "‖P‖/‖E‖"], order)]
    # aphasia examples (V-full, first seed)
    for v in ("v_full", "a15"):
        if v in runs and runs[v][0][1].get("aphasia"):
            ap = runs[v][0][1]["aphasia"]
            parts.append(f"\n\n## Test « aphasie » ({v}, graine {runs[v][0][0].name[-1]})\n")
            for i in range(min(4, len(ap["reference"]))):
                parts.append(f"- référence : `{ap['reference'][i]}`\n  - normal : `{ap['normal'][i]}`\n"
                             f"  - g = 0 : `{ap['gate0'][i]}`\n  - h_local coupé : `{ap['no_local'][i]}`")
    FLOOR = 1.6026
    parts.append(f"\n\nPlancher d'entropie exact (test_iid) : {FLOOR:.4f} nat/token.")
    Path(a.out).write_text("\n".join(parts) + "\n")
    print("\n".join(parts))
    idea_variants = [v for v in order if v in runs and "role_slot_matrix" in runs[v][0][1]]
    fig_collapse(runs, [v for v in idea_variants if v in ("v_full", "v_full_gc", "a8", "a7b", "a1")])
    fig_gate(runs, [v for v in idea_variants if v in ("v_full", "v_full_gc", "a2", "a4", "a6")])
    for v in [x for x in ("v_full", "v_full_gc", "a1") if x in runs]:
        fig_rho(runs, v)
        fig_role_slot(runs, v)


if __name__ == "__main__":
    main()
