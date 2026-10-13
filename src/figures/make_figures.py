
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, rankdata

# ── Paths resolved from this file's location ─────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))          # src/figures/
REPO = os.path.dirname(os.path.dirname(HERE))              # repository root
DATA_DIR = os.path.join(REPO, "data", "processed")
DEFAULT_TAU_NPZ = os.path.join(DATA_DIR, "raw_kendall_tau_per_seed.npz")
NMAC_NPZ = os.path.join(DATA_DIR, "raw_nmac_per_seed.npz")
FIG_DIR = os.path.join(REPO, "figures")

N_SEEDS = 20
MASKING_LEVELS = [0.10, 0.30, 0.50, 0.70]
MASKING_PCT = np.array([int(r * 100) for r in MASKING_LEVELS])

DATASETS = ["PBMC3K", "Paul15"]
DATASET_LABEL = {"PBMC3K": "PBMC 3K", "Paul15": "Paul15"}
DATASET_FILES = {  # expression matrix, stored global mean
    "PBMC3K": ("X_2000hvg.npy", "global_mean_2000hvg.npy"),
    "Paul15": ("X_paul15.npy", "global_mean_paul15.npy"),
}

# Estimators shown in the manuscript figures
ORDER = ["GMCC", "Cosine", "Spearman", "Pearson"]

# Distinguishable in grayscale: unique marker + linestyle per estimator
STYLE = {
    "GMCC":     dict(marker="o", linestyle="-",  color="black"),
    "Cosine":   dict(marker="s", linestyle="--", color="0.35"),
    "Spearman": dict(marker="v", linestyle=":",  color="0.55"),
    "Pearson":  dict(marker="D", linestyle=(0, (3, 1, 1, 1)), color="0.70"),
}
XLABEL = "Masked detected entries (%)"

# ── Elsevier-friendly style ──────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "figure.dpi": 150,
})


# ════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════
def load_per_seed(npz_path, hint):
    """Load per-seed values. Returns {(dataset, method, rate): array}."""
    if not os.path.exists(npz_path):
        sys.exit(f"ERROR: {npz_path} not found.\n{hint}")
    data = np.load(npz_path)
    out = {}
    for key in data.files:
        ds, method, rate = key.split("|")
        out[(ds, method, float(rate))] = data[key]
    return out


def save_per_seed(values, npz_path):
    np.savez(npz_path, **{f"{ds}|{m}|{r}": v for (ds, m, r), v in values.items()})


# ════════════════════════════════════════════════════════════════════
# Estimators and masking (identical to paired_difference_ci.py)
# ════════════════════════════════════════════════════════════════════
def _gram(Xc):
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    Z = Xc / norms
    return np.clip(Z @ Z.T, -1.0, 1.0)

def gmcc(X, mu):  return _gram(X - mu)
def pearson(X):   return _gram(X - X.mean(axis=1, keepdims=True))
def cosine(X):    return _gram(X)
def spearman(X):  return pearson(np.apply_along_axis(rankdata, 1, X))

def apply_dropout(matrix, rate, seed):
    np.random.seed(seed)
    result = matrix.copy()
    result[(matrix > 0) & (np.random.rand(*matrix.shape) < rate)] = 0
    return result


# ════════════════════════════════════════════════════════════════════
# NMAC computation
# ════════════════════════════════════════════════════════════════════
def nmac(s_orig, s_mask):
    mad = np.median(np.abs(s_orig - np.median(s_orig)))
    return np.nan if mad < 1e-12 else np.median(np.abs(s_mask - s_orig)) / mad


def detect_mu_rule(ds, X, mu_stored, tau, idx):
    """Find the GMCC centering rule that reproduces the tau .npz (seed 1, 10%)."""
    Xm = apply_dropout(X, 0.10, seed=1)

    cos_tau = kendalltau(cosine(X)[idx], cosine(Xm)[idx])[0]
    ref_cos = tau[(ds, "Cosine", 0.1)][0]
    if abs(cos_tau - ref_cos) > 1e-6:
        sys.exit(f"ERROR [{ds}]: Cosine tau_b (seed 1) = {cos_tau:.6f}, "
                 f".npz = {ref_cos:.6f}. Masks differ from Figure 1; "
                 "check apply_dropout/seeds in paired_difference_ci.py.")

    s0 = gmcc(X, mu_stored)[idx]
    ref = tau[(ds, "GMCC", 0.1)][0]
    for rule in ("fixed", "recompute"):
        mu = mu_stored if rule == "fixed" else Xm.mean()
        t = kendalltau(s0, gmcc(Xm, mu)[idx])[0]
        print(f"    GMCC tau_b seed 1 (mu {rule:<9}) = {t:.6f}   .npz = {ref:.6f}")
        if abs(t - ref) < 1e-6:
            return rule
    sys.exit(f"ERROR [{ds}]: neither GMCC centering rule reproduces the .npz.")


def compute_nmac(tau):
    values, rules = {}, {}
    for ds in DATASETS:
        xf, muf = DATASET_FILES[ds]
        X = np.load(os.path.join(DATA_DIR, xf))
        mu_stored = float(np.load(os.path.join(DATA_DIR, muf))[0])
        idx = np.triu_indices(X.shape[0], k=1)
        print(f"\n{DATASET_LABEL[ds]}: {X.shape[0]} genes x {X.shape[1]} cells, "
              f"stored mu = {mu_stored:.4f}")
        print("  Consistency check against the tau .npz:")
        rules[ds] = detect_mu_rule(ds, X, mu_stored, tau, idx)
        print(f"  -> matched: mu {rules[ds]}. Computing NMAC...")

        funcs = {
            "GMCC": lambda M, masked, ds=ds: gmcc(
                M, M.mean() if (masked and rules[ds] == "recompute") else mu_stored),
            "Pearson":  lambda M, masked: pearson(M),
            "Cosine":   lambda M, masked: cosine(M),
            "Spearman": lambda M, masked: spearman(M),
        }
        s_orig = {m: f(X, False)[idx] for m, f in funcs.items()}
        for r in MASKING_LEVELS:
            t0 = time.time()
            per_seed = {m: [] for m in funcs}
            for seed in range(1, N_SEEDS + 1):
                Xm = apply_dropout(X, r, seed)
                for m, f in funcs.items():
                    per_seed[m].append(nmac(s_orig[m], f(Xm, True)[idx]))
            for m in funcs:
                values[(ds, m, r)] = np.array(per_seed[m])
            print(f"    masking {int(r*100)}% done ({time.time()-t0:.0f}s)")
    return values, rules


# ════════════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════════════
def trajectory_figure(values, ylabel, outstem, ylim=None, ref_line=None):
    """Two panels (PBMC 3K | Paul15): mean +/- SD across seeds per masking level."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for method in ORDER:
            v = [values[(ds, method, r)] for r in MASKING_LEVELS]
            ax.errorbar(MASKING_PCT, [np.nanmean(x) for x in v],
                        yerr=[np.nanstd(x, ddof=1) for x in v],
                        capsize=2, label=method, **STYLE[method])
        if ref_line is not None:
            ax.axhline(ref_line, color="black", linewidth=0.8,
                       linestyle=":", zorder=0)
        ax.set_xlabel(XLABEL)
        ax.set_xticks(MASKING_PCT)
        ax.set_xlim(5, 75)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linewidth=0.4, alpha=0.4)
        ax.set_title(DATASET_LABEL[ds])

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ORDER),
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(f"{outstem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_1(tau, outstem):
    trajectory_figure(tau, "Kendall's $\\tau_b$", outstem, ylim=(0.15, 1.0))


def figure_2(nmac_values, outstem):
    ymax = max(np.nanmean(nmac_values[(ds, m, r)]) +
               np.nanstd(nmac_values[(ds, m, r)], ddof=1)
               for ds in DATASETS for m in ORDER for r in MASKING_LEVELS)
    trajectory_figure(nmac_values, "NMAC", outstem,
                      ylim=(0, max(1.1, ymax * 1.08)), ref_line=1.0)


def hodges_lehmann(d):
    """Hodges-Lehmann estimator: median of the Walsh averages."""
    d = np.asarray(d, dtype=float)
    i, j = np.triu_indices(len(d), k=0)
    return np.median((d[i] + d[j]) / 2.0)


def figure_gmcc_cosine(tau, outstem):
    """Exploratory: GMCC - Cosine difference (not in the manuscript)."""
    hl = {ds: [hodges_lehmann(tau[(ds, "GMCC", r)] - tau[(ds, "Cosine", r)])
               for r in MASKING_LEVELS] for ds in DATASETS}
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", zorder=1)
    ax.plot(MASKING_PCT, hl["PBMC3K"], marker="o", linestyle="-",
            color="black", label="PBMC 3K", zorder=3)
    ax.plot(MASKING_PCT, hl["Paul15"], marker="s", linestyle="--",
            color="0.40", label="Paul15", zorder=3)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Hodges–Lehmann difference in Kendall's $\\tau_b$\n"
                  "(GMCC $-$ Cosine)")
    ax.set_xticks(MASKING_PCT)
    ax.set_xlim(5, 75)
    ax.grid(True, linewidth=0.4, alpha=0.4, zorder=0)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{outstem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# Console summary
# ════════════════════════════════════════════════════════════════════
def print_summary(values, title):
    print(f"\n{title}, mean +/- SD across {N_SEEDS} seeds:")
    for ds in DATASETS:
        print(f"\n  {DATASET_LABEL[ds]}")
        print("    " + f"{'Method':<10}" +
              "".join(f"{int(r*100):>16}%" for r in MASKING_LEVELS))
        for method in ORDER:
            row = f"    {method:<10}"
            for r in MASKING_LEVELS:
                v = values[(ds, method, r)]
                row += f"{np.nanmean(v):>10.4f}+-{np.nanstd(v, ddof=1):.4f}"
            print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tau_npz", nargs="?", default=DEFAULT_TAU_NPZ)
    ap.add_argument("--recompute-nmac", action="store_true",
                    help="recompute NMAC even if raw_nmac_per_seed.npz exists")
    args = ap.parse_args()

    tau = load_per_seed(args.tau_npz,
                        "Run src/stability/paired_difference_ci.py first, or pass "
                        "the correct path as the first argument.")
    missing = [m for m in ORDER if not any(k[1] == m for k in tau)]
    if missing:
        sys.exit(f"ERROR: estimators missing from the tau .npz: {missing}")
    print(f"Loaded {len(tau)} tau conditions from {args.tau_npz}")

    if os.path.exists(NMAC_NPZ) and not args.recompute_nmac:
        nmac_values = load_per_seed(NMAC_NPZ, "")
        print(f"Loaded cached NMAC from {NMAC_NPZ} (use --recompute-nmac to rebuild)")
    else:
        nmac_values, rules = compute_nmac(tau)
        save_per_seed(nmac_values, NMAC_NPZ)
        print("\nGMCC centering rule used: " +
              ", ".join(f"{DATASET_LABEL[d]} = mu {rules[d]}" for d in DATASETS))

    os.makedirs(FIG_DIR, exist_ok=True)
    figure_1(tau, os.path.join(FIG_DIR, "Figure_1"))
    figure_2(nmac_values, os.path.join(FIG_DIR, "Figure_2"))
    figure_gmcc_cosine(tau, os.path.join(FIG_DIR, "gmcc_vs_cosine_exploratory"))

    print_summary(tau, "Kendall's tau_b (Figure 1)")
    print_summary(nmac_values, "NMAC (Figure 2)")
    print(f"\nWritten to {FIG_DIR}:")
    print("  Figure_1.{pdf,png}                      (manuscript)")
    print("  Figure_2.{pdf,png}                      (manuscript)")
    print("  gmcc_vs_cosine_exploratory.{pdf,png}    (not in manuscript)")
    print(f"Cached NMAC: {NMAC_NPZ}")