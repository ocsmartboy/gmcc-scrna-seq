
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# ── Configuration ────────────────────────────────────────────────────
DEFAULT_NPZ = ("d:/S3/Eksperimen/Bioinformatika/gmcc_biomedical/"
               "data/processed/raw_kendall_tau_per_seed.npz")

DROPOUT_LEVELS = [0.10, 0.30, 0.50, 0.70]
DROPOUT_PCT = np.array([int(r * 100) for r in DROPOUT_LEVELS])

DATASETS = ["PBMC3K", "Paul15"]
DATASET_LABEL = {"PBMC3K": "PBMC 3K", "Paul15": "Paul15"}

# Plot order: GMCC first, then comparators
ORDER = ["GMCC", "Cosine", "Bicor-SD", "Spearman", "Pearson"]

# Distinguishable in grayscale: unique marker + linestyle per method
STYLE = {
    "GMCC":     dict(marker="o", linestyle="-",  color="black"),
    "Cosine":   dict(marker="s", linestyle="--", color="0.35"),
    "Bicor-SD": dict(marker="^", linestyle="-.", color="0.50"),
    "Spearman": dict(marker="v", linestyle=":",  color="0.60"),
    "Pearson":  dict(marker="D", linestyle=(0, (3, 1, 1, 1)), color="0.70"),
}

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


# ── Data loading ─────────────────────────────────────────────────────
def load_tau(npz_path):
    """Load per-seed Kendall's Tau. Returns {(dataset, method, rate): array}."""
    if not os.path.exists(npz_path):
        sys.exit(
            f"ERROR: {npz_path} not found.\n"
            "Run paired_difference_ci.py first to generate it, or pass the\n"
            "correct path as the first command-line argument."
        )
    data = np.load(npz_path)
    tau = {}
    for key in data.files:
        ds, method, rate = key.split("|")
        tau[(ds, method, float(rate))] = data[key]
    return tau


# ── Statistics (identical to paired_difference_ci.py) ────────────────
def hodges_lehmann(d):
    """Hodges-Lehmann estimator: median of the Walsh averages."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    i, j = np.triu_indices(n, k=0)
    return np.median(np.sort((d[i] + d[j]) / 2.0))


# ── Figure 1: stability trajectories ─────────────────────────────────
def figure_1(tau, outstem):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        for method in ORDER:
            means = [tau[(ds, method, r)].mean() for r in DROPOUT_LEVELS]
            sds = [tau[(ds, method, r)].std(ddof=1) for r in DROPOUT_LEVELS]
            ax.errorbar(DROPOUT_PCT, means, yerr=sds, capsize=2,
                        label=method, **STYLE[method])
        ax.set_xlabel("Additional dropout (%)")
        ax.set_xticks(DROPOUT_PCT)
        ax.set_xlim(5, 75)
        ax.set_ylim(0.15, 1.0)
        ax.grid(True, linewidth=0.4, alpha=0.4)
        ax.set_title(DATASET_LABEL[ds])

    axes[0].set_ylabel("Kendall's $\\tau$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    for ext in ("pdf", "png"):
        fig.savefig(f"{outstem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: GMCC vs Cosine reversal ────────────────────────────────
def figure_2(tau, outstem):
    hl = {
        ds: [hodges_lehmann(tau[(ds, "GMCC", r)] - tau[(ds, "Cosine", r)])
             for r in DROPOUT_LEVELS]
        for ds in DATASETS
    }

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.axhspan(0, 0.22, color="0.93", zorder=0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", zorder=1)
    ax.plot(DROPOUT_PCT, hl["PBMC3K"], marker="o", linestyle="-",
            color="black", label="PBMC 3K", zorder=3)
    ax.plot(DROPOUT_PCT, hl["Paul15"], marker="s", linestyle="--",
            color="0.40", label="Paul15", zorder=3)

    ax.annotate("GMCC more stable", xy=(68, 0.185), ha="right",
                fontsize=7.5, style="italic", color="0.30")
    ax.annotate("Cosine more stable", xy=(68, -0.205), ha="right",
                fontsize=7.5, style="italic", color="0.30")

    ax.set_xlabel("Additional dropout (%)")
    ax.set_ylabel("Hodges\u2013Lehmann difference in Kendall's $\\tau$\n"
                  "(GMCC $-$ Cosine)")
    ax.set_xticks(DROPOUT_PCT)
    ax.set_xlim(5, 75)
    ax.set_ylim(-0.22, 0.22)
    ax.grid(True, linewidth=0.4, alpha=0.4, zorder=0)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.80))
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(f"{outstem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ── Console summary: values as plotted ───────────────────────────────
def print_summary(tau):
    print("\nKendall's Tau, mean +/- SD across 20 seeds (kendall_tau_dropout):")
    for ds in DATASETS:
        print(f"\n  {DATASET_LABEL[ds]}")
        header = "    " + f"{'Method':<10}" + "".join(
            f"{int(r*100):>16}%" for r in DROPOUT_LEVELS)
        print(header)
        for method in ORDER:
            row = f"    {method:<10}"
            for r in DROPOUT_LEVELS:
                v = tau[(ds, method, r)]
                row += f"{v.mean():>10.4f}+-{v.std(ddof=1):.4f}"
            print(row)

    print("\nHodges-Lehmann difference, GMCC - Cosine (gmcc_vs_cosine_diff):")
    for ds in DATASETS:
        vals = [hodges_lehmann(tau[(ds, "GMCC", r)] - tau[(ds, "Cosine", r)])
                for r in DROPOUT_LEVELS]
        print(f"    {DATASET_LABEL[ds]:<10}" +
              "".join(f"{v:>+10.4f}" for v in vals))


if __name__ == "__main__":
    npz_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NPZ
    tau = load_tau(npz_path)

    n_expected = len(DATASETS) * len(ORDER) * len(DROPOUT_LEVELS)
    print(f"Loaded {len(tau)} conditions from {npz_path} "
          f"(expected {n_expected})")
    if len(tau) != n_expected:
        print("WARNING: unexpected number of conditions; check the .npz file.")

    figure_1(tau, "Figure_1")
    figure_2(tau, "Figure_2")
    print_summary(tau)
    print("\nWritten: kendall_tau_dropout.{pdf,png}, gmcc_vs_cosine_diff.{pdf,png}")