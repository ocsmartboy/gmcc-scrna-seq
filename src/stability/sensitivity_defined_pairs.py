"""
Sensitivity analysis: Kendall's tau_b restricted to gene pairs whose
similarity is defined for every estimator.

Rationale
---------
In the primary analysis (paired_difference_ci.py), a gene whose profile
becomes constant after masking (in practice: entirely zero) has zero norm
after centring. The guard `norms[norms == 0] = 1e-10` then makes all of its
similarities exactly 0 for Pearson, Spearman, Cosine and Bicor-SD, whereas
GMCC remains defined because the global mean is subtracted. This script
checks whether GMCC's advantage depends on that implicit replacement.

A pair (i, j) is undefined for some estimator iff gene i or gene j has a
constant profile. Hence "exclude genes that become entirely zero" and
"keep only pairs defined for all estimators" select the same pairs.

Two restriction schemes are reported:
  per_level : pairs defined in the unperturbed matrix AND in the masked
              matrix of that realisation (pair set varies with level/seed).
  fixed_70  : pairs defined in the unperturbed matrix AND at 70% masking of
              the same seed. Because masking is nested within a seed
              (same uniform draws, threshold < rate), this set is defined at
              every level, so relative decreases are computed on one
              fixed pair set per realisation.

Within a realisation the same pair set is used for all five estimators,
so the paired GMCC-minus-comparator differences remain valid.

The primary-analysis tau (with the implicit replacement) is recomputed in
the same loop so that it can be checked against Table 1/2 of the
manuscript.

Usage
-----
    python sensitivity_defined_pairs.py                     # repo layout
    python sensitivity_defined_pairs.py --data-dir data/processed \
        --out-dir results
"""
import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import rankdata, kendalltau, norm, wilcoxon

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_DATA_DIR = os.path.join(REPO, "data", "processed")
DEFAULT_OUT_DIR = os.path.join(REPO, "results")

N_SEEDS = 20
MASKING_LEVELS = [0.10, 0.30, 0.50, 0.70]
METHODS = ["GMCC", "Pearson", "Cosine", "Spearman", "Bicor-SD"]
COMPARATORS = ["Pearson", "Cosine", "Spearman", "Bicor-SD"]
SCHEMES = ["primary", "per_level", "fixed_70"]


# ── Similarity measures (identical to paired_difference_ci.py) ───────
def gmcc_vectorized(X, mu):
    X_c = X - mu
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_c / norms) @ (X_c / norms).T, -1.0, 1.0)


def pearson_vectorized(X):
    X_c = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_c / norms) @ (X_c / norms).T, -1.0, 1.0)


def cosine_vectorized(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X / norms) @ (X / norms).T, -1.0, 1.0)


def spearman_vectorized(X):
    return pearson_vectorized(np.apply_along_axis(rankdata, 1, X))


def bicor_sd_vectorized(X):
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True)
    std_x = np.std(X, axis=1, keepdims=True)
    mad_reg = np.where(mad < 1e-8, std_x, mad)
    mad_reg[mad_reg < 1e-10] = 1e-10
    u = (X - med) / (9.0 * mad_reg)
    w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    X_t = w * (X - med)
    norms = np.linalg.norm(X_t, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_t / norms) @ (X_t / norms).T, -1.0, 1.0)


def all_similarities(X, mu):
    return {
        "GMCC":     gmcc_vectorized(X, mu),
        "Pearson":  pearson_vectorized(X),
        "Cosine":   cosine_vectorized(X),
        "Spearman": spearman_vectorized(X),
        "Bicor-SD": bicor_sd_vectorized(X),
    }


def apply_masking(matrix, rate, seed):
    np.random.seed(seed)
    result = matrix.copy()
    nonzero_mask = matrix > 0
    mask = np.random.rand(*matrix.shape) < rate
    result[nonzero_mask & mask] = 0
    return result


def defined_genes(M):
    """True for genes whose profile is not constant. A constant profile
    (in practice entirely zero) makes Pearson, Spearman, Cosine and
    Bicor-SD undefined for every pair that contains it."""
    return np.ptp(M, axis=1) > 0


# ── Statistics (identical to paired_difference_ci.py) ────────────────
def hodges_lehmann_ci(d, alpha=0.05):
    d = np.asarray(d, dtype=float)
    n = len(d)
    i, j = np.triu_indices(n, k=0)
    walsh = np.sort((d[i] + d[j]) / 2.0)
    M = len(walsh)
    z = norm.ppf(1 - alpha / 2)
    k = max(int(np.floor(n * (n + 1) / 4
                         - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24))), 0)
    return np.median(walsh), walsh[k], walsh[M - k - 1]


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.zeros(m)
    running = 0.0
    for rank, idx_ in enumerate(order):
        running = max(running, (m - rank) * p[idx_])
        adj[idx_] = min(running, 1.0)
    return adj


def load(data_dir, name):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found.")
    return np.load(path)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────
def run(data_dir, out_dir, n_seeds):
    os.makedirs(out_dir, exist_ok=True)
    datasets = {
        "PBMC3K": (load(data_dir, "X_2000hvg.npy"),
                   load(data_dir, "global_mean_2000hvg.npy")[0]),
        "Paul15": (load(data_dir, "X_paul15.npy"),
                   load(data_dir, "global_mean_paul15.npy")[0]),
    }
    seeds = range(1, n_seeds + 1)

    tau = {}        # (ds, scheme, method, rate) -> array over seeds
    coverage = []   # rows for coverage CSV

    for ds, (X, mu0) in datasets.items():
        n_genes = X.shape[0]
        idx = np.triu_indices(n_genes, k=1)
        n_pairs = len(idx[0])
        ref_def = defined_genes(X)
        print(f"\n{'=' * 72}\nDATASET: {ds}  genes={n_genes}  "
              f"constant genes in unperturbed matrix={int((~ref_def).sum())}"
              f"\n{'=' * 72}")

        S_ref = {k: v[idx] for k, v in all_similarities(X, mu0).items()}

        # Fixed pair set per seed: genes still defined at 70% masking.
        keep70 = {}
        for s in seeds:
            g = defined_genes(apply_masking(X, 0.70, s)) & ref_def
            keep70[s] = g[idx[0]] & g[idx[1]]

        for rate in MASKING_LEVELS:
            t0 = time.time()
            acc = {(sch, m): [] for sch in SCHEMES for m in METHODS}
            n_undef, n_excl = [], []

            for s in seeds:
                Xm = apply_masking(X, rate, s)
                S = {k: v[idx] for k, v in
                     all_similarities(Xm, Xm.mean()).items()}

                g = defined_genes(Xm) & ref_def
                keep = g[idx[0]] & g[idx[1]]
                n_undef.append(int((~defined_genes(Xm)).sum()))
                n_excl.append(int(n_pairs - keep.sum()))

                for m in METHODS:
                    t, _ = kendalltau(S_ref[m], S[m])
                    acc[("primary", m)].append(0.0 if np.isnan(t) else t)
                    t, _ = kendalltau(S_ref[m][keep], S[m][keep])
                    acc[("per_level", m)].append(t)
                    k70 = keep70[s]
                    t, _ = kendalltau(S_ref[m][k70], S[m][k70])
                    acc[("fixed_70", m)].append(t)

            for (sch, m), v in acc.items():
                tau[(ds, sch, m, rate)] = np.array(v)
            coverage.append([
                ds, rate, n_genes, int(np.min(n_undef)),
                int(np.max(n_undef)), round(float(np.mean(n_undef)), 2),
                n_pairs, round(float(np.mean(n_excl)), 1),
                round(100 * float(np.mean(n_excl)) / n_pairs, 3),
                round(100 * float(np.mean(
                    [n_pairs - keep70[s].sum() for s in seeds])) / n_pairs, 3),
            ])
            print(f"  {rate * 100:.0f}%: done ({time.time() - t0:.1f}s), "
                  f"undefined genes per realisation "
                  f"{min(n_undef)}-{max(n_undef)}")

    # Any NaN tau would silently break the paired tests.
    nan_cells = [k for k, v in tau.items() if np.isnan(v).any()]
    if nan_cells:
        print("\nWARNING: NaN tau in", nan_cells)

    np.savez(os.path.join(out_dir, "sensitivity_tau_per_seed.npz"),
             **{f"{d}|{sc}|{m}|{r}": v for (d, sc, m, r), v in tau.items()})

    write_csv(
        os.path.join(out_dir, "sensitivity_coverage.csv"),
        ["dataset", "masking", "n_genes", "undefined_genes_min",
         "undefined_genes_max", "undefined_genes_mean", "n_pairs",
         "excluded_pairs_mean", "excluded_pairs_pct_per_level",
         "excluded_pairs_pct_fixed_70"],
        coverage)

    # ── Mean tau and relative decrease ───────────────────────────────
    rows = []
    for ds in datasets:
        for sch in SCHEMES:
            for m in METHODS:
                vals = [tau[(ds, sch, m, r)].mean() for r in MASKING_LEVELS]
                rel = (vals[0] - vals[-1]) / vals[0] * 100
                rows.append([ds, sch, m] + [f"{v:.4f}" for v in vals]
                            + [f"{np.mean(vals):.4f}", f"{rel:.1f}"])
    write_csv(
        os.path.join(out_dir, "sensitivity_mean_tau.csv"),
        ["dataset", "scheme", "method"]
        + [f"tau_{int(r * 100)}" for r in MASKING_LEVELS]
        + ["avg_tau", "relative_decrease_pct"],
        rows)

    # ── Paired GMCC-minus-comparator tests ───────────────────────────
    # Holm over Pearson, Cosine, Spearman; Bicor-SD reported unadjusted,
    # exactly as in the primary analysis.
    rows = []
    for ds in datasets:
        for sch in SCHEMES:
            for rate in MASKING_LEVELS:
                res = []
                for comp in COMPARATORS:
                    d = (tau[(ds, sch, "GMCC", rate)]
                         - tau[(ds, sch, comp, rate)])
                    hl, lo, hi = hodges_lehmann_ci(d)
                    W, p = wilcoxon(d)
                    res.append((comp, hl, lo, hi, W, p,
                                int((d > 0).sum()), len(d)))
                adj = list(holm([r[5] for r in res[:3]])) + [res[3][5]]
                for (comp, hl, lo, hi, W, p, npos, n), pa in zip(res, adj):
                    rows.append([ds, sch, rate, comp, f"{hl:+.4f}",
                                 f"{lo:+.4f}", f"{hi:+.4f}", f"{W:.1f}",
                                 f"{p:.3e}", f"{pa:.3e}", f"{npos}/{n}"])
    write_csv(
        os.path.join(out_dir, "sensitivity_paired_tests.csv"),
        ["dataset", "scheme", "masking", "comparison", "HL", "CI_low",
         "CI_high", "W", "p_raw", "p_reported", "GMCC_better"],
        rows)

    # ── Console summary: primary vs restricted, side by side ─────────
    for ds in datasets:
        print(f"\n{'=' * 72}\nMEAN TAU - {ds}  (primary | per_level | "
              f"fixed_70)\n{'=' * 72}")
        for rate in MASKING_LEVELS:
            line = f"  {rate * 100:>3.0f}%"
            for m in METHODS:
                v = [tau[(ds, sc, m, rate)].mean() for sc in SCHEMES]
                line += f"  {m[:4]:>4} {v[0]:.3f}|{v[1]:.3f}|{v[2]:.3f}"
            print(line)


def main():
    ap = argparse.ArgumentParser(
        description="Kendall's tau_b sensitivity analysis restricted to "
                    "gene pairs defined for every estimator.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()
    run(a.data_dir, a.out_dir, a.n_seeds)


if __name__ == "__main__":
    main()
