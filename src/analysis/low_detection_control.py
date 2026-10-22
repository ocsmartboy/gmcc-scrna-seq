"""
Control analysis: does GMCC's rank robustness depend on pairs of
low-detection genes (shared zeros counted as agreement)?

Motivation
----------
GMCC centres every gene on one matrix-level mean mu. A gene detected in
few cells is then almost a constant vector of -mu, so two such genes look
similar mainly because they share zeros. Masking adds zeros, which may
preserve these similarities and inflate rank robustness.

What the script does
--------------------
Genes are split into dataset-specific tertiles of detection rate (fraction
of cells with a non-zero value) in the unperturbed matrix. Gene pairs are
then grouped into fixed pair sets:

    all        every gene pair (reproduces the primary analysis)
    excl_LL    all pairs except those in which BOTH genes are low-detection
    LL         both genes low-detection
    L_other    exactly one gene low-detection
    not_low    neither gene low-detection
    HH         both genes high-detection (top tertile)

For every dataset, masking level and masking realisation (seeds 1-20,
nested masking, identical to paired_difference_ci.py) Kendall's tau_b
between the unperturbed and masked similarity values is computed within
each pair set. The same pair set is used for all estimators, so paired
GMCC-minus-comparator differences remain valid; they are summarised by
Hodges-Lehmann estimates with exact CIs, exact Wilcoxon signed-rank tests
and Holm correction over Pearson, Cosine and Spearman.

Descriptive diagnostics on the unperturbed matrices:
  * Spearman correlation between each estimator's similarity and the
    shared-zero fraction of the pair (fraction of cells where both genes
    are zero), on a random subsample of 200,000 pairs (seed 0);
  * mean similarity per pair set;
  * composition of the 95th-percentile network (share of edges per set).

Pair sets use a sensitivity threshold as well: --split quartile uses the
bottom/top quartiles instead of tertiles.

Usage
-----
    python src/analysis/low_detection_control.py
    python src/analysis/low_detection_control.py --split quartile
"""
import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import kendalltau, spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src", "stability"))
from paired_difference_ci import (  # noqa: E402  (identical definitions)
    gmcc_vectorized, pearson_vectorized, cosine_vectorized,
    spearman_vectorized, apply_masking, hodges_lehmann_ci, holm)

warnings.filterwarnings("ignore")

DEFAULT_DATA_DIR = os.path.join(REPO, "data", "processed")
DEFAULT_OUT_DIR = os.path.join(REPO, "results", "low_detection_control")

N_SEEDS = 20
MASKING_LEVELS = [0.10, 0.30, 0.50, 0.70]
METHODS = ["GMCC", "Pearson", "Cosine", "Spearman"]
COMPARATORS = ["Pearson", "Cosine", "Spearman"]
PAIR_SETS = ["all", "excl_LL", "LL", "L_other", "not_low", "HH"]


def similarities(X, mu):
    return {
        "GMCC": gmcc_vectorized(X, mu),
        "Pearson": pearson_vectorized(X),
        "Cosine": cosine_vectorized(X),
        "Spearman": spearman_vectorized(X),
    }


def pair_sets(X, idx, split):
    det = (X > 0).mean(axis=1)
    q = (1 / 3, 2 / 3) if split == "tertile" else (0.25, 0.75)
    t_lo, t_hi = np.quantile(det, q)
    low, high = det <= t_lo, det > t_hi
    li, lj = low[idx[0]], low[idx[1]]
    hi_, hj = high[idx[0]], high[idx[1]]
    sets = {
        "all": np.ones(len(idx[0]), dtype=bool),
        "excl_LL": ~(li & lj),
        "LL": li & lj,
        "L_other": li ^ lj,
        "not_low": ~li & ~lj,
        "HH": hi_ & hj,
    }
    return sets, det, (t_lo, t_hi)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved: {path}")


def run(data_dir, out_dir, n_seeds, split):
    os.makedirs(out_dir, exist_ok=True)
    datasets = {
        "PBMC3K": ("X_2000hvg.npy", "global_mean_2000hvg.npy"),
        "Paul15": ("X_paul15.npy", "global_mean_paul15.npy"),
    }
    tau = {}
    diag_rows, comp_rows = [], []

    for ds, (xf, mf) in datasets.items():
        X = np.load(os.path.join(data_dir, xf))
        mu0 = np.load(os.path.join(data_dir, mf))[0]
        n_genes, n_cells = X.shape
        idx = np.triu_indices(n_genes, k=1)
        sets, det, (t_lo, t_hi) = pair_sets(X, idx, split)
        print(f"\n{'=' * 72}\n{ds}: {n_genes} genes, {n_cells} cells, "
              f"{split} thresholds on detection rate: low <= {t_lo:.4f}, "
              f"high > {t_hi:.4f}\n{'=' * 72}")
        for s, m in sets.items():
            print(f"  pair set {s:<8} {m.sum():>9,d} pairs ({m.mean() * 100:5.1f}%)")

        # ── Diagnostics on the unperturbed matrix ────────────────────
        S_ref = {k: v[idx] for k, v in similarities(X, mu0).items()}
        Z = (X == 0).astype(np.float32)
        shared_zero = ((Z @ Z.T) / n_cells)[idx]
        sub = np.random.default_rng(0).choice(len(shared_zero), 200_000,
                                              replace=False)
        print("\n  Unperturbed matrix diagnostics")
        for m in METHODS:
            s = S_ref[m]
            rho = spearmanr(s[sub], shared_zero[sub])[0]
            edge = s > np.percentile(s, 95)
            means = {k: float(s[v].mean()) for k, v in sets.items()}
            share = {k: float(v[edge].mean() * 100) for k, v in sets.items()}
            print(f"    {m:<9} rho(similarity, shared-zero fraction) = {rho:+.3f}; "
                  f"95th-percentile edges in LL = {share['LL']:.1f}%, "
                  f"HH = {share['HH']:.1f}%")
            diag_rows.append([ds, split, m, f"{rho:.4f}"]
                             + [f"{means[k]:.4f}" for k in PAIR_SETS[2:]]
                             + [f"{share[k]:.2f}" for k in PAIR_SETS[2:]])

        # ── Rank robustness within pair sets ─────────────────────────
        for rate in MASKING_LEVELS:
            t0 = time.time()
            acc = {(p, m): [] for p in PAIR_SETS for m in METHODS}
            for seed in range(1, n_seeds + 1):
                Xm = apply_masking(X, rate, seed)
                S = {k: v[idx] for k, v in similarities(Xm, Xm.mean()).items()}
                for m in METHODS:
                    for p, keep in sets.items():
                        t, _ = kendalltau(S_ref[m][keep], S[m][keep])
                        acc[(p, m)].append(0.0 if np.isnan(t) else t)
            for (p, m), v in acc.items():
                tau[(ds, p, m, rate)] = np.array(v)
            print(f"  masking {rate * 100:.0f}% done ({time.time() - t0:.0f}s)")

    np.savez(os.path.join(out_dir, f"tau_per_seed_{split}.npz"),
             **{f"{d}|{p}|{m}|{r}": v for (d, p, m, r), v in tau.items()})

    write_csv(os.path.join(out_dir, f"diagnostics_{split}.csv"),
              ["dataset", "split", "estimator", "rho_similarity_shared_zero"]
              + [f"mean_sim_{k}" for k in PAIR_SETS[2:]]
              + [f"edge_share_pct_{k}" for k in PAIR_SETS[2:]], diag_rows)

    # ── Mean tau per pair set ────────────────────────────────────────
    rows = []
    for ds in datasets:
        for p in PAIR_SETS:
            for m in METHODS:
                vals = [tau[(ds, p, m, r)].mean() for r in MASKING_LEVELS]
                rows.append([ds, split, p, m] + [f"{v:.4f}" for v in vals]
                            + [f"{np.mean(vals):.4f}"])
    write_csv(os.path.join(out_dir, f"mean_tau_{split}.csv"),
              ["dataset", "split", "pair_set", "estimator"]
              + [f"tau_{int(r * 100)}" for r in MASKING_LEVELS] + ["avg_tau"],
              rows)

    # ── Paired GMCC-minus-comparator tests ───────────────────────────
    for ds in datasets:
        for p in PAIR_SETS:
            for rate in MASKING_LEVELS:
                res = []
                for comp in COMPARATORS:
                    d = tau[(ds, p, "GMCC", rate)] - tau[(ds, p, comp, rate)]
                    hl, lo, hi = hodges_lehmann_ci(d)
                    W, pv = wilcoxon(d, method="exact")
                    res.append((comp, hl, lo, hi, W, pv, int((d > 0).sum())))
                adj = holm([r[5] for r in res])
                for (comp, hl, lo, hi, W, pv, npos), pa in zip(res, adj):
                    comp_rows.append([ds, split, p, rate, comp, f"{hl:+.4f}",
                                      f"{lo:+.4f}", f"{hi:+.4f}", f"{W:.1f}",
                                      f"{pa:.3e}", f"{npos}/{n_seeds}"])
    write_csv(os.path.join(out_dir, f"paired_tests_{split}.csv"),
              ["dataset", "split", "pair_set", "masking", "comparison", "HL",
               "CI_low", "CI_high", "W", "p_holm", "GMCC_better"], comp_rows)

    # ── Console summary ──────────────────────────────────────────────
    for ds in datasets:
        print(f"\n{'=' * 72}\nAVERAGE TAU ACROSS MASKING LEVELS - {ds} ({split})"
              f"\n{'=' * 72}")
        print(f"  {'pair set':<10}" + "".join(f"{m:>10}" for m in METHODS))
        for p in PAIR_SETS:
            print(f"  {p:<10}" + "".join(
                f"{np.mean([tau[(ds, p, m, r)].mean() for r in MASKING_LEVELS]):>10.4f}"
                for m in METHODS))


def main():
    ap = argparse.ArgumentParser(
        description="Rank robustness within detection-rate pair sets.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--split", choices=["tertile", "quartile"],
                    default="tertile")
    a = ap.parse_args()
    run(a.data_dir, a.out_dir, a.n_seeds, a.split)


if __name__ == "__main__":
    main()
