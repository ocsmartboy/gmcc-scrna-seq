"""
Remedy analysis: repeat the main robustness and network analyses on genes
that are adequately detected.

From the 2,000 highly variable genes, only genes detected (non-zero) in at
least a given fraction of cells of the unperturbed matrix are retained
(default thresholds: 1%, 5% and 10% of cells). For each filtered matrix:

  * the global mean of GMCC is recomputed from the analysed (filtered)
    matrix, and again from every masked matrix, as in the primary analysis;
  * masking uses exactly the same random draws as the primary analysis
    (seeds 1-20, nested levels): the full matrix is masked and the retained
    genes are then selected;
  * rank robustness (Kendall's tau_b; all pairs and, within the filtered
    genes, detection-tertile pair sets LL / not_low / HH) and NMAC are
    computed for GMCC,
    Pearson, Cosine and Spearman, with paired GMCC-minus-comparator
    Hodges-Lehmann estimates, exact Wilcoxon tests and Holm correction;
  * the unperturbed 95th-percentile network of each estimator is described:
    share of edges without any co-detected cell, median detection of edge
    genes, Spearman correlation between degree and detection rate, average
    clustering coefficient, number of connected components and of modules
    (components with >= 10 genes).

Usage
-----
    python src/analysis/detection_filter_analysis.py
    python src/analysis/detection_filter_analysis.py --thresholds 0.05
"""
import argparse
import csv
import os
import sys
import time
import warnings

import networkx as nx
import numpy as np
from scipy.stats import kendalltau, spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src", "stability"))
from paired_difference_ci import (  # noqa: E402
    gmcc_vectorized, pearson_vectorized, cosine_vectorized,
    spearman_vectorized, apply_masking, hodges_lehmann_ci, holm)

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(REPO, "data", "processed")
OUT_DIR = os.path.join(REPO, "results", "detection_filter")
N_SEEDS = 20
MASKING_LEVELS = [0.10, 0.30, 0.50, 0.70]
METHODS = ["GMCC", "Pearson", "Cosine", "Spearman"]
COMPARATORS = ["Pearson", "Cosine", "Spearman"]
DATASETS = {"PBMC3K": "X_2000hvg.npy", "Paul15": "X_paul15.npy"}


def similarities(X):
    """All estimators; GMCC uses the mean of the analysed matrix."""
    return {"GMCC": gmcc_vectorized(X, X.mean()),
            "Pearson": pearson_vectorized(X),
            "Cosine": cosine_vectorized(X),
            "Spearman": spearman_vectorized(X)}


def nmac(s_orig, s_mask):
    mad = np.median(np.abs(s_orig - np.median(s_orig)))
    return np.nan if mad < 1e-12 else np.median(np.abs(s_mask - s_orig)) / mad


def network_summary(s, idx, n_genes, n_cells, det, co):
    edge = s > np.percentile(s, 95)
    G = nx.Graph()
    G.add_nodes_from(range(n_genes))
    G.add_edges_from(zip(idx[0][edge], idx[1][edge]))
    deg = np.array([d for _, d in G.degree()])
    comps = [len(c) for c in nx.connected_components(G)]
    edge_det = np.r_[det[idx[0][edge]], det[idx[1][edge]]]
    return {
        "edges": int(edge.sum()),
        "pct_edges_no_codetection": float(np.mean(co[edge] == 0) * 100),
        "pct_edges_codetection_above_independence": float(np.mean(
            co[edge] > (np.outer(det, det) * n_cells)[idx][edge]) * 100),
        "median_detection_edge_genes_pct": float(np.median(edge_det) * 100),
        "rho_degree_detection": float(spearmanr(deg, det)[0]),
        "avg_clustering": float(nx.average_clustering(G)),
        "components": len(comps),
        "modules_ge10": int(sum(c >= 10 for c in comps)),
        "largest_component": int(max(comps)),
    }


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved: {path}")


def run(thresholds, n_seeds):
    os.makedirs(OUT_DIR, exist_ok=True)
    tau, nm = {}, {}
    gene_rows, net_rows = [], []

    for ds, xf in DATASETS.items():
        X_full = np.load(os.path.join(DATA_DIR, xf))
        det_full = (X_full > 0).mean(axis=1)
        for thr in thresholds:
            keep = det_full >= thr
            X = X_full[keep]
            n_genes, n_cells = X.shape
            det = det_full[keep]
            idx = np.triu_indices(n_genes, k=1)
            sparsity = float((X == 0).mean() * 100)
            print(f"\n{'=' * 72}\n{ds}: detection >= {thr * 100:g}% of cells -> "
                  f"{n_genes} genes, {len(idx[0]):,} pairs, sparsity "
                  f"{sparsity:.1f}%, mu = {X.mean():.4f}\n{'=' * 72}")
            gene_rows.append([ds, thr, n_genes, len(idx[0]),
                              f"{sparsity:.2f}", f"{X.mean():.4f}"])

            S_ref = {k: v[idx] for k, v in similarities(X).items()}
            t_lo, t_hi = np.quantile(det, [1 / 3, 2 / 3])
            lo_, hi_ = det <= t_lo, det > t_hi
            psets = {"all": np.ones(len(idx[0]), dtype=bool),
                     "LL": lo_[idx[0]] & lo_[idx[1]],
                     "not_low": ~lo_[idx[0]] & ~lo_[idx[1]],
                     "HH": hi_[idx[0]] & hi_[idx[1]]}
            B = (X > 0).astype(np.float64)
            co = (B @ B.T)[idx]
            for m in METHODS:
                ns = network_summary(S_ref[m], idx, n_genes, n_cells, det, co)
                net_rows.append([ds, thr, m] + [
                    f"{v:.4f}" if isinstance(v, float) else v
                    for v in ns.values()])
                print(f"  network {m:<9} no co-detection "
                      f"{ns['pct_edges_no_codetection']:5.1f}%  median det "
                      f"{ns['median_detection_edge_genes_pct']:5.1f}%  "
                      f"rho(deg,det) {ns['rho_degree_detection']:+.3f}  "
                      f"C {ns['avg_clustering']:.3f}  modules "
                      f"{ns['modules_ge10']}  largest {ns['largest_component']}")

            for rate in MASKING_LEVELS:
                t0 = time.time()
                acc_t = {(p, m): [] for p in psets for m in METHODS}
                acc_n = {m: [] for m in METHODS}
                for seed in range(1, n_seeds + 1):
                    Xm = apply_masking(X_full, rate, seed)[keep]
                    S = {k: v[idx] for k, v in similarities(Xm).items()}
                    for m in METHODS:
                        for p, kp in psets.items():
                            t, _ = kendalltau(S_ref[m][kp], S[m][kp])
                            acc_t[(p, m)].append(0.0 if np.isnan(t) else t)
                        acc_n[m].append(nmac(S_ref[m], S[m]))
                for m in METHODS:
                    for p in psets:
                        tau[(ds, thr, p, m, rate)] = np.array(acc_t[(p, m)])
                    nm[(ds, thr, m, rate)] = np.array(acc_n[m])
                print(f"  masking {rate * 100:.0f}%: " + "  ".join(
                    f"{m} {np.mean(acc_t[('all', m)]):.4f}" for m in METHODS)
                    + f"  ({time.time() - t0:.0f}s)")

    tag = "_".join(f"{t * 100:g}" for t in thresholds)
    np.savez(os.path.join(OUT_DIR, f"tau_per_seed_{tag}.npz"),
             **{f"{d}|{t}|{p}|{m}|{r}": v for (d, t, p, m, r), v in tau.items()})
    np.savez(os.path.join(OUT_DIR, f"nmac_per_seed_{tag}.npz"),
             **{f"{d}|{t}|{m}|{r}": v for (d, t, m, r), v in nm.items()})
    write_csv(os.path.join(OUT_DIR, f"gene_sets_{tag}.csv"),
              ["dataset", "min_detection", "n_genes", "n_pairs",
               "sparsity_pct", "global_mean"], gene_rows)
    write_csv(os.path.join(OUT_DIR, f"networks_{tag}.csv"),
              ["dataset", "min_detection", "estimator", "edges",
               "pct_edges_no_codetection",
               "pct_edges_codetection_above_independence",
               "median_detection_edge_genes_pct",
               "rho_degree_detection", "avg_clustering", "components",
               "modules_ge10", "largest_component"], net_rows)

    rows, test_rows = [], []
    PSETS = ["all", "LL", "not_low", "HH"]
    for ds in DATASETS:
        for thr in thresholds:
            for p in PSETS:
                for m in METHODS:
                    vt = [tau[(ds, thr, p, m, r)].mean() for r in MASKING_LEVELS]
                    vn = ([np.nanmean(nm[(ds, thr, m, r)]) for r in MASKING_LEVELS]
                          if p == "all" else [np.nan] * 4)
                    rel = (vt[0] - vt[-1]) / vt[0] * 100
                    rows.append([ds, thr, p, m] + [f"{v:.4f}" for v in vt]
                                + [f"{np.mean(vt):.4f}", f"{rel:.1f}"]
                                + [f"{v:.4f}" for v in vn])
                for rate in MASKING_LEVELS:
                    res = []
                    for comp in COMPARATORS:
                        d = (tau[(ds, thr, p, "GMCC", rate)]
                             - tau[(ds, thr, p, comp, rate)])
                        hl, lo, hi = hodges_lehmann_ci(d)
                        W, pv = wilcoxon(d, method="exact")
                        res.append((comp, hl, lo, hi, W, pv, int((d > 0).sum())))
                    adj = holm([r[5] for r in res])
                    for (comp, hl, lo, hi, W, pv, npos), pa in zip(res, adj):
                        test_rows.append([ds, thr, p, rate, comp, f"{hl:+.4f}",
                                          f"{lo:+.4f}", f"{hi:+.4f}", f"{W:.1f}",
                                          f"{pa:.3e}", f"{npos}/{n_seeds}"])
    write_csv(os.path.join(OUT_DIR, f"robustness_{tag}.csv"),
              ["dataset", "min_detection", "pair_set", "estimator"]
              + [f"tau_{int(r * 100)}" for r in MASKING_LEVELS]
              + ["avg_tau", "relative_decrease_pct"]
              + [f"nmac_{int(r * 100)}" for r in MASKING_LEVELS], rows)
    write_csv(os.path.join(OUT_DIR, f"paired_tests_{tag}.csv"),
              ["dataset", "min_detection", "pair_set", "masking", "comparison", "HL",
               "CI_low", "CI_high", "W", "p_holm", "GMCC_better"], test_rows)


def main():
    ap = argparse.ArgumentParser(
        description="Robustness and networks on detection-filtered genes.")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.01, 0.05, 0.10])
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()
    run(a.thresholds, a.n_seeds)


if __name__ == "__main__":
    main()
