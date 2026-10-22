"""
Co-detection of network edges in the unperturbed 95th-percentile networks.

For every edge (gene pair above the estimator-specific 95th percentile) the
script counts the cells in which BOTH genes are detected (non-zero) and
compares this with the count expected if the two genes were detected
independently (n_cells * d_i * d_j, with d the detection rate).

An edge with no co-detected cell links two genes that are never expressed
in the same cell; its similarity can only come from shared zeros.

Usage
-----
    python src/analysis/edge_codetection_check.py
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src", "stability"))
from paired_difference_ci import (  # noqa: E402
    gmcc_vectorized, pearson_vectorized, cosine_vectorized,
    spearman_vectorized)

DATA_DIR = os.path.join(REPO, "data", "processed")
OUT_DIR = os.path.join(REPO, "results", "low_detection_control")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for ds, xf, mf in [("PBMC3K", "X_2000hvg.npy", "global_mean_2000hvg.npy"),
                       ("Paul15", "X_paul15.npy", "global_mean_paul15.npy")]:
        X = np.load(os.path.join(DATA_DIR, xf))
        mu = np.load(os.path.join(DATA_DIR, mf))[0]
        n_cells = X.shape[1]
        idx = np.triu_indices(X.shape[0], k=1)
        B = (X > 0).astype(np.float64)
        det = B.mean(axis=1)
        co = (B @ B.T)[idx]
        expected = (np.outer(det, det) * n_cells)[idx]
        S = {"GMCC": gmcc_vectorized(X, mu)[idx],
             "Pearson": pearson_vectorized(X)[idx],
             "Cosine": cosine_vectorized(X)[idx],
             "Spearman": spearman_vectorized(X)[idx]}
        print(f"\n{ds}: genes detected in <= 1% of cells: "
              f"{np.mean(det <= 0.01) * 100:.1f}%; <= 5%: "
              f"{np.mean(det <= 0.05) * 100:.1f}%")
        for m, s in S.items():
            e = s > np.percentile(s, 95)
            edge_det = np.r_[det[idx[0][e]], det[idx[1][e]]]
            r = [ds, m, int(e.sum()),
                 f"{np.mean(co[e] == 0) * 100:.1f}",
                 f"{np.mean(co[e] > expected[e]) * 100:.1f}",
                 f"{np.median(edge_det) * 100:.2f}"]
            rows.append(r)
            print(f"  {m:<9} edges={r[2]:,}  no co-detected cell={r[3]}%  "
                  f"co-detection above independence={r[4]}%  "
                  f"median detection of edge genes={r[5]}%")
        sub = np.random.default_rng(0).choice(len(co), 200_000, replace=False)
        print(f"  rank correlation over pairs: GMCC~Pearson "
              f"{spearmanr(S['GMCC'][sub], S['Pearson'][sub])[0]:+.3f}, "
              f"GMCC~Cosine {spearmanr(S['GMCC'][sub], S['Cosine'][sub])[0]:+.3f}")

    path = os.path.join(OUT_DIR, "edge_codetection.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "estimator", "n_edges",
                    "pct_edges_no_codetected_cell",
                    "pct_edges_codetection_above_independence",
                    "median_detection_pct_of_edge_genes"])
        w.writerows(rows)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
