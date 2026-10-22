"""
Network analyses (manuscript Tables 3-4, Supplementary Tables S3-S4)
repeated on genes that are adequately detected.

Gene sets: the 2,000 highly variable genes ("all", reproduces the current
tables) and the subsets detected (non-zero) in at least 5% and 10% of the
cells of the unperturbed matrix. GMCC uses the global mean of the analysed
matrix (recomputed after filtering and after masking).

For each dataset, gene set and estimator (GMCC, Pearson, Cosine, Spearman):

  Table 3   average clustering coefficient (networkx, all nodes) and number
            of connected components of the unperturbed 95th-percentile
            network; cross-dataset |difference| of the clustering
            coefficient
  Table 4   Spearman correlation between node degree and detection rate
  edges     share of edges without any co-detected cell and with
            co-detection above the independence expectation
  S3        average clustering coefficient under masking (seeds 1-20,
            nested levels; masks drawn on the full matrix, then filtered,
            so realisations are identical to the primary analysis);
            min / max cross-dataset |difference| over masking levels
  S4        clustering coefficient at the 90th, 95th and 97.5th percentiles

Usage
-----
    python src/network/filtered_networks.py
"""
import csv
import os
import sys
import time

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src", "stability"))
from paired_difference_ci import (  # noqa: E402
    gmcc_vectorized, pearson_vectorized, cosine_vectorized,
    spearman_vectorized, apply_masking)
from network_topology_masking import fast_average_clustering  # noqa: E402

DATA_DIR = os.path.join(REPO, "data", "processed")
OUT_DIR = os.path.join(REPO, "results", "filtered_networks")
DATASETS = {"PBMC3K": "X_2000hvg.npy", "Paul15": "X_paul15.npy"}
GENE_SETS = {"all": 0.0, "det5": 0.05, "det10": 0.10}
METHODS = ["GMCC", "Pearson", "Cosine", "Spearman"]
MASKING_LEVELS = [0.10, 0.30, 0.50, 0.70]
PERCENTILES = [90, 95, 97.5]
N_SEEDS = 20


def similarities(X):
    return {"GMCC": gmcc_vectorized(X, X.mean()),
            "Pearson": pearson_vectorized(X),
            "Cosine": cosine_vectorized(X),
            "Spearman": spearman_vectorized(X)}


def graph(sim, idx, n, percentile=95):
    upper = sim[idx]
    edge = upper > np.percentile(upper, percentile)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(idx[0][edge], idx[1][edge]))
    return G, edge


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Saved: {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    unpert, pct_cc, mask_cc = {}, {}, {}
    gene_rows = []

    for ds, xf in DATASETS.items():
        X_full = np.load(os.path.join(DATA_DIR, xf))
        det_full = (X_full > 0).mean(axis=1)
        for gs, thr in GENE_SETS.items():
            keep = det_full >= thr
            X = X_full[keep]
            n, n_cells = X.shape
            det = det_full[keep]
            idx = np.triu_indices(n, k=1)
            B = (X > 0).astype(np.float64)
            co = (B @ B.T)[idx]
            expected = (np.outer(det, det) * n_cells)[idx]
            gene_rows.append([ds, gs, thr, n, len(idx[0]),
                              f"{(X == 0).mean() * 100:.2f}", f"{X.mean():.4f}"])
            print(f"\n{ds} / {gs}: {n} genes, mu = {X.mean():.4f}")

            S = similarities(X)
            for m in METHODS:
                G, edge = graph(S[m], idx, n)
                deg = np.array([d for _, d in G.degree()])
                unpert[(ds, gs, m)] = {
                    "C": nx.average_clustering(G),
                    "components": nx.number_connected_components(G),
                    "isolated": int((deg == 0).sum()),
                    "rho_det": spearmanr(deg, det)[0],
                    "no_codet": np.mean(co[edge] == 0) * 100,
                    "above_ind": np.mean(co[edge] > expected[edge]) * 100,
                }
                for p in PERCENTILES:
                    pct_cc[(ds, gs, m, p)] = fast_average_clustering(
                        S[m], idx, n, percentile=p)
                u = unpert[(ds, gs, m)]
                print(f"  {m:<9} C={u['C']:.4f} comp={u['components']:>5} "
                      f"rho(deg,det)={u['rho_det']:+.3f} "
                      f"no-codet={u['no_codet']:.1f}% "
                      f"above-ind={u['above_ind']:.1f}%")

            t0 = time.time()
            for rate in MASKING_LEVELS:
                vals = {m: [] for m in METHODS}
                for seed in range(1, N_SEEDS + 1):
                    Xm = apply_masking(X_full, rate, seed)[keep]
                    Sm = similarities(Xm)
                    for m in METHODS:
                        vals[m].append(fast_average_clustering(Sm[m], idx, n))
                for m in METHODS:
                    mask_cc[(ds, gs, m, rate)] = np.array(vals[m])
            print(f"  masking done ({time.time() - t0:.0f}s)")

    write_csv(os.path.join(OUT_DIR, "gene_sets.csv"),
              ["dataset", "gene_set", "min_detection", "n_genes", "n_pairs",
               "sparsity_pct", "global_mean"], gene_rows)

    # Table 3 + Table 4 + edge co-detection
    rows = []
    for gs in GENE_SETS:
        for m in METHODS:
            a, b = unpert[("PBMC3K", gs, m)], unpert[("Paul15", gs, m)]
            rows.append([gs, m, f"{a['C']:.4f}", f"{b['C']:.4f}",
                         f"{abs(a['C'] - b['C']):.4f}",
                         a["components"], b["components"],
                         a["isolated"], b["isolated"],
                         f"{a['rho_det']:.3f}", f"{b['rho_det']:.3f}",
                         f"{a['no_codet']:.2f}", f"{b['no_codet']:.2f}",
                         f"{a['above_ind']:.2f}", f"{b['above_ind']:.2f}"])
    write_csv(os.path.join(OUT_DIR, "table3_table4_unperturbed.csv"),
              ["gene_set", "estimator", "C_PBMC3K", "C_Paul15", "abs_diff_C",
               "components_PBMC3K", "components_Paul15", "isolated_PBMC3K",
               "isolated_Paul15", "rho_degree_detection_PBMC3K",
               "rho_degree_detection_Paul15", "pct_edges_no_codetection_PBMC3K",
               "pct_edges_no_codetection_Paul15",
               "pct_edges_above_independence_PBMC3K",
               "pct_edges_above_independence_Paul15"], rows)

    # S3: clustering coefficient under masking
    rows = []
    for gs in GENE_SETS:
        for m in METHODS:
            gaps = [abs(mask_cc[("PBMC3K", gs, m, r)].mean()
                        - mask_cc[("Paul15", gs, m, r)].mean())
                    for r in MASKING_LEVELS]
            rows.append([gs, m]
                        + [f"{mask_cc[(d, gs, m, r)].mean():.4f}"
                           for d in DATASETS for r in MASKING_LEVELS]
                        + [f"{g:.4f}" for g in gaps]
                        + [f"{min(gaps):.4f}", f"{max(gaps):.4f}"])
    write_csv(os.path.join(OUT_DIR, "tableS3_clustering_masking.csv"),
              ["gene_set", "estimator"]
              + [f"C_{d}_{int(r * 100)}" for d in DATASETS for r in MASKING_LEVELS]
              + [f"abs_diff_{int(r * 100)}" for r in MASKING_LEVELS]
              + ["min_abs_diff", "max_abs_diff"], rows)
    np.savez(os.path.join(OUT_DIR, "clustering_masking_per_seed.npz"),
             **{f"{d}|{g}|{m}|{r}": v for (d, g, m, r), v in mask_cc.items()})

    # S4: percentile sensitivity
    rows = []
    for gs in GENE_SETS:
        for m in METHODS:
            row = [gs, m]
            for p in PERCENTILES:
                a, b = pct_cc[("PBMC3K", gs, m, p)], pct_cc[("Paul15", gs, m, p)]
                row += [f"{a:.4f}", f"{b:.4f}", f"{abs(a - b):.4f}"]
            rows.append(row)
    write_csv(os.path.join(OUT_DIR, "tableS4_percentile.csv"),
              ["gene_set", "estimator"]
              + [f"{k}_p{p}" for p in PERCENTILES
                 for k in ("C_PBMC3K", "C_Paul15", "abs_diff")], rows)


if __name__ == "__main__":
    main()
