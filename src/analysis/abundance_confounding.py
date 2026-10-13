import os
import sys

import numpy as np
import networkx as nx
from scipy.stats import rankdata, spearmanr

# ── Paths resolved from this file's location ─────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SAVE_DIR = os.path.join(REPO, "data", "processed")

PERCENTILE = 95


# ── Similarity measures (identical to the main experiments) ──────────
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


def degrees_at_percentile(sim, n_genes, percentile=PERCENTILE):
    """Build the percentile-thresholded network and return node degrees."""
    idx = np.triu_indices(n_genes, k=1)
    upper = sim[idx]
    threshold = np.percentile(upper, percentile)
    G = nx.Graph()
    G.add_nodes_from(range(n_genes))
    mask = upper > threshold
    G.add_edges_from(zip(idx[0][mask], idx[1][mask]))
    return np.array([G.degree(i) for i in range(n_genes)])


def load(name):
    path = os.path.join(SAVE_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run the preprocessing stage first.")
    return np.load(path)


def main():
    datasets = {
        "PBMC 3K": (load("X_2000hvg.npy"), load("global_mean_2000hvg.npy")[0]),
        "Paul15":  (load("X_paul15.npy"),  load("global_mean_paul15.npy")[0]),
    }

    print("=" * 74)
    print("ABUNDANCE-CONFOUNDING DIAGNOSTIC")
    print("=" * 74)

    for ds_name, (X, mu) in datasets.items():
        n_genes = X.shape[0]
        detection_rate = (X > 0).mean(axis=1)   # fraction of cells with non-zero
        mean_expr = X.mean(axis=1)
        gene_var = X.var(axis=1)

        methods = {
            "GMCC":     lambda M: gmcc_vectorized(M, mu),
            "Pearson":  pearson_vectorized,
            "Cosine":   cosine_vectorized,
            "Spearman": spearman_vectorized,
            "Bicor-SD": bicor_sd_vectorized,
        }

        print(f"\n{ds_name}  ({n_genes} genes x {X.shape[1]} cells, "
              f"mu = {mu:.4f})")
        print(f"  detection rate: min {detection_rate.min():.3f}, "
              f"median {np.median(detection_rate):.3f}, "
              f"max {detection_rate.max():.3f}")
        print()
        print(f"  {'Method':<10}{'rho(deg, detect)':>19}"
              f"{'rho(deg, mean)':>17}{'rho(deg, var)':>16}{'isolated':>11}")
        print("  " + "-" * 71)

        for m_name, func in methods.items():
            sim = func(X)
            deg = degrees_at_percentile(sim, n_genes)
            r_det, _ = spearmanr(deg, detection_rate)
            r_mean, _ = spearmanr(deg, mean_expr)
            r_var, _ = spearmanr(deg, gene_var)
            n_iso = int((deg == 0).sum())
            print(f"  {m_name:<10}{r_det:>19.4f}{r_mean:>17.4f}"
                  f"{r_var:>16.4f}{n_iso:>11}")

        # Block comparison: do rarely-detected genes appear mutually similar?
        q_lo, q_hi = np.quantile(detection_rate, [0.25, 0.75])
        lo = detection_rate <= q_lo
        hi = detection_rate >= q_hi
        print()
        print(f"  Mean similarity by detection-rate group "
              f"(low <= {q_lo:.3f}, high >= {q_hi:.3f}):")
        print(f"  {'Method':<10}{'low-low':>12}{'high-high':>12}{'low-high':>12}")
        print("  " + "-" * 46)
        for m_name, func in methods.items():
            S = func(X)
            ll = S[np.ix_(lo, lo)][np.triu_indices(int(lo.sum()), k=1)].mean()
            hh = S[np.ix_(hi, hi)][np.triu_indices(int(hi.sum()), k=1)].mean()
            lh = S[np.ix_(lo, hi)].mean()
            print(f"  {m_name:<10}{ll:>12.4f}{hh:>12.4f}{lh:>12.4f}")

    print("\n" + "=" * 74)
    print("Guide: |rho| < 0.3 weak | 0.3-0.5 moderate | > 0.5 strong.")
    print("Compare GMCC against Pearson, which centers each gene on its own")
    print("mean and is therefore the abundance-insensitive reference.")
    print("=" * 74)


if __name__ == "__main__":
    main()