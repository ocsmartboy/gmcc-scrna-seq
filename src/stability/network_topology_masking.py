import os
import sys
import time

import numpy as np
from scipy.stats import rankdata
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SAVE_DIR = os.path.join(REPO, "data", "processed")

N_SEEDS = 20
MASKING_LEVELS = [0.10, 0.30, 0.50, 0.70]
PERCENTILE = 95
METHODS = ["GMCC", "Pearson", "Cosine", "Spearman", "Bicor-SD"]


# ── Similarity measures ──────────────────────────────────────────────
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


def apply_masking(matrix, rate, seed):
    """Same RNG call as paired_difference_ci.py, so the 20 realizations
    per level are identical across the two scripts."""
    np.random.seed(seed)
    result = matrix.copy()
    nonzero_mask = matrix > 0
    mask = np.random.rand(*matrix.shape) < rate
    result[nonzero_mask & mask] = 0
    return result


def fast_average_clustering(sim_matrix, idx, n_genes, percentile=PERCENTILE):
    """Sparse-matrix average clustering coefficient. Verified against
    networkx.average_clustering (agreement to ~1e-18) and substantially
    faster."""
    upper = sim_matrix[idx]
    threshold = np.percentile(upper, percentile)
    mask = upper > threshold

    rows, cols = idx[0][mask], idx[1][mask]
    if len(rows) == 0:
        return 0.0

    data = np.ones(len(rows) * 2)
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    A = sparse.csr_matrix((data, (all_rows, all_cols)),
                          shape=(n_genes, n_genes))

    degree = np.array(A.sum(axis=1)).flatten()
    A2 = A @ A
    triangles = np.array((A2.multiply(A)).sum(axis=1)).flatten() / 2

    denom = degree * (degree - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        local = np.where(denom > 0, 2 * triangles / denom, 0.0)
    return local.mean()


def load(name):
    path = os.path.join(SAVE_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found.")
    return np.load(path)


def run_dataset(X, label):
    n_genes = X.shape[0]
    idx = np.triu_indices(n_genes, k=1)
    results = {m: {r: [] for r in MASKING_LEVELS} for m in METHODS}

    print(f"\n{label}  ({n_genes} genes x {X.shape[1]} cells)")
    for rate in MASKING_LEVELS:
        print(f"  Masking {rate * 100:.0f}% ({N_SEEDS} seeds)...",
              end="", flush=True)
        t0 = time.time()
        for seed in range(1, N_SEEDS + 1):
            Xm = apply_masking(X, rate, seed)
            mu_m = Xm.mean()          # CHANGED: recomputed per realization

            sims = {
                "GMCC":     gmcc_vectorized(Xm, mu_m),
                "Pearson":  pearson_vectorized(Xm),
                "Cosine":   cosine_vectorized(Xm),
                "Spearman": spearman_vectorized(Xm),
                "Bicor-SD": bicor_sd_vectorized(Xm),
            }
            for m, S in sims.items():
                results[m][rate].append(
                    fast_average_clustering(S, idx, n_genes))
        print(f" done ({time.time() - t0:.1f}s)")
    return results


def main():
    print("=" * 72)
    print("NETWORK CLUSTERING COEFFICIENT UNDER CONTROLLED MASKING")
    print("Protocol: global mean RECOMPUTED from each masked matrix")
    print("=" * 72)

    res = {
        "PBMC 3K": run_dataset(load("X_2000hvg.npy"), "PBMC 3K"),
        "Paul15":  run_dataset(load("X_paul15.npy"),  "Paul15"),
    }

    # ── Per-level means and SDs ──────────────────────────────────────
    for ds, r in res.items():
        print(f"\n{'=' * 72}\nCLUSTERING COEFFICIENT - {ds}\n{'=' * 72}")
        print(f"\n  {'Method':<12}" +
              "".join(f"{int(v * 100):>16}%" for v in MASKING_LEVELS))
        print("  " + "-" * 76)
        for m in METHODS:
            row = f"  {m:<12}"
            for rate in MASKING_LEVELS:
                v = np.array(r[m][rate])
                row += f"{v.mean():>10.4f}+-{v.std(ddof=1):.4f}"
            print(row)

    # ── Table 4: cross-dataset gap per level, then min/max ───────────
    print(f"\n{'=' * 72}")
    print("TABLE 4: CROSS-DATASET GAP IN MEAN CLUSTERING COEFFICIENT")
    print("=" * 72)
    print(f"\n  {'Method':<12}" +
          "".join(f"{int(v * 100):>11}%" for v in MASKING_LEVELS) +
          f"{'Min |D|':>11}{'Max |D|':>11}")
    print("  " + "-" * 78)
    for m in METHODS:
        gaps = []
        for rate in MASKING_LEVELS:
            a = np.mean(res["PBMC 3K"][m][rate])
            b = np.mean(res["Paul15"][m][rate])
            gaps.append(abs(a - b))
        row = f"  {m:<12}" + "".join(f"{g:>12.4f}" for g in gaps)
        row += f"{min(gaps):>11.4f}{max(gaps):>11.4f}"
        print(row)

    # ── Max SD for GMCC (cited in the manuscript) ────────────────────
    sds = [np.array(res[ds]["GMCC"][r]).std(ddof=1)
           for ds in res for r in MASKING_LEVELS]
    print(f"\n  Largest GMCC standard deviation across all conditions: "
          f"{max(sds):.4f}")

    np.savez(os.path.join(SAVE_DIR, "clustering_coefficient_masking.npz"),
             **{f"{ds}|{m}|{r}": np.array(res[ds][m][r])
                for ds in res for m in METHODS for r in MASKING_LEVELS})
    print(f"\nSaved: {os.path.join(SAVE_DIR, 'clustering_coefficient_masking.npz')}")


if __name__ == "__main__":
    main()
