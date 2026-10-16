
import argparse
import os
import sys

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SAVE_DIR = os.path.join(REPO, "data", "processed")

N_SEEDS = 20
MASKING_LEVELS = [0.10, 0.30, 0.50, 0.70]
PERCENTILES = [90, 95, 97.5]
ESTIMATORS = ["GMCC", "Pearson", "Cosine", "Spearman"]


# ── Identical to paired_difference_ci.py ─────────────────────────────
def _gram(Xc):
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    Z = Xc / norms
    return np.clip(Z @ Z.T, -1.0, 1.0)


def similarity(name, X, mu=None):
    if name == "GMCC":
        return _gram(X - mu)
    if name == "Pearson":
        return _gram(X - X.mean(axis=1, keepdims=True))
    if name == "Cosine":
        return _gram(X)
    if name == "Spearman":
        R = np.apply_along_axis(rankdata, 1, X)
        return _gram(R - R.mean(axis=1, keepdims=True))
    raise ValueError(name)


def apply_masking(matrix, rate, seed):
    np.random.seed(seed)
    result = matrix.copy()
    nonzero_mask = matrix > 0
    mask = np.random.rand(*matrix.shape) < rate
    result[nonzero_mask & mask] = 0
    return result


def load(name):
    path = os.path.join(SAVE_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found. Run the preprocessing scripts first.")
    return np.load(path)


def edge_stats(s, percentile):
    thr = np.percentile(s, percentile)
    n_edges = int((s > thr).sum())
    n_ties = int((s == thr).sum())
    return thr, n_edges, n_ties


def check_dataset(label, X, mu0, masked_networks):
    n_genes, n_cells = X.shape
    idx = np.triu_indices(n_genes, k=1)
    n_pairs = len(idx[0])
    nonzero = X > 0
    n_nonzero = int(nonzero.sum())

    print(f"\n{'=' * 74}\n{label}: {n_genes} genes x {n_cells} cells, "
          f"mu0 = {mu0:.6f}, gene pairs = {n_pairs:,}\n{'=' * 74}")

    # 4. Unperturbed matrix
    print("\n[Unperturbed matrix]")
    print(f"  all-zero gene profiles        : {int((~nonzero).all(axis=1).sum())}")
    print(f"  zero-variance gene profiles   : {int((X.std(axis=1) == 0).sum())}")
    print(f"  GMCC zero-norm profiles (x-mu): "
          f"{int((np.linalg.norm(X - mu0, axis=1) == 0).sum())}")

    # 1-3. Masking
    print("\n[Controlled masking]")
    print(f"  {'Level':<7}{'realised mean':>14}{'min':>9}{'max':>9}"
          f"{'max const. genes':>18}{'max all-zero genes':>20}")
    masks_prev = None
    nested_ok = True
    for rate in MASKING_LEVELS:
        realised, n_const, n_allzero, masks = [], [], [], []
        for seed in range(1, N_SEEDS + 1):
            Xm = apply_masking(X, rate, seed)
            masked = nonzero & (Xm == 0)
            masks.append(masked)
            realised.append(masked.sum() / n_nonzero)
            n_const.append(int((Xm.std(axis=1) == 0).sum()))
            n_allzero.append(int((Xm.max(axis=1) == 0).sum()))
        if masks_prev is not None:
            for a, b in zip(masks_prev, masks):
                if np.any(a & ~b):
                    nested_ok = False
        masks_prev = masks
        print(f"  {int(rate * 100):>3}%   {np.mean(realised):>14.6f}"
              f"{np.min(realised):>9.4f}{np.max(realised):>9.4f}"
              f"{max(n_const):>18}{max(n_allzero):>20}")
    print(f"  Masking levels nested within each seed: {nested_ok}")

    # 5-6. Baseline similarities
    print("\n[Unperturbed similarity values]")
    print(f"  {'Estimator':<10}{'NaN':>6}{'MAD(s0)':>11}"
          + "".join(f"{'edges P' + str(p):>15}{'ties':>7}" for p in PERCENTILES))
    for name in ESTIMATORS:
        s0 = similarity(name, X, mu0)[idx]
        mad = np.median(np.abs(s0 - np.median(s0)))
        row = f"  {name:<10}{int(np.isnan(s0).sum()):>6}{mad:>11.6f}"
        for p in PERCENTILES:
            _, n_e, n_t = edge_stats(s0, p)
            row += f"{n_e:>15,}{n_t:>7}"
        print(row)
    print("  Expected edges with no ties: "
          + ", ".join(f"P{p}: {int(round(n_pairs * (100 - p) / 100)):,}"
                      for p in PERCENTILES))

    # 7. Optional: masked networks
    if masked_networks:
        print("\n[Masked networks, 95th percentile, recomputed mu]")
        print(f"  {'Estimator':<10}{'Level':>7}{'min edges':>12}"
              f"{'max edges':>12}{'max ties':>10}{'NaN':>6}")
        for rate in MASKING_LEVELS:
            stats = {n: [] for n in ESTIMATORS}
            for seed in range(1, N_SEEDS + 1):
                Xm = apply_masking(X, rate, seed)
                for name in ESTIMATORS:
                    s = similarity(name, Xm, Xm.mean())[idx]
                    _, n_e, n_t = edge_stats(s, 95)
                    stats[name].append((n_e, n_t, int(np.isnan(s).sum())))
            for name in ESTIMATORS:
                a = np.array(stats[name])
                print(f"  {name:<10}{int(rate * 100):>6}%{a[:, 0].min():>12,}"
                      f"{a[:, 0].max():>12,}{a[:, 1].max():>10}{a[:, 2].max():>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked", action="store_true",
                    help="also check edges/ties in every masked network (slow)")
    ap.add_argument("--data-dir", default=None,
                    help="folder containing X_2000hvg.npy etc. "
                         "(default: <repo>/data/processed)")
    args = ap.parse_args()
    if args.data_dir:
        global SAVE_DIR
        SAVE_DIR = args.data_dir
    print(f"Data folder: {SAVE_DIR}")

    check_dataset("PBMC 3K", load("X_2000hvg.npy"),
                  float(load("global_mean_2000hvg.npy")[0]), args.masked)
    check_dataset("Paul15", load("X_paul15.npy"),
                  float(load("global_mean_paul15.npy")[0]), args.masked)


if __name__ == "__main__":
    main()