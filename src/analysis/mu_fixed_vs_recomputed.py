import os
import sys
import time

import numpy as np
from scipy.stats import kendalltau, norm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SAVE_DIR = os.path.join(REPO, "data", "processed")

N_SEEDS = 20
MASKING_LEVELS = [0.10, 0.30, 0.50, 0.70]


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


def apply_masking(matrix, rate, seed):
    """Identical to the primary experiment: mask a proportion of detected
    entries to zero."""
    np.random.seed(seed)
    result = matrix.copy()
    nonzero_mask = matrix > 0
    mask = np.random.rand(*matrix.shape) < rate
    result[nonzero_mask & mask] = 0
    return result


def hodges_lehmann(d):
    """Median of the Walsh averages (same as paired_difference_ci.py)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    i, j = np.triu_indices(n, k=0)
    return np.median(np.sort((d[i] + d[j]) / 2.0))


def load(name):
    path = os.path.join(SAVE_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found.")
    return np.load(path)


def main():
    print("=" * 72)
    print("GLOBAL MEAN: FIXED vs RECOMPUTED (sensitivity analysis)")
    print("=" * 72)

    datasets = {
        "PBMC 3K": (load("X_2000hvg.npy"), load("global_mean_2000hvg.npy")[0]),
        "Paul15":  (load("X_paul15.npy"),  load("global_mean_paul15.npy")[0]),
    }

    all_tau = {}

    for ds_name, (X, mu0) in datasets.items():
        print(f"\n{'=' * 72}\nDATASET: {ds_name}   (mu of unperturbed matrix = "
              f"{mu0:.6f})\n{'=' * 72}")
        idx = np.triu_indices(X.shape[0], k=1)

        # Reference similarity from the unperturbed matrix.
        # Both GMCC variants share the same reference, since on unperturbed
        # data the recomputed mu equals mu0 by definition.
        ref = {
            "GMCC-fixed":      gmcc_vectorized(X, mu0)[idx],
            "GMCC-recomputed": gmcc_vectorized(X, mu0)[idx],
            "Pearson":         pearson_vectorized(X)[idx],
            "Cosine":          cosine_vectorized(X)[idx],
        }

        for rate in MASKING_LEVELS:
            print(f"  Masking {rate * 100:.0f}% ({N_SEEDS} seeds)...",
                  end="", flush=True)
            t0 = time.time()
            taus = {k: [] for k in ref}
            mus_recomputed = []

            for seed in range(1, N_SEEDS + 1):
                Xm = apply_masking(X, rate, seed)
                mu_new = Xm.mean()
                mus_recomputed.append(mu_new)

                variants = {
                    "GMCC-fixed":      gmcc_vectorized(Xm, mu0),
                    "GMCC-recomputed": gmcc_vectorized(Xm, mu_new),
                    "Pearson":         pearson_vectorized(Xm),
                    "Cosine":          cosine_vectorized(Xm),
                }
                for name, S in variants.items():
                    tau, _ = kendalltau(ref[name], S[idx])
                    taus[name].append(tau if not np.isnan(tau) else 0.0)

            for name in ref:
                all_tau[(ds_name, name, rate)] = np.array(taus[name])
            print(f" done ({time.time() - t0:.1f}s), "
                  f"mean recomputed mu = {np.mean(mus_recomputed):.6f}")

        # ── Per-level summary ────────────────────────────────────────
        print(f"\n  Kendall's Tau, mean +/- SD across {N_SEEDS} seeds:\n")
        print(f"  {'Method':<18}" +
              "".join(f"{int(r * 100):>16}%" for r in MASKING_LEVELS))
        print("  " + "-" * 82)
        for name in ["GMCC-fixed", "GMCC-recomputed", "Pearson", "Cosine"]:
            row = f"  {name:<18}"
            for r in MASKING_LEVELS:
                v = all_tau[(ds_name, name, r)]
                row += f"{v.mean():>10.4f}+-{v.std(ddof=1):.4f}"
            print(row)

        # ── Paired difference: fixed minus recomputed ────────────────
        print(f"\n  Hodges-Lehmann paired difference "
              f"(GMCC-fixed minus GMCC-recomputed):\n")
        print(f"  {'Masking level':<18}{'HL difference':>16}{'Direction':>28}")
        print("  " + "-" * 62)
        for r in MASKING_LEVELS:
            d = (all_tau[(ds_name, "GMCC-fixed", r)]
                 - all_tau[(ds_name, "GMCC-recomputed", r)])
            hl = hodges_lehmann(d)
            if abs(hl) < 0.005:
                direction = "negligible"
            elif hl > 0:
                direction = "fixed reference helps GMCC"
            else:
                direction = "fixed reference hurts GMCC"
            print(f"  {int(r * 100):>3}%{'':<14}{hl:>+16.4f}{direction:>28}")

    print("\n" + "=" * 72)
    print("Reading the result:")
    print("  |HL difference| < 0.005  -> the fixed reference makes no")
    print("     practical difference; GMCC's stability is attributable to")
    print("     global centering, not to holding mu fixed.")
    print("  HL difference clearly positive -> the fixed reference")
    print("     contributes to the reported stability and must be stated.")
    print("=" * 72)


if __name__ == "__main__":
    main()