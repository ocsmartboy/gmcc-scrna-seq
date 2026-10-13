import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import rankdata, kendalltau, norm, wilcoxon

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SAVE_DIR = os.path.join(REPO, "data", "processed")

N_SEEDS = 20
MASKING_LEVELS = [0.10, 0.30, 0.50, 0.70]
COMPARATORS = ["Pearson", "Cosine", "Spearman", "Bicor-SD"]


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
    np.random.seed(seed)
    result = matrix.copy()
    nonzero_mask = matrix > 0
    mask = np.random.rand(*matrix.shape) < rate
    result[nonzero_mask & mask] = 0
    return result


# ── Statistics ───────────────────────────────────────────────────────
def hodges_lehmann_ci(d, alpha=0.05):
    """Hodges-Lehmann estimate (median of Walsh averages) with a
    distribution-free CI. The index k uses the normal approximation to the
    signed-rank null distribution."""
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
    """Holm-adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.zeros(m)
    running = 0.0
    for rank, idx_ in enumerate(order):
        running = max(running, (m - rank) * p[idx_])
        adj[idx_] = min(running, 1.0)
    return adj


def load(name):
    path = os.path.join(SAVE_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found.")
    return np.load(path)


def main():
    print("=" * 72)
    print("PAIRED DIFFERENCE + CONFIDENCE INTERVAL (Hodges-Lehmann)")
    print("Protocol: global mean RECOMPUTED from each masked matrix")
    print("=" * 72)

    datasets = {
        "PBMC3K": (load("X_2000hvg.npy"), load("global_mean_2000hvg.npy")[0]),
        "Paul15": (load("X_paul15.npy"),  load("global_mean_paul15.npy")[0]),
    }

    all_tau = {}

    for ds_name, (X, mu0) in datasets.items():
        print(f"\n{'=' * 72}\nDATASET: {ds_name}   "
              f"(mu of unperturbed matrix = {mu0:.6f})\n{'=' * 72}")
        idx = np.triu_indices(X.shape[0], k=1)

        # Reference similarities from the unperturbed matrix.
        print("  Computing reference similarities...", end="", flush=True)
        sim_original = {
            "GMCC":     gmcc_vectorized(X, mu0)[idx],
            "Pearson":  pearson_vectorized(X)[idx],
            "Cosine":   cosine_vectorized(X)[idx],
            "Spearman": spearman_vectorized(X)[idx],
            "Bicor-SD": bicor_sd_vectorized(X)[idx],
        }
        print(" done")

        for rate in MASKING_LEVELS:
            print(f"  Masking {rate * 100:.0f}% ({N_SEEDS} seeds)...",
                  end="", flush=True)
            t0 = time.time()
            taus = {k: [] for k in sim_original}
            mus = []

            for seed in range(1, N_SEEDS + 1):
                Xm = apply_masking(X, rate, seed)

                # CHANGED: mu is recomputed from the masked matrix.
                mu_m = Xm.mean()
                mus.append(mu_m)

                perturbed = {
                    "GMCC":     gmcc_vectorized(Xm, mu_m),
                    "Pearson":  pearson_vectorized(Xm),
                    "Cosine":   cosine_vectorized(Xm),
                    "Spearman": spearman_vectorized(Xm),
                    "Bicor-SD": bicor_sd_vectorized(Xm),
                }
                for name, S in perturbed.items():
                    tau, _ = kendalltau(sim_original[name], S[idx])
                    taus[name].append(tau if not np.isnan(tau) else 0.0)

            for name in sim_original:
                all_tau[(ds_name, name, rate)] = np.array(taus[name])
            print(f" done ({time.time() - t0:.1f}s), "
                  f"mean recomputed mu = {np.mean(mus):.6f}")

    # ── Save raw values (consumed by make_figures.py) ────────────────
    flat = {f"{ds}|{m}|{r}": v for (ds, m, r), v in all_tau.items()}
    np.savez(os.path.join(SAVE_DIR, "raw_kendall_tau_per_seed.npz"), **flat)
    print(f"\nSaved: {os.path.join(SAVE_DIR, 'raw_kendall_tau_per_seed.npz')}")

    # ── Table 1 equivalent: average tau and degradation ──────────────
    for ds_name in datasets:
        print(f"\n{'=' * 72}\nAVERAGE KENDALL'S TAU AND DEGRADATION - {ds_name}"
              f"\n{'=' * 72}")
        print(f"\n  {'Method':<12}{'Avg tau':>10}{'Degradation':>14}")
        print("  " + "-" * 36)
        for name in ["GMCC"] + COMPARATORS:
            vals = [all_tau[(ds_name, name, r)].mean() for r in MASKING_LEVELS]
            avg = np.mean(vals)
            deg = (vals[0] - vals[-1]) / vals[0] * 100
            print(f"  {name:<12}{avg:>10.4f}{deg:>13.1f}%")

    # ── Table 2 equivalent: HL + CI + Holm-adjusted Wilcoxon ─────────
    for ds_name in datasets:
        print(f"\n{'=' * 72}\nHODGES-LEHMANN + 95% CI - {ds_name}\n{'=' * 72}")
        for rate in MASKING_LEVELS:
            raw_p = []
            rows = []
            for comp in COMPARATORS:
                d = (all_tau[(ds_name, "GMCC", rate)]
                     - all_tau[(ds_name, comp, rate)])
                hl, lo, hi = hodges_lehmann_ci(d)
                stat, p = wilcoxon(d)
                raw_p.append(p)
                rows.append((comp, hl, lo, hi, stat))
            adj_p = list(holm(raw_p[:3])) + [raw_p[3]]

            print(f"\n  Masking {rate * 100:.0f}%:")
            print(f"    {'Comparison':<16}{'HL':>10}{'95% CI':>24}"
                  f"{'W':>8}{'p (Holm)':>13}")
            print("    " + "-" * 71)
            for (comp, hl, lo, hi, stat), pa in zip(rows, adj_p):
                print(f"    vs {comp:<13}{hl:>+10.4f}   "
                      f"[{lo:+.4f}, {hi:+.4f}]{stat:>8.1f}{pa:>13.2e}")

    print("\nDone. Regenerate Figures 1 and 2 with:")
    print("    python src/figures/make_figures.py")


if __name__ == "__main__":
    main()
