import scanpy as sc
import numpy as np
from scipy.stats import rankdata, kendalltau, wilcoxon, t as t_dist
from scipy import sparse
import time
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("MULTIPLE-SEED STABILITY: KENDALL TAU + NETWORK TOPOLOGY")
print("(20 seed per dropout level, Wilcoxon Signed-Rank)")
print("=" * 65)

save_dir = ("data/processed")
N_SEEDS = 20
DROPOUT_LEVELS = [0.1, 0.3, 0.5, 0.7]
PERCENTILE = 95  # konsisten dengan Stage 5

# ── FUNGSI SIMILARITY (TIDAK BERUBAH, sudah teruji) ─────────────
def gmcc_vectorized(X, mu):
    X_c   = X - mu
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_c/norms) @ (X_c/norms).T, -1.0, 1.0)

def pearson_vectorized(X):
    X_c   = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_c/norms) @ (X_c/norms).T, -1.0, 1.0)

def cosine_vectorized(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X/norms) @ (X/norms).T, -1.0, 1.0)

def spearman_vectorized(X):
    X_r = np.apply_along_axis(rankdata, 1, X)
    return pearson_vectorized(X_r)

def bicor_vectorized(X):
    med     = np.median(X, axis=1, keepdims=True)
    mad     = np.median(np.abs(X - med), axis=1, keepdims=True)
    std_x   = np.std(X, axis=1, keepdims=True)
    mad_reg = np.where(mad < 1e-8, std_x, mad)
    mad_reg[mad_reg < 1e-10] = 1e-10
    u       = (X - med) / (9.0 * mad_reg)
    w       = np.where(np.abs(u) < 1, (1 - u**2)**2, 0.0)
    X_t     = w * (X - med)
    norms   = np.linalg.norm(X_t, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_t/norms) @ (X_t/norms).T, -1.0, 1.0)

def apply_dropout(X, rate, seed):
    rng          = np.random.default_rng(seed)
    result       = X.copy()
    nonzero_mask = X > 0
    dropout_mask = rng.random(X.shape) < rate
    result[nonzero_mask & dropout_mask] = 0
    return result

# ── clustering coefficient tervektorisasi ────────────────────────
# Diverifikasi identik dengan nx.average_clustering() (selisih ~1e-18)
# tapi ~5.7x lebih cepat menggunakan operasi sparse matrix.
def fast_average_clustering(sim_matrix, idx, N_GENES,
                             percentile=PERCENTILE):
    upper     = sim_matrix[idx]
    threshold = np.percentile(upper, percentile)
    mask      = upper > threshold

    rows, cols = idx[0][mask], idx[1][mask]
    if len(rows) == 0:
        return 0.0

    data     = np.ones(len(rows) * 2)
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    A        = sparse.csr_matrix(
        (data, (all_rows, all_cols)), shape=(N_GENES, N_GENES)
    )

    degree = np.array(A.sum(axis=1)).flatten()
    A2     = A @ A
    triangles = np.array((A2.multiply(A)).sum(axis=1)).flatten() / 2

    denom = degree * (degree - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        local_clustering = np.where(denom > 0,
                                     2 * triangles / denom, 0.0)
    return local_clustering.mean()

def compute_ci(values, confidence=0.95):
    values = np.array(values)
    n      = len(values)
    mean   = values.mean()
    sd     = values.std(ddof=1)
    se     = sd / np.sqrt(n)
    t_crit = t_dist.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se
    return mean, sd, mean - margin, mean + margin

def significance_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

# ── FUNGSI UTAMA: MULTI-SEED (tau + clustering dalam loop sama) ─
def run_multiseed_analysis(X, global_mean, dataset_name):
    print(f"\n{'='*65}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*65}")

    N_GENES = X.shape[0]
    idx     = np.triu_indices(N_GENES, k=1)

    methods = {
        "GMCC"    : lambda X: gmcc_vectorized(X, global_mean),
        "Pearson" : pearson_vectorized,
        "Cosine"  : cosine_vectorized,
        "Spearman": spearman_vectorized,
        "Bicor"   : bicor_vectorized,
    }

    print(f"\n  Menghitung similarity original...", end="",
          flush=True)
    sim_original = {name: func(X) for name, func in methods.items()}
    print(" ✓")

    results_tau        = {name: {r: [] for r in DROPOUT_LEVELS}
                           for name in methods}
    results_clustering  = {name: {r: [] for r in DROPOUT_LEVELS}
                           for name in methods}

    for rate in DROPOUT_LEVELS:
        print(f"\n  Dropout {rate*100:.0f}% "
              f"({N_SEEDS} seed)...", end="", flush=True)
        t0 = time.time()

        for seed in range(1, N_SEEDS + 1):
            X_drop = apply_dropout(X, rate, seed=seed)
            for name, func in methods.items():
                sim_drop = func(X_drop)

                # ── Metrik 1: Kendall Tau ──
                s_orig = sim_original[name][idx]
                s_drop = sim_drop[idx]
                tau, _ = kendalltau(s_orig, s_drop)
                tau    = tau if not np.isnan(tau) else 0.0
                results_tau[name][rate].append(tau)

                # ── Metrik 2: Clustering coefficient ──
                # Disisipkan di sini — sim_drop masih tersedia,
                # belum ditimpa iterasi berikutnya.
                clust = fast_average_clustering(
                    sim_drop, idx, N_GENES
                )
                results_clustering[name][rate].append(clust)

        print(f" ✓ ({time.time()-t0:.1f}s)")

    return results_tau, results_clustering, methods

# ── RINGKASAN + UJI STATISTIK (generik untuk tau ATAU clustering) ─
def summarize_and_test(results, methods, dataset_name, metric_name):
    print(f"\n{'-'*65}")
    print(f"RINGKASAN STATISTIK [{metric_name}] — {dataset_name}")
    print(f"{'-'*65}")

    print(f"\n  {metric_name}: mean ± SD [95% CI], n={N_SEEDS} seed\n")
    for rate in DROPOUT_LEVELS:
        print(f"  Dropout {rate*100:.0f}%:")
        for name in methods:
            vals = results[name][rate]
            mean, sd, lo, hi = compute_ci(vals)
            marker = " ←" if name == "GMCC" else ""
            print(f"    {name:<10}: {mean:.4f} ± {sd:.4f} "
                  f"[{lo:.4f}, {hi:.4f}]{marker}")
        print()

    print(f"  {'-'*61}")
    print(f"  Uji Wilcoxon Signed-Rank (GMCC vs pembanding) "
          f"— {metric_name}")
    print(f"  {'-'*61}")

    for rate in DROPOUT_LEVELS:
        print(f"\n  Dropout {rate*100:.0f}%:")
        gmcc_vals = results["GMCC"][rate]
        for name in methods:
            if name == "GMCC":
                continue
            other_vals = results[name][rate]
            try:
                stat, p = wilcoxon(gmcc_vals, other_vals)
            except ValueError:
                stat, p = np.nan, 1.0
            sig         = significance_stars(p)
            median_diff = np.median(
                np.array(gmcc_vals) - np.array(other_vals)
            )
            direction = "GMCC lebih tinggi" if median_diff > 0 \
                        else "GMCC lebih rendah"
            print(f"    vs {name:<10}: W={stat:>7.1f}  "
                  f"p={p:.2e} {sig:<3}  "
                  f"Δmedian={median_diff:+.4f} ({direction})")

    print(f"\n  {'-'*61}")
    print(f"  Uji Pooled (semua dropout digabung, n={N_SEEDS*4}) "
          f"— {metric_name}")
    print(f"  {'-'*61}\n")

    gmcc_pooled = np.concatenate(
        [results["GMCC"][r] for r in DROPOUT_LEVELS]
    )
    for name in methods:
        if name == "GMCC":
            continue
        other_pooled = np.concatenate(
            [results[name][r] for r in DROPOUT_LEVELS]
        )
        stat, p = wilcoxon(gmcc_pooled, other_pooled)
        sig     = significance_stars(p)
        median_diff = np.median(gmcc_pooled - other_pooled)
        print(f"    GMCC vs {name:<10}: W={stat:>8.1f}  "
              f"p={p:.2e} {sig:<3}  Δmedian={median_diff:+.4f}")

# ── EKSEKUSI: DATASET 1 — PBMC 3K ───────────────────────────────
X_pbmc  = np.load(f"{save_dir}/X_2000hvg.npy")
mu_pbmc = np.load(f"{save_dir}/global_mean_2000hvg.npy")[0]

tau_pbmc, clust_pbmc, methods_pbmc = run_multiseed_analysis(
    X_pbmc, mu_pbmc, "PBMC 3K (2000 HVG)"
)
summarize_and_test(tau_pbmc, methods_pbmc, "PBMC 3K", "Kendall Tau")
summarize_and_test(clust_pbmc, methods_pbmc, "PBMC 3K",
                    "Clustering Coefficient")

# ── EKSEKUSI: DATASET 2 — PAUL ET AL. 2015 ──────────────────────
X_paul  = np.load(f"{save_dir}/X_paul15.npy")
mu_paul = np.load(f"{save_dir}/global_mean_paul15.npy")[0]

tau_paul, clust_paul, methods_paul = run_multiseed_analysis(
    X_paul, mu_paul, "Paul et al. 2015 (Mouse Bone Marrow)"
)
summarize_and_test(tau_paul, methods_paul, "Paul et al.", "Kendall Tau")
summarize_and_test(clust_paul, methods_paul, "Paul et al.",
                    "Clustering Coefficient")

# ── SIMPAN HASIL ──────────────────────────────────────────────────
np.save(f"{save_dir}/multiseed_tau_pbmc.npy", tau_pbmc)
np.save(f"{save_dir}/multiseed_clustering_pbmc.npy", clust_pbmc)
np.save(f"{save_dir}/multiseed_tau_paul15.npy", tau_paul)
np.save(f"{save_dir}/multiseed_clustering_paul15.npy", clust_paul)

print("\n" + "=" * 65)
print("SELESAI — Hasil disimpan di data/processed/")
print("=" * 65)
print("\nSignifikansi: *** p<0.001, ** p<0.01, * p<0.05, ns=tidak signifikan")
