import numpy as np
import scanpy as sc
from scipy.stats import rankdata, kendalltau, norm
from scipy.sparse import issparse
import time
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("PAIRED DIFFERENCE + CONFIDENCE INTERVAL (Hodges-Lehmann)")
print("=" * 65)

save_dir = "d:/S3/Eksperimen/Bioinformatika/gmcc_biomedical/data/processed"
N_SEEDS = 20
DROPOUT_LEVELS = [0.10, 0.30, 0.50, 0.70]

# ── Fungsi similarity (identik dengan eksperimen sebelumnya) ──
def gmcc_vectorized(X, mu):
    X_c = X - mu
    norms = np.linalg.norm(X_c, axis=1, keepdims=True); norms[norms == 0] = 1e-10
    return np.clip((X_c/norms) @ (X_c/norms).T, -1.0, 1.0)

def pearson_vectorized(X):
    X_c = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True); norms[norms == 0] = 1e-10
    return np.clip((X_c/norms) @ (X_c/norms).T, -1.0, 1.0)

def cosine_vectorized(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1e-10
    return np.clip((X/norms) @ (X/norms).T, -1.0, 1.0)

def spearman_vectorized(X):
    return pearson_vectorized(np.apply_along_axis(rankdata, 1, X))

def bicor_sd_vectorized(X):
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True)
    std_x = np.std(X, axis=1, keepdims=True)
    mad_reg = np.where(mad < 1e-8, std_x, mad); mad_reg[mad_reg < 1e-10] = 1e-10
    u = (X - med) / (9.0 * mad_reg)
    w = np.where(np.abs(u) < 1, (1-u**2)**2, 0.0)
    X_t = w * (X - med)
    norms = np.linalg.norm(X_t, axis=1, keepdims=True); norms[norms == 0] = 1e-10
    return np.clip((X_t/norms) @ (X_t/norms).T, -1.0, 1.0)

def apply_dropout(matrix, rate, seed):
    np.random.seed(seed)
    result = matrix.copy()
    nonzero_mask = matrix > 0
    dropout_mask = np.random.rand(*matrix.shape) < rate
    result[nonzero_mask & dropout_mask] = 0
    return result

# ── Statistik: Hodges-Lehmann + CI distribution-free ──────────
def hodges_lehmann_ci(d, alpha=0.05):
    d = np.asarray(d, dtype=float); n = len(d)
    i, j = np.triu_indices(n, k=0)
    walsh = np.sort((d[i] + d[j]) / 2.0)
    M = len(walsh)
    z = norm.ppf(1 - alpha/2)
    k = max(int(np.floor(n*(n+1)/4 - z*np.sqrt(n*(n+1)*(2*n+1)/24))), 0)
    return np.median(walsh), walsh[k], walsh[M-k-1]

def bootstrap_median_ci(d, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(np.asarray(d,float), size=(n_boot,len(d)), replace=True), axis=1)
    return np.median(d), np.percentile(boots,100*alpha/2), np.percentile(boots,100*(1-alpha/2))

# ── Muat data ─────────────────────────────────────────────────
X_pbmc = np.load(f"{save_dir}/X_2000hvg.npy")
mu_pbmc = np.load(f"{save_dir}/global_mean_2000hvg.npy")[0]
X_paul = np.load(f"{save_dir}/X_paul15.npy")
mu_paul = np.load(f"{save_dir}/global_mean_paul15.npy")[0]

datasets = {"PBMC3K": (X_pbmc, mu_pbmc), "Paul15": (X_paul, mu_paul)}
COMPARATORS = ["Pearson", "Cosine", "Spearman", "Bicor-SD"]

all_tau = {}   # (dataset, method, rate) -> array 20 nilai

for ds_name, (X, mu) in datasets.items():
    print(f"\n{'='*65}\nDATASET: {ds_name}\n{'='*65}")
    methods = {
        "GMCC":     lambda M: gmcc_vectorized(M, mu),
        "Pearson":  pearson_vectorized,
        "Cosine":   cosine_vectorized,
        "Spearman": spearman_vectorized,
        "Bicor-SD": bicor_sd_vectorized,
    }
    idx = np.triu_indices(X.shape[0], k=1)
    print("  Menghitung similarity original...", end="", flush=True)
    sim_original = {name: func(X)[idx] for name, func in methods.items()}
    print(" ✓")

    for rate in DROPOUT_LEVELS:
        print(f"  Dropout {rate*100:.0f}% ({N_SEEDS} seed)...", end="", flush=True)
        t0 = time.time()
        taus = {name: [] for name in methods}
        for seed in range(1, N_SEEDS+1):
            X_drop = apply_dropout(X, rate, seed)
            for name, func in methods.items():
                s_drop = func(X_drop)[idx]
                tau, _ = kendalltau(sim_original[name], s_drop)
                taus[name].append(tau if not np.isnan(tau) else 0.0)
        for name in methods:
            all_tau[(ds_name, name, rate)] = np.array(taus[name])
        print(f" ✓ ({time.time()-t0:.1f}s)")

# ── SIMPAN NILAI MENTAH (agar tidak perlu run ulang lagi) ─────
flat = {f"{ds}|{m}|{r}": v for (ds, m, r), v in all_tau.items()}
np.savez(f"{save_dir}/raw_kendall_tau_per_seed.npz", **flat)
print(f"\n✓ Nilai tau mentah disimpan: raw_kendall_tau_per_seed.npz")

# ── HASIL: CI per level dan pooled ────────────────────────────
for ds_name in datasets:
    print(f"\n{'='*65}\nHODGES-LEHMANN + 95% CI — {ds_name}\n{'='*65}")

    print(f"\n  POOLED (n=80, semua dropout digabung):")
    print(f"  {'Comparison':<22}{'HL est':>10}{'95% CI':>26}")
    print("  " + "-"*58)
    for comp in COMPARATORS:
        d = np.concatenate([all_tau[(ds_name,"GMCC",r)] - all_tau[(ds_name,comp,r)]
                            for r in DROPOUT_LEVELS])
        hl, lo, hi = hodges_lehmann_ci(d)
        bm, blo, bhi = bootstrap_median_ci(d)
        print(f"  GMCC vs {comp:<14}{hl:>+10.4f}   [{lo:+.4f}, {hi:+.4f}]")
        print(f"  {'(bootstrap check)':<22}{bm:>+10.4f}   [{blo:+.4f}, {bhi:+.4f}]")

    print(f"\n  PER LEVEL (n=20):")
    for rate in DROPOUT_LEVELS:
        print(f"\n    Dropout {rate*100:.0f}%:")
        for comp in COMPARATORS:
            d = all_tau[(ds_name,"GMCC",rate)] - all_tau[(ds_name,comp,rate)]
            hl, lo, hi = hodges_lehmann_ci(d)
            print(f"      vs {comp:<12}{hl:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]")

print("\nSELESAI")