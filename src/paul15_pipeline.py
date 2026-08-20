import scanpy as sc
import numpy as np
from scipy.stats import rankdata, kendalltau
import time
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("PIPELINE PAUL ET AL. 2015 (MOUSE BONE MARROW)")
print("=" * 60)

save_dir = ("data/processed")

# ── STAGE 1: PREPROCESSING ────────────────────────────────────
print("\n[STAGE 1] Preprocessing...")

adata = sc.datasets.paul15()
print(f"  Data awal   : {adata.shape[0]} sel × "
      f"{adata.shape[1]} gen")
print(f"  Sparsity awal: "
      f"{(adata.X == 0).sum() / adata.X.size * 100:.1f}%")

# Preprocessing standar
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

X = adata.X if not hasattr(adata.X, 'toarray') \
    else adata.X.toarray()
X = X.T  # (n_genes × n_cells)

global_mean = X.mean()
sparsity    = (X == 0).sum() / X.size * 100

print(f"\n  Setelah preprocessing:")
print(f"  Shape       : {X.shape[0]} gen × {X.shape[1]} sel")
print(f"  Sparsity    : {sparsity:.1f}%")
print(f"  Global mean : {global_mean:.6f}")
print(f"  Min ekspresi: {X.min():.4f}")
print(f"  Max ekspresi: {X.max():.4f}")

# Simpan
np.save(f"{save_dir}/X_paul15.npy", X)
np.save(f"{save_dir}/global_mean_paul15.npy",
        np.array([global_mean]))
print("  ✓ Data disimpan")

# ── STAGE 2: SIMILARITY ────────────────────────────────────────
print("\n[STAGE 2] Menghitung similarity matrix...")

def gmcc_vectorized(X, mu):
    X_c   = X - mu
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_c/norms) @ (X_c/norms).T, -1, 1)

def pearson_vectorized(X):
    X_c   = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_c/norms) @ (X_c/norms).T, -1, 1)

def cosine_vectorized(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X/norms) @ (X/norms).T, -1, 1)

def spearman_vectorized(X):
    X_r = np.apply_along_axis(rankdata, 1, X)
    return pearson_vectorized(X_r)

def bicor_vectorized(X):
    med     = np.median(X, axis=1, keepdims=True)
    mad     = np.median(np.abs(X-med), axis=1, keepdims=True)
    std_x   = np.std(X, axis=1, keepdims=True)
    mad_reg = np.where(mad < 1e-8, std_x, mad)
    mad_reg[mad_reg < 1e-10] = 1e-10
    u       = (X - med) / (9.0 * mad_reg)
    w       = np.where(np.abs(u) < 1, (1-u**2)**2, 0.0)
    X_t     = w * (X - med)
    norms   = np.linalg.norm(X_t, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_t/norms) @ (X_t/norms).T, -1, 1)

methods = {
    "GMCC"    : lambda X: gmcc_vectorized(X, global_mean),
    "Pearson" : pearson_vectorized,
    "Cosine"  : cosine_vectorized,
    "Spearman": spearman_vectorized,
    "Bicor"   : bicor_vectorized,
}

sim_original = {}
idx = np.triu_indices(X.shape[0], k=1)

print(f"\n  {'Metode':<12} {'Mean':>8} {'Std':>8} "
      f"{'Time':>6}")
print(f"  {'-'*38}")

for name, func in methods.items():
    t0  = time.time()
    sim = func(X)
    np.fill_diagonal(sim, 1.0)
    sim_original[name] = sim
    elapsed = time.time() - t0
    upper   = sim[idx]
    print(f"  {name:<12} {upper.mean():>8.4f} "
          f"{upper.std():>8.4f} {elapsed:>5.1f}s")

# ── STAGE 3&4: DROPOUT + STABILITY ────────────────────────────
print("\n[STAGE 3&4] Dropout simulation & stability analysis...")

def apply_dropout(X, rate, seed=42):
    np.random.seed(seed)
    result = X.copy()
    mask   = (X > 0) & (np.random.rand(*X.shape) < rate)
    result[mask] = 0
    return result

dropout_levels = [0.1, 0.3, 0.5, 0.7]
results = {
    name: {"kendall": [], "norm_mad": []}
    for name in methods
}

for rate in dropout_levels:
    X_drop   = apply_dropout(X, rate)
    sparsity = (X_drop == 0).sum() / X_drop.size * 100
    print(f"\n  Dropout {rate*100:.0f}% "
          f"(sparsity: {sparsity:.1f}%):")

    for name, func in methods.items():
        sim_drop = func(X_drop)
        s_orig   = sim_original[name][idx]
        s_drop   = sim_drop[idx]

        mad      = np.mean(np.abs(s_orig - s_drop))
        std_orig = np.std(s_orig)
        norm_mad = mad / std_orig if std_orig > 1e-10 else 0
        tau, _   = kendalltau(s_orig, s_drop)
        tau      = tau if not np.isnan(tau) else 0.0

        results[name]["kendall"].append(tau)
        results[name]["norm_mad"].append(norm_mad)

        print(f"    {name:<12} "
              f"Kendall={tau:.4f}  "
              f"NormMAD={norm_mad:.4f}")

# ── RINGKASAN ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN PAUL ET AL. 2015")
print("=" * 60)

print(f"\n  Karakteristik dataset:")
print(f"    Organisme  : Mus musculus (tikus)")
print(f"    Jaringan   : Sumsum tulang (bone marrow)")
print(f"    Sel        : {X.shape[1]}")
print(f"    HVG        : {X.shape[0]}")
print(f"    Sparsity   : {sparsity:.1f}%")
print(f"    Global mean: {global_mean:.6f}")

print(f"\n── Kendall Tau (lebih besar = lebih stabil) ──")
print(f"\n  {'Dropout':<10}", end="")
for name in methods:
    print(f"{name:>12}", end="")
print()
print(f"  {'-'*70}")
for i, rate in enumerate(dropout_levels):
    print(f"  {rate*100:.0f}%{'':<8}", end="")
    for name in methods:
        print(f"{results[name]['kendall'][i]:>12.4f}", end="")
    print()

print(f"\n── Rata-rata Kendall Tau ──")
for name in methods:
    avg = np.mean(results[name]["kendall"])
    deg = (results[name]["kendall"][0] -
           results[name]["kendall"][-1]) / \
          max(results[name]["kendall"][0], 1e-10) * 100
    print(f"  {name:<12}: {avg:.4f}  "
          f"(degradasi: {deg:.1f}%)")

best = max(methods.keys(),
           key=lambda n: np.mean(results[n]["kendall"]))
print(f"\n  Metode paling stabil: {best}")

# ── PERBANDINGAN LINTAS DATASET ───────────────────────────────
print(f"\n── Perbandingan Lintas Dataset ──")
print(f"\n  {'Dataset':<25} {'GMCC':>8} "
      f"{'Pearson':>8} {'Bicor':>8}")
print(f"  {'-'*52}")

# PBMC 3K (dari eksperimen sebelumnya)
pbmc_gmcc    = 0.8340
pbmc_pearson = 0.4669
pbmc_bicor   = 0.6686

paul_gmcc    = np.mean(results["GMCC"]["kendall"])
paul_pearson = np.mean(results["Pearson"]["kendall"])
paul_bicor   = np.mean(results["Bicor"]["kendall"])

print(f"  {'PBMC 3K (human, 91.4%)':<25} "
      f"{pbmc_gmcc:>8.4f} "
      f"{pbmc_pearson:>8.4f} "
      f"{pbmc_bicor:>8.4f}")
print(f"  {'Paul15 (mouse, ~?%)':<25} "
      f"{paul_gmcc:>8.4f} "
      f"{paul_pearson:>8.4f} "
      f"{paul_bicor:>8.4f}")

# Konsistensi
gmcc_diff    = abs(pbmc_gmcc - paul_gmcc)
pearson_diff = abs(pbmc_pearson - paul_pearson)
print(f"\n  Selisih GMCC antar dataset   : {gmcc_diff:.4f}")
print(f"  Selisih Pearson antar dataset: {pearson_diff:.4f}")

if gmcc_diff < 0.1:
    print(f"\n  ✓ GMCC konsisten lintas dataset "
          f"(selisih < 0.1)")
else:
    print(f"\n  ⚠ GMCC menunjukkan variasi antar dataset")

print("\n✓ Pipeline Paul et al. 2015 selesai.")
