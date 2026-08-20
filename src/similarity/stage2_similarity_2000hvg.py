import scanpy as sc
import numpy as np
from scipy.stats import rankdata
from scipy.sparse import issparse
import time
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("STAGE 2 (2000 HVG): IMPLEMENTASI SIMILARITY — VECTORIZED")
print("=" * 60)

# ── 1. PERSIAPAN DATA ─────────────────────────────────────────
print("\n[1/4] Mempersiapkan data (2000 HVG)...")
adata = sc.datasets.pbmc3k()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

X = adata.X.toarray() if issparse(adata.X) else adata.X
X = X.T  # shape: (2000 gen × 2700 sel)

global_mean = X.mean()
N_GENES, N_CELLS = X.shape

print(f"      Shape  : {N_GENES} gen × {N_CELLS} sel")
print(f"      Sparsity: {(X==0).sum()/X.size*100:.1f}%")
print(f"      Global mean (μ) = {global_mean:.6f}")

# ── 2. IMPLEMENTASI VECTORIZED ────────────────────────────────
print("\n[2/4] Mendefinisikan fungsi similarity vectorized...")

def gmcc_vectorized(X, mu):
    """
    GMCC: center dengan global mean, lalu cosine antar vektor.
    X shape: (n_genes, n_cells)
    """
    X_c = X - mu
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X_c / norms
    return X_norm @ X_norm.T  # (n_genes, n_genes)


def pearson_vectorized(X):
    """
    Pearson: center dengan local mean per gen.
    """
    X_c = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X_c / norms
    return X_norm @ X_norm.T


def cosine_vectorized(X):
    """
    Cosine: tanpa centering.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X / norms
    return X_norm @ X_norm.T


def spearman_vectorized(X):
    """
    Spearman: Pearson pada data yang sudah di-rank.
    Ekuivalen dengan Spearman rank correlation.
    """
    # Rank setiap gen di sepanjang dimensi sel
    X_ranked = np.apply_along_axis(rankdata, 1, X)
    return pearson_vectorized(X_ranked)


def bicor_vectorized(X):
    """
    Biweight Midcorrelation (Bicor) — versi dikoreksi untuk sparse data.

    Pada data scRNA-seq yang sangat sparse (>50% nol):
    - median per gen ≈ 0
    - MAD ≈ 0 (karena mayoritas nilai sudah nol)
    - Solusi: gunakan std sebagai fallback ketika MAD = 0
      (pendekatan standar pada implementasi WGCNA untuk sparse data)
    """
    # Step 1: Median per gen
    med = np.median(X, axis=1, keepdims=True)

    # Step 2: MAD per gen
    mad = np.median(np.abs(X - med), axis=1, keepdims=True)

    # Step 3: Regularisasi MAD untuk sparse data
    # Jika MAD = 0, gunakan std sebagai fallback
    # Ini mencegah kegagalan numerik saat data sangat sparse
    std_x = np.std(X, axis=1, keepdims=True)
    mad_reg = np.where(mad < 1e-8, std_x, mad)
    mad_reg[mad_reg < 1e-10] = 1e-10  # fallback terakhir

    # Step 4: u values dengan MAD yang sudah diregularisasi
    u = (X - med) / (9.0 * mad_reg)

    # Step 5: Bobot biweight
    w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)

    # Step 6: Weighted centered values
    X_tilde = w * (X - med)

    # Step 7: Normalize dan hitung cosine similarity
    norms = np.linalg.norm(X_tilde, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X_tilde / norms
    return X_norm @ X_norm.T

print("      ✓ GMCC, Pearson, Cosine, Spearman, Bicor siap")

# ── 3. HITUNG SIMILARITY MATRIX ───────────────────────────────
print("\n[3/4] Menghitung similarity matrix (2000 × 2000)...")

methods = {
    "GMCC"    : lambda X: gmcc_vectorized(X, global_mean),
    "Pearson" : pearson_vectorized,
    "Cosine"  : cosine_vectorized,
    "Spearman": spearman_vectorized,
    "Bicor"   : bicor_vectorized,
}

sim_matrices = {}

for name, func in methods.items():
    print(f"\n      {name}...", end="", flush=True)
    t0 = time.time()

    sim = func(X)

    # Pastikan diagonal = 1 dan nilai terkliping [-1, 1]
    np.fill_diagonal(sim, 1.0)
    sim = np.clip(sim, -1.0, 1.0)

    elapsed = time.time() - t0
    sim_matrices[name] = sim

    # Statistik dari upper triangle (tanpa diagonal)
    idx   = np.triu_indices(N_GENES, k=1)
    upper = sim[idx]

    print(f" selesai ({elapsed:.1f}s)")
    print(f"        Mean  : {upper.mean():.4f}")
    print(f"        Std   : {upper.std():.4f}")
    print(f"        Min   : {upper.min():.4f}")
    print(f"        Max   : {upper.max():.4f}")

# ── 4. RINGKASAN ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN STAGE 2 (2000 HVG)")
print("=" * 60)

print(f"\n  Gen dianalisis  : {N_GENES}")
print(f"  Sel             : {N_CELLS}")
print(f"  Pasangan gen    : {N_GENES*(N_GENES-1)//2:,}")
print(f"  Global mean μ   : {global_mean:.6f}")

print(f"\n  {'Metode':<12} {'Mean':>8} {'Std':>8} "
      f"{'Min':>8} {'Max':>8}")
print(f"  {'-'*48}")

for name, sim in sim_matrices.items():
    idx   = np.triu_indices(N_GENES, k=1)
    upper = sim[idx]
    print(f"  {name:<12} {upper.mean():>8.4f} "
          f"{upper.std():>8.4f} "
          f"{upper.min():>8.4f} "
          f"{upper.max():>8.4f}")

# Simpan untuk digunakan di stage berikutnya
import os
save_dir = "data/processed"
os.makedirs(save_dir, exist_ok=True)
np.save(f"{save_dir}/X_2000hvg.npy", X)
np.save(f"{save_dir}/global_mean_2000hvg.npy", np.array([global_mean]))
print("\n  File disimpan di data/processed/")
print("\n✓ Stage 2 selesai. Lanjut ke Stage 3 (dropout simulation).")
