import numpy as np
import networkx as nx
from scipy.stats import rankdata
import time
import os

print("=" * 60)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 60)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
save_dir = os.path.join(REPO, "data", "processed")

# ── Fungsi similarity (disalin persis dari kode existing) ────
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

def clustering_at_percentile(sim, n_genes, percentile):
    idx       = np.triu_indices(n_genes, k=1)
    upper     = sim[idx]
    threshold = np.percentile(upper, percentile)
    G         = nx.Graph()
    G.add_nodes_from(range(n_genes))
    rows, cols = idx
    mask       = upper > threshold
    G.add_edges_from(zip(rows[mask], cols[mask]))
    return nx.average_clustering(G)

# ── Muat data yang sudah diproses (dari Stage 2) ──────────────
print("\n[1/2] Memuat data...")
X_pbmc  = np.load(f"{save_dir}/X_2000hvg.npy")
mu_pbmc = np.load(f"{save_dir}/global_mean_2000hvg.npy")[0]
X_paul  = np.load(f"{save_dir}/X_paul15.npy")
mu_paul = np.load(f"{save_dir}/global_mean_paul15.npy")[0]

datasets = {"PBMC3K": (X_pbmc, mu_pbmc), "Paul15": (X_paul, mu_paul)}
methods  = {
    "GMCC"    : lambda X, mu: gmcc_vectorized(X, mu),
    "Pearson" : lambda X, mu: pearson_vectorized(X),
    "Cosine"  : lambda X, mu: cosine_vectorized(X),
    "Spearman": lambda X, mu: spearman_vectorized(X),
    "Bicor"   : lambda X, mu: bicor_vectorized(X),
}
percentiles = [90, 95, 97.5]

# ── Hitung clustering coefficient di tiap kombinasi ───────────
print("\n[2/2] Menghitung clustering coefficient...")
results = {pct: {} for pct in percentiles}

for pct in percentiles:
    print(f"\n  Percentile {pct}:")
    for m_name, func in methods.items():
        results[pct][m_name] = {}
        for ds_name, (X, mu) in datasets.items():
            t0  = time.time()
            sim = func(X, mu)
            cc  = clustering_at_percentile(sim, X.shape[0], pct)
            results[pct][m_name][ds_name] = cc
            print(f"    {m_name:<10} {ds_name:<8}: CC={cc:.4f} ({time.time()-t0:.1f}s)")

# ── Ringkasan: |Delta| lintas dataset per percentile ──────────
print("\n" + "=" * 60)
print("RINGKASAN: |Delta| Cross-Dataset per Percentile Threshold")
print("=" * 60)
header = f"{'Method':<10} " + " ".join(f"P{p:<8}" for p in percentiles)
print(f"\n{header}")
for m_name in methods:
    row = f"{m_name:<10} "
    for pct in percentiles:
        d = abs(results[pct][m_name]["PBMC3K"] - results[pct][m_name]["Paul15"])
        row += f"{d:<9.4f}"
    print(row)

print("\nSELESAI")
