import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("STAGE 7 (2000 HVG): CELL CLUSTERING + ABLATION STUDY")
print("=" * 60)

# ── 1. PERSIAPAN DATA ─────────────────────────────────────────
print("\n[1/5] Mempersiapkan data...")
save_dir = "data/processed"
X           = np.load(f"{save_dir}/X_2000hvg.npy")
global_mean = np.load(f"{save_dir}/global_mean_2000hvg.npy")[0]

# X shape: (2000 gen × 2700 sel)
# Untuk clustering sel, kita butuh (2700 sel × 2000 gen)
X_cells = X.T  # (2700 × 2000)

# Load ulang adata untuk pipeline scanpy
adata = sc.datasets.pbmc3k()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]

N_CELLS, N_GENES = X_cells.shape
print(f"      Shape (sel × gen) : {N_CELLS} × {N_GENES}")
print(f"      Global mean       : {global_mean:.6f}")

# ── 2. FUNGSI GMCC UNTUK SEL ──────────────────────────────────
print("\n[2/5] Mendefinisikan fungsi...")

def gmcc_cell_similarity(X_cells, mu):
    """
    Hitung GMCC similarity antar SEL.
    X_cells shape: (n_cells, n_genes)
    """
    X_c    = X_cells - mu
    norms  = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X_c / norms
    return np.clip(X_norm @ X_norm.T, -1.0, 1.0)

def build_neighbor_graph(sim_matrix, adata_obj,
                         n_neighbors=15):
    """
    Konversi similarity matrix ke neighbor graph
    yang kompatibel dengan Scanpy.
    """
    n     = sim_matrix.shape[0]
    dist  = np.clip(1 - sim_matrix, 0, None)
    np.fill_diagonal(dist, 0)

    # Untuk setiap sel, ambil n_neighbors terdekat
    conn  = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dist[i])[1:n_neighbors+1]
        for j in neighbors:
            conn[i, j] = sim_matrix[i, j]
            conn[j, i] = sim_matrix[i, j]

    adata_obj.obsp["connectivities"] = csr_matrix(conn)
    adata_obj.obsp["distances"]      = csr_matrix(dist)
    adata_obj.uns["neighbors"]       = {
        "params": {
            "n_neighbors": n_neighbors,
            "method"     : "gmcc"
        }
    }
    return adata_obj

print("      ✓ Fungsi siap")

# ── 3. PIPELINE 1: STANDAR SCANPY (PCA + PEARSON) ─────────────
print("\n[3/5] Pipeline 1: Standar Scanpy (PCA + Pearson)...")
adata_std = adata.copy()
sc.pp.scale(adata_std, max_value=10)
sc.pp.pca(adata_std, n_comps=50)
sc.pp.neighbors(adata_std, n_neighbors=15, n_pcs=50)
sc.tl.leiden(adata_std, resolution=0.5, random_state=42)
sc.tl.umap(adata_std, random_state=42)

n_clusters_std = len(adata_std.obs["leiden"].unique())
labels_std     = adata_std.obs["leiden"].astype(int).values
print(f"      Jumlah cluster : {n_clusters_std}")
dist_std = adata_std.obs["leiden"].value_counts().sort_index()
for c, cnt in dist_std.items():
    bar = "█" * (cnt // 100)
    print(f"        Cluster {c}: {cnt:4d} sel  {bar}")

# ── 4. PIPELINE 2: GMCC TANPA PCA ─────────────────────────────
print("\n[4/5] Pipeline 2: GMCC tanpa PCA...")

adata_gmcc = adata.copy()
print("      Menghitung GMCC similarity antar sel...",
      end="", flush=True)
sim_gmcc = gmcc_cell_similarity(X_cells, global_mean)
print(" ✓")

print("      Membangun neighbor graph...", end="", flush=True)
adata_gmcc = build_neighbor_graph(
    sim_gmcc, adata_gmcc, n_neighbors=15
)
print(" ✓")

sc.tl.leiden(adata_gmcc, resolution=0.5, random_state=42)
sc.tl.umap(adata_gmcc, random_state=42)

n_clusters_gmcc = len(adata_gmcc.obs["leiden"].unique())
labels_gmcc     = adata_gmcc.obs["leiden"].astype(int).values
print(f"      Jumlah cluster : {n_clusters_gmcc}")
dist_gmcc = adata_gmcc.obs["leiden"].value_counts().sort_index()
for c, cnt in dist_gmcc.items():
    bar = "█" * (cnt // 100)
    print(f"        Cluster {c}: {cnt:4d} sel  {bar}")

# ── 5. PIPELINE 3: PCA + GMCC (ABLATION STUDY) ────────────────
print("\n[5/5] Pipeline 3: PCA + GMCC (Ablation Study)...")
print("      Tujuan: apakah PCA mengubah karakteristik GMCC?")

adata_pca_gmcc = adata.copy()

# Jalankan PCA dulu
sc.pp.scale(adata_pca_gmcc, max_value=10)
sc.pp.pca(adata_pca_gmcc, n_comps=50)

# Hitung GMCC pada ruang PCA (bukan expression space)
X_pca      = adata_pca_gmcc.obsm["X_pca"]  # (2700 × 50)
mu_pca     = X_pca.mean()
print(f"      Global mean PCA space : {mu_pca:.6f}")

print("      Menghitung GMCC pada ruang PCA...",
      end="", flush=True)
sim_pca_gmcc = gmcc_cell_similarity(X_pca, mu_pca)
print(" ✓")

print("      Membangun neighbor graph...", end="", flush=True)
adata_pca_gmcc = build_neighbor_graph(
    sim_pca_gmcc, adata_pca_gmcc, n_neighbors=15
)
print(" ✓")

sc.tl.leiden(adata_pca_gmcc, resolution=0.5, random_state=42)
sc.tl.umap(adata_pca_gmcc, random_state=42)

n_clusters_pca_gmcc = len(
    adata_pca_gmcc.obs["leiden"].unique()
)
labels_pca_gmcc = \
    adata_pca_gmcc.obs["leiden"].astype(int).values
print(f"      Jumlah cluster : {n_clusters_pca_gmcc}")
dist_pca = adata_pca_gmcc.obs["leiden"].value_counts()\
           .sort_index()
for c, cnt in dist_pca.items():
    bar = "█" * (cnt // 100)
    print(f"        Cluster {c}: {cnt:4d} sel  {bar}")

# ── 6. EVALUASI ────────────────────────────────────────────────
print("\n[6/6] Evaluasi clustering...")

# Silhouette Score (sampel 500 sel untuk efisiensi)
X_pca_eval = adata_std.obsm["X_pca"]  # referensi PCA

sil_std      = silhouette_score(
    X_pca_eval, labels_std, sample_size=500, random_state=42
)
sil_gmcc     = silhouette_score(
    X_pca_eval, labels_gmcc, sample_size=500, random_state=42
)
sil_pca_gmcc = silhouette_score(
    X_pca_eval, labels_pca_gmcc, sample_size=500,
    random_state=42
)

# ARI dan NMI
ari_gmcc_vs_std     = adjusted_rand_score(
    labels_std, labels_gmcc
)
ari_pcagmcc_vs_std  = adjusted_rand_score(
    labels_std, labels_pca_gmcc
)
ari_gmcc_vs_pcagmcc = adjusted_rand_score(
    labels_gmcc, labels_pca_gmcc
)

nmi_gmcc_vs_std     = normalized_mutual_info_score(
    labels_std, labels_gmcc
)
nmi_pcagmcc_vs_std  = normalized_mutual_info_score(
    labels_std, labels_pca_gmcc
)
nmi_gmcc_vs_pcagmcc = normalized_mutual_info_score(
    labels_gmcc, labels_pca_gmcc
)

# ── RINGKASAN ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN STAGE 7 (2000 HVG)")
print("=" * 60)

print(f"\n{'Metrik':<30} {'Standar':>10} "
      f"{'GMCC':>10} {'PCA+GMCC':>10}")
print("-" * 62)
print(f"{'Jumlah cluster':<30} {n_clusters_std:>10} "
      f"{n_clusters_gmcc:>10} {n_clusters_pca_gmcc:>10}")
print(f"{'Silhouette Score':<30} {sil_std:>10.4f} "
      f"{sil_gmcc:>10.4f} {sil_pca_gmcc:>10.4f}")

print(f"\n{'Perbandingan ARI (vs Standar)':<40}")
print(f"  GMCC vs Standar          : {ari_gmcc_vs_std:.4f}")
print(f"  PCA+GMCC vs Standar      : {ari_pcagmcc_vs_std:.4f}")
print(f"  GMCC vs PCA+GMCC         : {ari_gmcc_vs_pcagmcc:.4f}")

print(f"\n{'Perbandingan NMI (vs Standar)':<40}")
print(f"  GMCC vs Standar          : {nmi_gmcc_vs_std:.4f}")
print(f"  PCA+GMCC vs Standar      : {nmi_pcagmcc_vs_std:.4f}")
print(f"  GMCC vs PCA+GMCC         : {nmi_gmcc_vs_pcagmcc:.4f}")

# ── INTERPRETASI ABLATION STUDY ───────────────────────────────
print("\n── Interpretasi Ablation Study ──")
if ari_gmcc_vs_pcagmcc > 0.85:
    print("  GMCC ≈ PCA+GMCC (ARI > 0.85)")
    print("  → GMCC tidak memerlukan PCA preprocessing")
    print("  → Global centering sudah menangkap")
    print("    struktur varians yang relevan")
elif ari_gmcc_vs_pcagmcc > 0.6:
    print("  GMCC dan PCA+GMCC sebagian besar sepakat")
    print("  → PCA memberikan sedikit perbedaan")
    print("    tapi tidak fundamental")
else:
    print("  GMCC dan PCA+GMCC menghasilkan")
    print("  struktur yang berbeda secara substansial")
    print("  → PCA memberikan pengaruh signifikan")

# ── SIMPAN VISUALISASI ────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sc.pl.umap(
    adata_std,
    color="leiden",
    title=f"Standar (PCA+Pearson)\n"
          f"n={n_clusters_std} clusters, "
          f"Sil={sil_std:.3f}",
    ax=axes[0], show=False,
    legend_loc="on data",
)
sc.pl.umap(
    adata_gmcc,
    color="leiden",
    title=f"GMCC (tanpa PCA)\n"
          f"n={n_clusters_gmcc} clusters, "
          f"Sil={sil_gmcc:.3f}",
    ax=axes[1], show=False,
    legend_loc="on data",
)
sc.pl.umap(
    adata_pca_gmcc,
    color="leiden",
    title=f"PCA + GMCC\n"
          f"n={n_clusters_pca_gmcc} clusters, "
          f"Sil={sil_pca_gmcc:.3f}",
    ax=axes[2], show=False,
    legend_loc="on data",
)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
out = "figures/stage7_clustering_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n  Visualisasi disimpan: {out}")
print("\n✓ Stage 7 selesai.")
