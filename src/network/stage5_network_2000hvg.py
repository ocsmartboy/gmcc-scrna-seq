import scanpy as sc
import numpy as np
import networkx as nx
from scipy.stats import rankdata
from scipy.sparse import issparse
import time
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("STAGE 5 (2000 HVG): GENE CO-EXPRESSION NETWORK")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("\n[1/4] Mempersiapkan data...")
save_dir = "d:/S3/Eksperimen/Bioinformatika/gmcc_biomedical/data/processed"
X           = np.load(f"{save_dir}/X_2000hvg.npy")
global_mean = np.load(f"{save_dir}/global_mean_2000hvg.npy")[0]

# Load nama gen
adata = sc.datasets.pbmc3k()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata      = adata[:, adata.var.highly_variable]
gene_names = list(adata.var_names)

N_GENES, N_CELLS = X.shape
print(f"      Shape      : {N_GENES} gen × {N_CELLS} sel")
print(f"      Gen tersedia: {len(gene_names)}")

# ── 2. DEFINISI FUNGSI ────────────────────────────────────────
print("\n[2/4] Mendefinisikan fungsi...")

def gmcc_vectorized(X, mu):
    X_c   = X - mu
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X_c / norms
    return np.clip(X_norm @ X_norm.T, -1.0, 1.0)

def pearson_vectorized(X):
    X_c   = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X_c / norms
    return np.clip(X_norm @ X_norm.T, -1.0, 1.0)

def cosine_vectorized(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X / norms
    return np.clip(X_norm @ X_norm.T, -1.0, 1.0)

def spearman_vectorized(X):
    X_ranked = np.apply_along_axis(rankdata, 1, X)
    return pearson_vectorized(X_ranked)

def bicor_vectorized(X):
    med     = np.median(X, axis=1, keepdims=True)
    mad     = np.median(np.abs(X - med), axis=1, keepdims=True)
    std_x   = np.std(X, axis=1, keepdims=True)
    mad_reg = np.where(mad < 1e-8, std_x, mad)
    mad_reg[mad_reg < 1e-10] = 1e-10
    u       = (X - med) / (9.0 * mad_reg)
    w       = np.where(np.abs(u) < 1, (1 - u**2)**2, 0.0)
    X_tilde = w * (X - med)
    norms   = np.linalg.norm(X_tilde, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm  = X_tilde / norms
    return np.clip(X_norm @ X_norm.T, -1.0, 1.0)

def build_network_percentile(sim_matrix, gene_names,
                              percentile=95):
    """
    Bangun jaringan menggunakan threshold berbasis persentil.
    Dua gen dihubungkan jika similarity mereka masuk
    top (100-percentile)% dari semua pasangan gen.
    """
    # Ambil upper triangle (tanpa diagonal)
    idx       = np.triu_indices(len(gene_names), k=1)
    upper     = sim_matrix[idx]
    threshold = np.percentile(upper, percentile)

    G = nx.Graph()
    G.add_nodes_from(gene_names)

    # Tambahkan edge yang melewati threshold
    n = len(gene_names)
    rows, cols = idx
    mask       = upper > threshold
    for i, j, sim in zip(rows[mask], cols[mask], upper[mask]):
        G.add_edge(gene_names[i], gene_names[j], weight=float(sim))

    return G, threshold

def analyze_network(G, name, threshold):
    """Hitung properti jaringan."""
    n_nodes        = G.number_of_nodes()
    n_edges        = G.number_of_edges()
    density        = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    n_components   = nx.number_connected_components(G)
    degrees        = [d for _, d in G.degree()]
    avg_degree     = np.mean(degrees) if degrees else 0

    print(f"\n      {name} (threshold={threshold:.4f}):")
    print(f"        Edge (koneksi)      : {n_edges:,}")
    print(f"        Density             : {density:.6f}")
    print(f"        Avg clustering coef : {avg_clustering:.4f}")
    print(f"        Connected components: {n_components}")
    print(f"        Avg degree          : {avg_degree:.2f}")

    return {
        "edges"      : n_edges,
        "density"    : density,
        "clustering" : avg_clustering,
        "components" : n_components,
        "avg_degree" : avg_degree,
        "threshold"  : threshold,
    }

print("      ✓ Semua fungsi siap")

# ── 3. HITUNG SIMILARITY & BANGUN JARINGAN ────────────────────
print("\n[3/4] Menghitung similarity dan membangun jaringan...")
print("      (Menggunakan threshold persentil ke-95)")
print("      = top 5% pasangan gen paling mirip")

methods = {
    "GMCC"    : lambda X: gmcc_vectorized(X, global_mean),
    "Pearson" : pearson_vectorized,
    "Cosine"  : cosine_vectorized,
    "Spearman": spearman_vectorized,
    "Bicor"   : bicor_vectorized,
}

networks    = {}
net_stats   = {}
thresholds  = {}

for name, func in methods.items():
    print(f"\n      Menghitung {name}...", end="", flush=True)
    t0  = time.time()
    sim = func(X)
    print(f" similarity ✓ ({time.time()-t0:.1f}s)", end="")

    G, threshold = build_network_percentile(
        sim, gene_names, percentile=95
    )
    networks[name]   = G
    thresholds[name] = threshold
    net_stats[name]  = analyze_network(G, name, threshold)

# ── 4. ANALISIS LANJUTAN ──────────────────────────────────────
print("\n\n[4/4] Analisis lanjutan...")

# Top 10 gen paling terhubung per metode
print("\n  ── Top 10 gen paling terhubung ──")
for name, G in networks.items():
    degree_sorted = sorted(
        G.degree(), key=lambda x: x[1], reverse=True
    )
    top10 = degree_sorted[:10]
    print(f"\n  {name}:")
    for i, (gene, deg) in enumerate(top10):
        print(f"    {i+1:2}. {gene:<20} → {deg} koneksi")

# Overlap jaringan GMCC vs metode lain
print("\n  ── Overlap edge GMCC vs metode lain ──")
gmcc_edges = set(
    frozenset([u, v])
    for u, v in networks["GMCC"].edges()
)

for name in ["Pearson", "Cosine", "Spearman", "Bicor"]:
    other_edges = set(
        frozenset([u, v])
        for u, v in networks[name].edges()
    )
    overlap = len(gmcc_edges & other_edges)
    pct     = overlap / len(gmcc_edges) * 100
    print(f"  GMCC ∩ {name:<12}: "
          f"{overlap:,} edge bersama "
          f"({pct:.1f}% dari GMCC edges)")

# ── RINGKASAN ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN STAGE 5 (2000 HVG, Persentil ke-95)")
print("=" * 60)

print(f"\n{'Properti':<28}", end="")
for name in methods:
    print(f"{name:>10}", end="")
print()
print("-" * 78)

props = [
    ("Threshold"           , "threshold" , ".4f"),
    ("Jumlah edge"         , "edges"     , ","),
    ("Density"             , "density"   , ".6f"),
    ("Avg clustering coef" , "clustering", ".4f"),
    ("Connected components", "components", ""),
    ("Avg degree"          , "avg_degree", ".2f"),
]

for label, key, fmt in props:
    print(f"{label:<28}", end="")
    for name in methods:
        val = net_stats[name][key]
        if fmt == ",":
            print(f"{val:>10,}", end="")
        elif fmt == "":
            print(f"{val:>10}", end="")
        else:
            print(f"{val:>10{fmt}}", end="")
    print()

print("\n✓ Stage 5 selesai. Siap untuk Stage 6 (GO Enrichment).")