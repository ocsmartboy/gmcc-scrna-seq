import scanpy as sc
import numpy as np
import networkx as nx
from scipy.stats import rankdata
import time
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("STAGE 5 — PAUL ET AL. 2015: GENE CO-EXPRESSION NETWORK")
print("(Topologi network saja, tanpa GO enrichment)")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("\n[1/4] Memuat data...")
save_dir = ("data/processed")

X           = np.load(f"{save_dir}/X_paul15.npy")
global_mean = np.load(f"{save_dir}/global_mean_paul15.npy")[0]

# Load nama gen dari adata (urutan harus sama persis
# dengan saat X disimpan)
adata = sc.datasets.paul15()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata      = adata[:, adata.var.highly_variable]
gene_names = list(adata.var_names)

N_GENES = X.shape[0]
print(f"      Gen tersedia: {N_GENES}")
print(f"      Global mean : {global_mean:.6f}")

# ── 2. DEFINISI FUNGSI ────────────────────────────────────────
print("\n[2/4] Mendefinisikan fungsi...")

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

def build_network_percentile(sim, gene_names, percentile=95):
    idx       = np.triu_indices(len(gene_names), k=1)
    upper     = sim[idx]
    threshold = np.percentile(upper, percentile)
    G         = nx.Graph()
    G.add_nodes_from(gene_names)
    rows, cols = idx
    mask       = upper > threshold
    for i, j, s in zip(rows[mask], cols[mask], upper[mask]):
        G.add_edge(gene_names[i], gene_names[j],
                   weight=float(s))
    return G, threshold

def analyze_network(G, name, threshold):
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
print("      (threshold persentil ke-95, konsisten dengan PBMC 3K)")

methods = {
    "GMCC"    : lambda X: gmcc_vectorized(X, global_mean),
    "Pearson" : pearson_vectorized,
    "Cosine"  : cosine_vectorized,
    "Spearman": spearman_vectorized,
    "Bicor"   : bicor_vectorized,
}

networks   = {}
net_stats  = {}

for name, func in methods.items():
    print(f"\n      Menghitung {name}...", end="", flush=True)
    t0  = time.time()
    sim = func(X)
    print(f" similarity ✓ ({time.time()-t0:.1f}s)", end="")

    G, threshold      = build_network_percentile(
        sim, gene_names, percentile=95
    )
    networks[name]  = G
    net_stats[name] = analyze_network(G, name, threshold)

# ── 4. TOP GENES & OVERLAP ────────────────────────────────────
print("\n\n[4/4] Analisis lanjutan...")

print("\n  ── Top 5 gen paling terhubung per metode ──")
for name, G in networks.items():
    degree_sorted = sorted(
        G.degree(), key=lambda x: x[1], reverse=True
    )
    print(f"\n  {name}:")
    for i, (gene, deg) in enumerate(degree_sorted[:5]):
        print(f"    {i+1}. {gene:<15} → {deg} koneksi")

print("\n  ── Overlap edge GMCC vs metode lain ──")
gmcc_edges = set(
    frozenset([u, v]) for u, v in networks["GMCC"].edges()
)
for name in ["Pearson", "Cosine", "Spearman", "Bicor"]:
    other_edges = set(
        frozenset([u, v]) for u, v in networks[name].edges()
    )
    overlap = len(gmcc_edges & other_edges)
    pct     = overlap / len(gmcc_edges) * 100
    print(f"  GMCC ∩ {name:<12}: {overlap:,} edge bersama "
          f"({pct:.1f}% dari GMCC edges)")

# ── RINGKASAN ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN STAGE 5 — PAUL ET AL. 2015")
print("=" * 60)

print(f"\n{'Properti':<28}", end="")
for name in methods:
    print(f"{name:>10}", end="")
print()
print("-" * 78)

props = [
    ("Threshold"           , "threshold" , ".4f"),
    ("Jumlah edge"         , "edges"     , ","),
    ("Avg clustering coef" , "clustering", ".4f"),
    ("Connected components", "components", ""),
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

# ── PERBANDINGAN LINTAS DATASET ───────────────────────────────
print("\n── Perbandingan Clustering Coefficient Lintas Dataset ──")
print(f"\n  {'Metode':<12} {'PBMC 3K':>10} {'Paul15':>10}")
print(f"  {'-'*34}")

# Nilai dari eksperimen PBMC 3K sebelumnya
pbmc_clustering = {
    "GMCC": 0.3032, "Pearson": 0.2169, "Cosine": 0.6111,
    "Spearman": 0.2294, "Bicor": 0.5506,
}

for name in methods:
    pbmc_val = pbmc_clustering[name]
    paul_val = net_stats[name]["clustering"]
    print(f"  {name:<12} {pbmc_val:>10.4f} {paul_val:>10.4f}")

print("\n✓ Stage 5 Paul et al. selesai.")
print("  Semua eksperimen inti untuk Paper 2 SELESAI.")
