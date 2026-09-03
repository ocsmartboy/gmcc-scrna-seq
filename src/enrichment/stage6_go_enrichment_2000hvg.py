import scanpy as sc
import numpy as np
import networkx as nx
import gseapy as gp
from scipy.stats import rankdata
from scipy.sparse import issparse
import time
import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("STAGE 6 (2000 HVG): BIOLOGICAL VALIDATION")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("\n[1/4] Mempersiapkan data...")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
save_dir = os.path.join(REPO, "data", "processed")
X           = np.load(f"{save_dir}/X_2000hvg.npy")
global_mean = np.load(f"{save_dir}/global_mean_2000hvg.npy")[0]

adata = sc.datasets.pbmc3k()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata      = adata[:, adata.var.highly_variable]
gene_names = list(adata.var_names)

N_GENES = X.shape[0]
print(f"      Gen tersedia: {N_GENES}")

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
    w       = np.where(np.abs(u) < 1, (1-u**2)**2, 0.0)
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

def extract_modules(G, min_size=10, max_modules=5):
    """
    Ekstrak modul gen:
    - Untuk jaringan dengan banyak komponen:
      gunakan connected components langsung
    - Untuk jaringan dengan 1 komponen:
      gunakan greedy modularity communities
    """
    n_comp = nx.number_connected_components(G)

    if n_comp > 1:
        # Gunakan connected components langsung
        components = sorted(
            nx.connected_components(G),
            key=len, reverse=True
        )
        modules = [
            sorted(c) for c in components
            if len(c) >= min_size
        ]
    else:
        # Gunakan community detection
        communities = nx.community.greedy_modularity_communities(G)
        modules     = sorted(
            [sorted(c) for c in communities
             if len(c) >= min_size],
            key=len, reverse=True
        )

    return modules[:max_modules]

def run_go_enrichment(gene_list, max_retries=4):
    """Jalankan GO enrichment dengan retry; return jumlah GO terms signifikan."""
    for attempt in range(max_retries):
        try:
            result = gp.enrichr(
                gene_list  = gene_list,
                gene_sets  = "GO_Biological_Process_2023",
                outdir     = None,
                verbose    = False,
            )
            df = result.results
            if df is not None and len(df) > 0:
                df_sig = df[df["Adjusted P-value"] < 0.05]
                df_sig = df_sig.sort_values("Adjusted P-value")
                return len(df_sig), df_sig
            return 0, None
        except Exception as e:
            wait = 5 * (attempt + 1)          # 5s, 10s, 15s, 20s
            if attempt < max_retries - 1:
                print(f" [retry {attempt+1}/{max_retries-1} "
                      f"in {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [failed: {type(e).__name__}]", end="")
    return 0, None

# ── 3. BANGUN JARINGAN & EKSTRAK MODUL ────────────────────────
print("\n[3/4] Membangun jaringan dan mengekstrak modul...")
print("      (menggunakan threshold persentil ke-95)")

methods = {
    "GMCC"    : lambda X: gmcc_vectorized(X, global_mean),
    "Pearson" : pearson_vectorized,
    "Cosine"  : cosine_vectorized,
    "Spearman": spearman_vectorized,
    "Bicor-SD"   : bicor_vectorized,
}

all_modules = {}

for name, func in methods.items():
    print(f"\n      {name}...", end="", flush=True)
    sim      = func(X)
    G, thr   = build_network_percentile(sim, gene_names)
    modules  = extract_modules(G, min_size=10, max_modules=5)

    all_modules[name] = modules
    n_comp = nx.number_connected_components(G)
    method = "connected components" if n_comp > 1 \
             else "community detection"

    print(f" ✓  ({method})")
    print(f"        Modul ditemukan: {len(modules)}")
    for i, mod in enumerate(modules):
        print(f"        Modul {i+1}: {len(mod)} gen — "
              f"{', '.join(mod[:4])}...")

# ── 4. GO ENRICHMENT ANALYSIS ─────────────────────────────────
print("\n[4/4] GO Enrichment Analysis...")
print("      (memerlukan koneksi internet)\n")

enrichment_summary = {}

for name, modules in all_modules.items():
    print(f"\n  ── {name} ──")
    enrichment_summary[name] = {
        "n_modules"    : len(modules),
        "sig_per_module": [],
        "total_sig"    : 0,
        "top_terms"    : [],
    }

    for i, module in enumerate(modules):
        print(f"    Modul {i+1} ({len(module)} gen):",
              end="", flush=True)

        n_sig, df_sig = run_go_enrichment(module)
        enrichment_summary[name]["sig_per_module"].append(n_sig)
        enrichment_summary[name]["total_sig"] += n_sig

        print(f" {n_sig} GO terms signifikan")
        time.sleep(3)

        if df_sig is not None and len(df_sig) > 0:
            out_dir = os.path.join(REPO, "results", "go_enrichment")
            os.makedirs(out_dir, exist_ok=True)
            df_sig.to_csv(
                os.path.join(out_dir,
                             f"{name}_module{i+1}_{len(module)}genes.csv"),
                index=False)

            for _, row in df_sig.head(3).iterrows():
                print(f"      • {row['Term'][:45]}")
                print(f"        p-adj={row['Adjusted P-value']:.2e}")
            enrichment_summary[name]["top_terms"].append(
                df_sig.iloc[0]["Term"][:40])

# ── RINGKASAN ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN STAGE 6 (2000 HVG)")
print("=" * 60)

print(f"\n{'Metrik':<35}", end="")
for name in methods:
    print(f"{name:>10}", end="")
print()
print("-" * 85)

# Jumlah modul
print(f"{'Jumlah modul (≥10 gen)':<35}", end="")
for name in methods:
    print(f"{enrichment_summary[name]['n_modules']:>10}", end="")
print()

# Total GO terms signifikan
print(f"{'Total GO terms signifikan':<35}", end="")
for name in methods:
    print(f"{enrichment_summary[name]['total_sig']:>10}", end="")
print()

# Rata-rata GO terms per modul
print(f"{'Avg GO terms per modul':<35}", end="")
for name in methods:
    sig_list = enrichment_summary[name]["sig_per_module"]
    avg      = np.mean(sig_list) if sig_list else 0
    print(f"{avg:>10.1f}", end="")
print()

# Proporsi modul yang signifikan (>0 GO terms)
print(f"{'Modul dengan GO terms > 0':<35}", end="")
for name in methods:
    sig_list = enrichment_summary[name]["sig_per_module"]
    n_sig    = sum(1 for x in sig_list if x > 0)
    total    = len(sig_list)
    pct      = n_sig/total*100 if total > 0 else 0
    print(f"{pct:>9.0f}%", end="")
print()

print("\n✓ Stage 6 selesai.")
print("  Siap untuk Stage 7 (Cell Clustering).")
