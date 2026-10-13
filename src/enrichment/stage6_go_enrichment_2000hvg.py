

import os
import time

import numpy as np
import scanpy as sc
import networkx as nx
import gseapy as gp
from scipy.stats import rankdata

# ── Paths resolved from this file's location ─────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
save_dir = os.path.join(REPO, "data", "processed")
out_dir = os.path.join(REPO, "results", "go_enrichment")

GENE_SETS = "GO_Biological_Process_2023"
PERCENTILE = 95
MIN_MODULE_SIZE = 10
MAX_MODULES = 5

print("=" * 60)
print("STAGE 6 (2000 HVG): NETWORK MODULES AND GO ENRICHMENT")
print("Background: 2,000 highly variable genes (not library default)")
print("=" * 60)

# ── 1. Data ──────────────────────────────────────────────────────────
print("\n[1/4] Preparing data...")
X = np.load(os.path.join(save_dir, "X_2000hvg.npy"))
global_mean = np.load(os.path.join(save_dir, "global_mean_2000hvg.npy"))[0]

adata = sc.datasets.pbmc3k()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]
gene_names = list(adata.var_names)

# This is the enrichment background: every gene that could enter a module.
BACKGROUND = gene_names
print(f"      Genes available: {len(gene_names)}")
print(f"      Enrichment background: {len(BACKGROUND)} genes")


# ── 2. Functions ─────────────────────────────────────────────────────
print("\n[2/4] Defining functions...")


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


def build_network_percentile(sim, gene_names, percentile=PERCENTILE):
    idx = np.triu_indices(len(gene_names), k=1)
    upper = sim[idx]
    threshold = np.percentile(upper, percentile)
    G = nx.Graph()
    G.add_nodes_from(gene_names)
    rows, cols = idx
    mask = upper > threshold
    G.add_edges_from(zip(
        [gene_names[i] for i in rows[mask]],
        [gene_names[j] for j in cols[mask]],
    ))
    return G


def extract_modules(G, min_size=MIN_MODULE_SIZE, max_modules=MAX_MODULES):
    """sorted() rather than list() so module membership is reproducible
    across runs; set iteration order is not stable between processes."""
    n_comp = nx.number_connected_components(G)
    if n_comp > 1:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        modules = [sorted(c) for c in components if len(c) >= min_size]
        method = "connected components"
    else:
        communities = nx.community.greedy_modularity_communities(G)
        modules = sorted(
            [sorted(c) for c in communities if len(c) >= min_size],
            key=len, reverse=True)
        method = "community detection"
    return modules[:max_modules], method


def run_go_enrichment(gene_list, background, max_retries=4):
    """Local hypergeometric test against an explicit background.

    gp.enrich() is used instead of gp.enrichr(): passing a background
    requires the local test, since the Enrichr web API always applies its
    own genome-wide background.
    """
    for attempt in range(max_retries):
        try:
            result = gp.enrich(
                gene_list=gene_list,
                gene_sets=GENE_SETS,
                background=background,
                outdir=None,
                verbose=False,
            )
            df = result.results
            if df is not None and len(df) > 0:
                df_sig = df[df["Adjusted P-value"] < 0.05]
                df_sig = df_sig.sort_values("Adjusted P-value")
                return len(df_sig), df_sig
            return 0, None
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f" [retry {attempt + 1} in {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [failed: {type(e).__name__}: {e}]", end="")
    return 0, None


print("      Functions ready")

# ── 3. Networks and modules ──────────────────────────────────────────
print("\n[3/4] Building networks and extracting modules...")
print(f"      (percentile-{PERCENTILE} threshold)")

methods = {
    "GMCC": lambda X: gmcc_vectorized(X, global_mean),
    "Pearson": pearson_vectorized,
    "Cosine": cosine_vectorized,
    "Spearman": spearman_vectorized,
}

all_modules = {}
for name, func in methods.items():
    print(f"      {name}...", end="", flush=True)
    sim = func(X)
    G = build_network_percentile(sim, gene_names)
    modules, method_used = extract_modules(G)
    all_modules[name] = modules
    print(f" done  ({method_used})")
    print(f"        Modules found: {len(modules)}")
    for i, mod in enumerate(modules):
        preview = ", ".join(mod[:4])
        print(f"        Module {i + 1}: {len(mod)} genes - {preview}...")

# ── 4. GO enrichment ─────────────────────────────────────────────────
print("\n[4/4] GO enrichment analysis...")
print(f"      Background: {len(BACKGROUND)} HVGs")
os.makedirs(out_dir, exist_ok=True)

enrichment_summary = {}
for name, modules in all_modules.items():
    print(f"\n  -- {name} --")
    enrichment_summary[name] = {
        "n_modules": len(modules),
        "genes_tested": sum(len(m) for m in modules),
        "sig_per_module": [],
        "total_sig": 0,
        "top_terms": [],
    }

    for i, module in enumerate(modules):
        print(f"    Module {i + 1} ({len(module)} genes):", end=" ", flush=True)
        n_sig, df_sig = run_go_enrichment(module, background=BACKGROUND)
        enrichment_summary[name]["sig_per_module"].append(n_sig)
        enrichment_summary[name]["total_sig"] += n_sig
        print(f"{n_sig} significant GO terms")

        if df_sig is not None and len(df_sig) > 0:
            fn = f"{name}_module{i + 1}_{len(module)}genes_hvgbg.csv"
            df_sig.to_csv(os.path.join(out_dir, fn), index=False)
            for _, row in df_sig.head(3).iterrows():
                print(f"      - {row['Term'][:45]}")
                print(f"        p-adj={row['Adjusted P-value']:.2e}")
            enrichment_summary[name]["top_terms"].append(
                df_sig.iloc[0]["Term"][:40])

        time.sleep(3)

# ── Summary (manuscript Table 6) ─────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY - equivalent of manuscript Table 6")
print("=" * 60)
print(f"\n  {'Method':<10}{'Modules':>9}{'Genes tested':>14}"
      f"{'GO terms':>10}{'Terms/gene':>12}")
print("  " + "-" * 55)
for name, s in enrichment_summary.items():
    ratio = s["total_sig"] / s["genes_tested"] if s["genes_tested"] else 0.0
    print(f"  {name:<10}{s['n_modules']:>9}{s['genes_tested']:>14,}"
          f"{s['total_sig']:>10}{ratio:>12.3f}")

print(f"\n  Per-module counts:")
for name, s in enrichment_summary.items():
    print(f"    {name:<10}: {s['sig_per_module']}")

print(f"\n  CSV files written to {out_dir}")
print("  Compare against the earlier default-background results before")
print("  updating the manuscript.")