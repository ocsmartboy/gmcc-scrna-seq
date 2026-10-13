

import os
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import gseapy as gp
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# ── Paths resolved from this file's location ─────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
save_dir = os.path.join(REPO, "data", "processed")
out_dir = os.path.join(REPO, "results", "go_enrichment")

GENE_SETS = "GO_Biological_Process_2023"
PERCENTILE = 95
MIN_MODULE_SIZE = 10
MAX_MODULES = 5
MIN_CONVERTED = 5

print("=" * 68)
print("PAUL15: GO ENRICHMENT FOR ALL ESTIMATORS")
print("Background: human orthologs of the 2,000 HVGs")
print("=" * 68)


# ── Similarity measures ──────────────────────────────────────────────
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


# ── Network construction and module extraction ───────────────────────
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
    if nx.number_connected_components(G) > 1:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        modules = [sorted(c) for c in comps if len(c) >= min_size]
        method = "connected components"
    else:
        comms = nx.community.greedy_modularity_communities(G)
        modules = sorted([sorted(c) for c in comms if len(c) >= min_size],
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
                return df_sig.sort_values("Adjusted P-value")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f" [retry {attempt + 1} in {wait}s]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f" [failed: {type(e).__name__}: {e}]", end="")
    return None


# ── 1. Data ──────────────────────────────────────────────────────────
print("\n[1/5] Loading data...")
X = np.load(os.path.join(save_dir, "X_paul15.npy"))
global_mean = np.load(os.path.join(save_dir, "global_mean_paul15.npy"))[0]

adata = sc.datasets.paul15()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]
gene_names = list(adata.var_names)
print(f"      Genes available (mouse symbols): {len(gene_names)}")

# ── 2. Networks and modules ──────────────────────────────────────────
print("\n[2/5] Building networks and extracting modules...")

methods = {
    "GMCC": lambda M: gmcc_vectorized(M, global_mean),
    "Pearson": pearson_vectorized,
    "Cosine": cosine_vectorized,
    "Spearman": spearman_vectorized,
}

all_modules = {}
for name, func in methods.items():
    print(f"      {name}...", end="", flush=True)
    sim = func(X)
    G = build_network_percentile(sim, gene_names)
    modules, how = extract_modules(G)
    all_modules[name] = modules
    print(f" done  ({how})")
    print(f"        Modules found: {len(modules)}")
    for i, mod in enumerate(modules):
        print(f"        Module {i + 1}: {len(mod)} genes - "
              f"{', '.join(mod[:4])}...")

# ── 3. Ortholog table (cached) ───────────────────────────────────────
print("\n[3/5] Mouse-to-human ortholog table...")

cache_path = os.path.join(save_dir, "mouse_human_orthologs.csv")

if os.path.exists(cache_path):
    tbl = pd.read_csv(cache_path)
    print(f"      Loaded from cache: {cache_path}")
else:
    from gseapy import Biomart
    tbl = None
    for host in ("www.ensembl.org", "useast.ensembl.org",
                 "asia.ensembl.org"):
        for attempt in range(3):
            try:
                print(f"      Querying {host} "
                      f"(attempt {attempt + 1})...", end="", flush=True)
                bm = Biomart(host=host)
                res = bm.query(
                    dataset="mmusculus_gene_ensembl",
                    attributes=["external_gene_name",
                                "hsapiens_homolog_associated_gene_name"],
                )
                if res is not None and len(res) > 0:
                    tbl = res
                    print(" done")
                    break
                print(" empty response")
            except Exception as e:
                print(f" failed ({type(e).__name__})")
            time.sleep(10)
        if tbl is not None:
            break

    if tbl is None:
        raise SystemExit(
            "      Could not retrieve the ortholog table from any mirror.\n"
            "      Ensembl Biomart may be temporarily unavailable; "
            "retry later."
        )

    tbl = tbl.dropna()
    tbl.to_csv(cache_path, index=False)
    print(f"      Cached to {cache_path}")

tbl = tbl.dropna()
mouse_to_human = dict(zip(
    tbl["external_gene_name"],
    tbl["hsapiens_homolog_associated_gene_name"]))
print(f"      Ortholog pairs available: {len(mouse_to_human):,}")


def convert(genes):
    """Mouse symbols -> unique human orthologs."""
    out = [mouse_to_human[g] for g in genes
           if g in mouse_to_human and mouse_to_human[g] != ""]
    return sorted(set(out))


# ── 4. Background: orthologs of ALL HVGs ─────────────────────────────
print("\n[4/5] Building enrichment background...")
BACKGROUND = convert(gene_names)
bg_rate = len(BACKGROUND) / len(gene_names) * 100
print(f"      HVGs (mouse): {len(gene_names)}")
print(f"      Converted to human orthologs: {len(BACKGROUND)} "
      f"({bg_rate:.1f}%)")
print("      NOTE: report this background mapping rate in the manuscript")
print("            alongside the per-module rate.")

# ── 5. Enrichment ────────────────────────────────────────────────────
print("\n[5/5] GO enrichment against the ortholog background...")
os.makedirs(out_dir, exist_ok=True)

summary = {}
for name, modules in all_modules.items():
    print(f"\n  -- {name} --")
    summary[name] = {
        "n_modules": len(modules),
        "genes_mouse": 0,
        "genes_converted": 0,
        "total_sig": 0,
        "per_module": [],
        "top_terms": [],
    }

    for i, module in enumerate(modules):
        converted = convert(module)
        rate = len(converted) / len(module) * 100
        summary[name]["genes_mouse"] += len(module)
        summary[name]["genes_converted"] += len(converted)
        print(f"    Module {i + 1} ({len(module)} genes, "
              f"{len(converted)} converted, {rate:.1f}%):",
              end=" ", flush=True)

        n_sig = 0
        if len(converted) >= MIN_CONVERTED:
            df_sig = run_go_enrichment(converted, background=BACKGROUND)
            n_sig = len(df_sig) if df_sig is not None else 0
            print(f"{n_sig} significant GO terms")
            if df_sig is not None and len(df_sig) > 0:
                fn = (f"Paul15_{name}_module{i + 1}_"
                      f"{len(module)}genes_hvgbg.csv")
                df_sig.to_csv(os.path.join(out_dir, fn), index=False)
                for _, row in df_sig.head(3).iterrows():
                    print(f"      - {row['Term'][:50]}")
                    print(f"        p-adj={row['Adjusted P-value']:.2e}")
                summary[name]["top_terms"].append(df_sig.iloc[0]["Term"][:45])
            time.sleep(3)
        else:
            print(f"[skipped] fewer than {MIN_CONVERTED} converted genes")

        summary[name]["per_module"].append(n_sig)
        summary[name]["total_sig"] += n_sig

# ── Summary: Paul15 rows for the manuscript table ────────────────────
print("\n" + "=" * 68)
print("SUMMARY - Paul15 rows for the functional-enrichment table")
print("=" * 68)
print(f"\n  Background: {len(BACKGROUND)} human orthologs from "
      f"{len(gene_names)} HVGs ({bg_rate:.1f}%)")
print(f"\n  {'Estimator':<11}{'Modules':>8}{'Genes':>8}{'Conv.':>8}"
      f"{'Rate':>8}{'Terms':>8}")
print("  " + "-" * 51)
for name, s in summary.items():
    rate = (s["genes_converted"] / s["genes_mouse"] * 100
            if s["genes_mouse"] else 0.0)
    print(f"  {name:<11}{s['n_modules']:>8}{s['genes_mouse']:>8}"
          f"{s['genes_converted']:>8}{rate:>7.1f}%{s['total_sig']:>8}")

print("\n  Per-module counts and leading term:")
for name, s in summary.items():
    top = s["top_terms"][0] if s["top_terms"] else "none"
    print(f"    {name:<11}: {s['per_module']}   leading: {top}")

print(f"\n  CSV files written to {out_dir}")