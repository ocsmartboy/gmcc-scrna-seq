"""
Paul15 GO enrichment: sensitivity to the mouse-to-human orthologue rule.

Companion to paul15_go_enrichment_ortholog.py. Networks, modules,
gene-set library and enrichment test are identical; only the orthologue
mapping differs. Each rule is applied to BOTH the module genes and the
background, so every comparison uses a consistent universe.

Rules
-----
  last_entry : original rule (dict(zip(...)) keeps the final entry of the
               BioMart table for a mouse symbol). Reproduces the primary
               analysis from data/processed/mouse_human_orthologs.csv.
  one2one    : Ensembl orthology type == "ortholog_one2one".
  one2one_hc : one2one AND Ensembl orthology confidence == 1.

The typed table (with orthology type and confidence) is retrieved once
from Ensembl BioMart and cached to
data/processed/mouse_human_orthologs_typed.csv together with a small
metadata file recording the retrieval date and host. Archive both files in
the repository: Ensembl releases change over time.

If BioMart cannot be reached, a fallback "one2one_table" rule is derived
from the cached two-column table (mouse symbol and human symbol each
appear exactly once). This is an approximation and is labelled as such in
the output.

Outputs (results/go_enrichment_orthologue_sensitivity/)
-------
  orthologue_rule_summary.csv   one row per rule x estimator
  orthologue_rule_modules.csv   one row per rule x estimator x module
  Paul15_<rule>_<est>_module<k>_<n>genes.csv   significant terms
"""
import argparse
import datetime
import json
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SAVE_DIR = os.path.join(REPO, "data", "processed")
OUT_DIR = os.path.join(REPO, "results", "go_enrichment_orthologue_sensitivity")

LEGACY_TABLE = os.path.join(SAVE_DIR, "mouse_human_orthologs.csv")
TYPED_TABLE = os.path.join(SAVE_DIR, "mouse_human_orthologs_typed.csv")
TYPED_META = os.path.join(SAVE_DIR, "mouse_human_orthologs_typed.json")

GENE_SETS = "GO_Biological_Process_2023"
PERCENTILE = 95
MIN_MODULE_SIZE = 10
MAX_MODULES = 5
MIN_CONVERTED = 5


# ── Similarity measures (identical to paul15_go_enrichment_ortholog.py) ─
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


# ── Network and modules (identical) ──────────────────────────────────
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
    if nx.number_connected_components(G) > 1:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        modules = [sorted(c) for c in comps if len(c) >= min_size]
        method = "connected components"
    else:
        comms = nx.community.greedy_modularity_communities(G)
        modules = sorted([sorted(c) for c in comms if len(c) >= min_size],
                         key=len, reverse=True)
        method = "greedy modularity"
    return modules[:max_modules], method


def run_go_enrichment(gene_list, background, max_retries=4):
    for attempt in range(max_retries):
        try:
            result = gp.enrich(gene_list=gene_list, gene_sets=GENE_SETS,
                               background=background, outdir=None,
                               verbose=False)
            df = result.results
            if df is not None and len(df) > 0:
                df_sig = df[df["Adjusted P-value"] < 0.05]
                return df_sig.sort_values("Adjusted P-value")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f" [failed: {type(e).__name__}: {e}]", end="")
    return None


# ── Orthologue tables ────────────────────────────────────────────────
def fetch_typed_table():
    """Mouse->human homologues with Ensembl orthology type and confidence."""
    if os.path.exists(TYPED_TABLE):
        print(f"      Typed table loaded from cache: {TYPED_TABLE}")
        return pd.read_csv(TYPED_TABLE)

    from gseapy import Biomart
    attrs = ["ensembl_gene_id", "external_gene_name",
             "hsapiens_homolog_ensembl_gene",
             "hsapiens_homolog_associated_gene_name",
             "hsapiens_homolog_orthology_type",
             "hsapiens_homolog_orthology_confidence"]
    for host in ("www.ensembl.org", "useast.ensembl.org",
                 "asia.ensembl.org"):
        for attempt in range(3):
            try:
                print(f"      Querying {host} (attempt {attempt + 1})...",
                      end="", flush=True)
                res = Biomart(host=host).query(
                    dataset="mmusculus_gene_ensembl", attributes=attrs)
                if res is not None and len(res) > 0:
                    res = res.dropna(subset=[
                        "external_gene_name",
                        "hsapiens_homolog_associated_gene_name",
                        "hsapiens_homolog_orthology_type"])
                    res.to_csv(TYPED_TABLE, index=False)
                    with open(TYPED_META, "w") as f:
                        json.dump({
                            "retrieved": datetime.date.today().isoformat(),
                            "host": host,
                            "dataset": "mmusculus_gene_ensembl",
                            "attributes": attrs,
                            "rows": int(len(res)),
                            "note": "Record the Ensembl release shown on "
                                    "the host on the retrieval date.",
                        }, f, indent=2)
                    print(f" done ({len(res):,} rows)")
                    print(f"      Cached to {TYPED_TABLE}")
                    return res
                print(" empty response")
            except Exception as e:
                print(f" failed ({type(e).__name__})")
            time.sleep(10)
    print("      WARNING: BioMart unreachable; Ensembl-typed rules skipped.")
    return None


def unique_symbol_map(df):
    """Mouse symbol -> human symbol, keeping only mouse symbols that map
    to exactly one distinct human symbol."""
    g = df.groupby("external_gene_name")[
        "hsapiens_homolog_associated_gene_name"].nunique()
    ok = set(g.index[g == 1])
    d = df[df["external_gene_name"].isin(ok)]
    return dict(zip(d["external_gene_name"],
                    d["hsapiens_homolog_associated_gene_name"]))


def build_rules():
    rules, notes = {}, {}

    legacy = pd.read_csv(LEGACY_TABLE).dropna()
    rules["last_entry"] = dict(zip(
        legacy["external_gene_name"],
        legacy["hsapiens_homolog_associated_gene_name"]))
    notes["last_entry"] = "original rule: final entry of BioMart table"

    typed = fetch_typed_table()
    if typed is not None:
        o2o = typed[typed["hsapiens_homolog_orthology_type"]
                    == "ortholog_one2one"]
        rules["one2one"] = unique_symbol_map(o2o)
        notes["one2one"] = "Ensembl ortholog_one2one"
        conf = pd.to_numeric(o2o["hsapiens_homolog_orthology_confidence"],
                             errors="coerce")
        rules["one2one_hc"] = unique_symbol_map(o2o[conf == 1])
        notes["one2one_hc"] = "Ensembl ortholog_one2one, confidence = 1"
    else:
        m = legacy.groupby("external_gene_name")[
            "hsapiens_homolog_associated_gene_name"].transform("nunique")
        h = legacy.groupby("hsapiens_homolog_associated_gene_name")[
            "external_gene_name"].transform("nunique")
        sub = legacy[(m == 1) & (h == 1)]
        rules["one2one_table"] = dict(zip(
            sub["external_gene_name"],
            sub["hsapiens_homolog_associated_gene_name"]))
        notes["one2one_table"] = ("APPROXIMATION: unique in both directions "
                                  "within the cached two-column table")
    return rules, notes


def convert(genes, mapping):
    return sorted({mapping[g] for g in genes
                   if g in mapping and mapping[g] != ""})


# ── Main ─────────────────────────────────────────────────────────────
def main(skip_enrichment=False):
    print("=" * 68)
    print("PAUL15 GO ENRICHMENT: ORTHOLOGUE-RULE SENSITIVITY")
    print("=" * 68)

    print("\n[1/4] Loading data...")
    X = np.load(os.path.join(SAVE_DIR, "X_paul15.npy"))
    mu = np.load(os.path.join(SAVE_DIR, "global_mean_paul15.npy"))[0]
    adata = sc.datasets.paul15()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    gene_names = list(adata.var_names)
    if len(gene_names) != X.shape[0]:
        raise SystemExit("Gene list does not match X_paul15.npy rows.")

    print("\n[2/4] Networks and modules...")
    methods = {
        "GMCC": lambda M: gmcc_vectorized(M, mu),
        "Pearson": pearson_vectorized,
        "Cosine": cosine_vectorized,
        "Spearman": spearman_vectorized,
    }
    all_modules = {}
    for name, f in methods.items():
        mods, how = extract_modules(build_network_percentile(f(X),
                                                             gene_names))
        all_modules[name] = (mods, how)
        print(f"      {name}: {[len(m) for m in mods]} ({how})")

    print("\n[3/4] Orthologue rules...")
    rules, notes = build_rules()
    hvg_mapped = {}
    for r, mp in rules.items():
        bg = convert(gene_names, mp)
        n_hvg = sum(1 for g in gene_names if g in mp and mp[g] != "")
        hvg_mapped[r] = (n_hvg, bg)
        print(f"      {r:<14} HVGs mapped {n_hvg:>5} / {len(gene_names)}"
              f"  -> {len(bg):>5} unique human symbols   [{notes[r]}]")

    os.makedirs(OUT_DIR, exist_ok=True)
    # Membership of each HVG under each rule (for the repository).
    pd.DataFrame({
        "mouse_symbol": gene_names,
        **{f"human_{r}": [mp.get(g, "") for g in gene_names]
           for r, mp in rules.items()},
    }).to_csv(os.path.join(OUT_DIR, "hvg_orthologue_by_rule.csv"),
              index=False)

    if skip_enrichment:
        print("\n--skip-enrichment: stopping after mapping summary.")
        return

    print("\n[4/4] GO enrichment...")
    summ, mod_rows = [], []
    for r, mp in rules.items():
        n_hvg, bg = hvg_mapped[r]
        print(f"\n  == rule: {r} (background {len(bg)}) ==")
        for name, (mods, how) in all_modules.items():
            tot_m = tot_c = tot_sig = 0
            for k, mod in enumerate(mods, 1):
                conv = convert(mod, mp)
                n_sig, top, top_p = 0, "", np.nan
                if len(conv) >= MIN_CONVERTED:
                    df = run_go_enrichment(conv, bg)
                    if df is not None and len(df) > 0:
                        n_sig = len(df)
                        top = df.iloc[0]["Term"]
                        top_p = df.iloc[0]["Adjusted P-value"]
                        df.to_csv(os.path.join(
                            OUT_DIR, f"Paul15_{r}_{name}_module{k}_"
                                     f"{len(mod)}genes.csv"), index=False)
                    time.sleep(1)
                tot_m += len(mod)
                tot_c += len(conv)
                tot_sig += n_sig
                mod_rows.append([r, name, k, len(mod), len(conv),
                                 round(100 * len(conv) / len(mod), 1),
                                 n_sig, top, top_p])
                print(f"    {name:<9} M{k} {len(mod):>4} genes, "
                      f"{len(conv):>4} mapped: {n_sig:>4} terms"
                      f"  {top[:45]}")
            summ.append([r, notes[r], n_hvg, len(bg), name, how, len(mods),
                         tot_m, tot_c,
                         round(100 * tot_c / tot_m, 1) if tot_m else 0.0,
                         tot_sig])

    pd.DataFrame(summ, columns=[
        "rule", "rule_description", "hvg_mapped", "background_size",
        "estimator", "module_method", "n_modules", "module_genes",
        "module_genes_mapped", "mapping_rate_pct", "significant_terms",
    ]).to_csv(os.path.join(OUT_DIR, "orthologue_rule_summary.csv"),
              index=False)
    pd.DataFrame(mod_rows, columns=[
        "rule", "estimator", "module", "module_genes", "mapped",
        "mapping_rate_pct", "significant_terms", "top_term", "top_padj",
    ]).to_csv(os.path.join(OUT_DIR, "orthologue_rule_modules.csv"),
              index=False)
    print(f"\nSaved summaries to {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Paul15 GO enrichment under alternative "
                    "mouse-to-human orthologue rules.")
    ap.add_argument("--skip-enrichment", action="store_true",
                    help="only build and summarise the orthologue mappings")
    main(ap.parse_args().skip_enrichment)
