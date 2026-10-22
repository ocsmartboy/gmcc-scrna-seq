"""
GO Biological Process enrichment of network modules (manuscript Table 5,
Supplementary Tables S5-S6) for the full 2,000-HVG gene set and for the
detection-filtered gene sets (detected in >= 5% and >= 10% of cells).

Identical to the primary enrichment scripts except that
  * the gene-set library is read from a local GMT file
    (GO_Biological_Process_2023, downloaded from Enrichr), so the analysis
    does not depend on the Enrichr web service;
  * the background is the gene set eligible for network construction:
    the retained HVGs for PBMC 3K and their one-to-one human orthologues
    for Paul15;
  * GMCC uses the global mean of the analysed matrix.

The "all" gene set must reproduce Table 5 (PBMC 3K: 147 / 10 / 0 / 9
significant terms; Paul15: 9 / 0 / 7 / 0); the script reports whether it
does.

With --modules community, modules are greedy-modularity communities
(>= 10 genes, five largest) for every network. This is needed for the
filtered gene sets, whose networks contain one large connected component
and otherwise only components smaller than 10 genes.

Usage
-----
    python src/enrichment/filtered_go_enrichment.py \
        --gmt data/GO_Biological_Process_2023.gmt
    python src/enrichment/filtered_go_enrichment.py --modules community
"""
import argparse
import csv
import os
import sys
import warnings

import gseapy as gp
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src", "stability"))
from paired_difference_ci import (  # noqa: E402
    gmcc_vectorized, pearson_vectorized, cosine_vectorized,
    spearman_vectorized)

warnings.filterwarnings("ignore")
sc.settings.datasetdir = os.path.join(REPO, "data")

DATA_DIR = os.path.join(REPO, "data", "processed")
OUT_DIR = os.path.join(REPO, "results", "filtered_go_enrichment")
TYPED_TABLE = os.path.join(DATA_DIR, "mouse_human_orthologs_typed.csv")
GENE_SETS = {"all": 0.0, "det5": 0.05, "det10": 0.10}
METHODS = ["GMCC", "Pearson", "Cosine", "Spearman"]
PERCENTILE = 95
MIN_MODULE_SIZE = 10
MAX_MODULES = 5
MIN_CONVERTED = 5
EXPECTED_ALL = {"PBMC3K": {"GMCC": 147, "Pearson": 10, "Cosine": 0, "Spearman": 9},
                "Paul15": {"GMCC": 9, "Pearson": 0, "Cosine": 7, "Spearman": 0}}


def hvg_names(loader, n_rows):
    adata = loader()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    names = list(adata.var_names[adata.var.highly_variable])
    if len(names) != n_rows:
        sys.exit("Gene list does not match the stored matrix.")
    return names


def similarity(m, X, mu):
    return {"GMCC": lambda: gmcc_vectorized(X, mu),
            "Pearson": lambda: pearson_vectorized(X),
            "Cosine": lambda: cosine_vectorized(X),
            "Spearman": lambda: spearman_vectorized(X)}[m]()


def modules_of(sim, names, rule="component"):
    idx = np.triu_indices(len(names), k=1)
    upper = sim[idx]
    edge = upper > np.percentile(upper, PERCENTILE)
    G = nx.Graph()
    G.add_nodes_from(names)
    G.add_edges_from(zip([names[i] for i in idx[0][edge]],
                         [names[j] for j in idx[1][edge]]))
    if rule == "component" and nx.number_connected_components(G) > 1:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        mods = [sorted(c) for c in comps if len(c) >= MIN_MODULE_SIZE]
        how = "connected components"
    else:
        comms = nx.community.greedy_modularity_communities(G)
        mods = sorted([sorted(c) for c in comms if len(c) >= MIN_MODULE_SIZE],
                      key=len, reverse=True)
        how = "greedy modularity"
    return mods[:MAX_MODULES], how


def one2one_map():
    t = pd.read_csv(TYPED_TABLE)
    t = t[t["hsapiens_homolog_orthology_type"] == "ortholog_one2one"]
    g = t.groupby("external_gene_name")[
        "hsapiens_homolog_associated_gene_name"].nunique()
    ok = set(g.index[g == 1])
    t = t[t["external_gene_name"].isin(ok)]
    return dict(zip(t["external_gene_name"],
                    t["hsapiens_homolog_associated_gene_name"]))


def enrich(genes, background, gmt):
    res = gp.enrich(gene_list=genes, gene_sets=gmt, background=background,
                    outdir=None, verbose=False).results
    if res is None or len(res) == 0:
        return None
    sig = res[res["Adjusted P-value"] < 0.05]
    return sig.sort_values("Adjusted P-value") if len(sig) else None


def main(gmt, rule="component"):
    out = OUT_DIR if rule == "component" else OUT_DIR + "_community"
    os.makedirs(out, exist_ok=True)
    o2o = one2one_map()
    datasets = {
        "PBMC3K": ("X_2000hvg.npy", "global_mean_2000hvg.npy",
                   sc.datasets.pbmc3k, None),
        "Paul15": ("X_paul15.npy", "global_mean_paul15.npy",
                   sc.datasets.paul15, o2o),
    }
    summary, mod_rows = [], []
    totals = {}
    for ds, (xf, mf, loader, mapping) in datasets.items():
        X_full = np.load(os.path.join(DATA_DIR, xf))
        mu_stored = float(np.load(os.path.join(DATA_DIR, mf))[0])
        names_full = hvg_names(loader, X_full.shape[0])
        det = (X_full > 0).mean(axis=1)
        conv = ((lambda gl: sorted({mapping[g] for g in gl if g in mapping}))
                if mapping else (lambda gl: sorted(gl)))
        for gs, thr in GENE_SETS.items():
            keep = det >= thr
            X = X_full[keep]
            names = [n for n, k in zip(names_full, keep) if k]
            mu = mu_stored if gs == "all" else float(X.mean())
            background = conv(names)
            print(f"\n{ds} / {gs}: {len(names)} genes, background "
                  f"{len(background)}, mu = {mu:.4f}")
            for m in METHODS:
                mods, how = modules_of(similarity(m, X, mu), names, rule)
                n_sig_tot = 0
                for k, mod in enumerate(mods, 1):
                    genes = conv(mod)
                    df = enrich(genes, background, gmt) \
                        if len(genes) >= MIN_CONVERTED else None
                    n_sig = 0 if df is None else len(df)
                    n_sig_tot += n_sig
                    top = "" if df is None else df.iloc[0]["Term"]
                    top_p = "" if df is None else f"{df.iloc[0]['Adjusted P-value']:.2e}"
                    if df is not None:
                        df.to_csv(os.path.join(
                            out, f"{ds}_{gs}_{m}_module{k}_{len(mod)}genes.csv"),
                            index=False)
                    mod_rows.append([ds, gs, m, k, len(mod), len(genes), n_sig,
                                     top, top_p])
                    print(f"  {m:<9} M{k} {len(mod):>4} genes "
                          f"({len(genes)} tested): {n_sig:>4} terms  {top[:50]}")
                totals[(ds, gs, m)] = n_sig_tot
                summary.append([ds, gs, len(names), len(background), m, how,
                                len(mods), sum(len(x) for x in mods), n_sig_tot])

    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "gene_set", "n_genes", "background", "estimator",
                    "module_method", "n_modules", "module_genes",
                    "significant_terms"])
        w.writerows(summary)
    with open(os.path.join(out, "modules.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "gene_set", "estimator", "module", "module_genes",
                    "genes_tested", "significant_terms", "top_term", "top_padj"])
        w.writerows(mod_rows)

    if rule != "component":
        return
    ok = all(totals[(d, "all", m)] == v
             for d, e in EXPECTED_ALL.items() for m, v in e.items())
    print("\nReproduction of Table 5 with the full gene set: "
          + ("MATCH" if ok else "DIFFERENT - check the GMT file"))
    for d, e in EXPECTED_ALL.items():
        print(f"  {d}: " + ", ".join(
            f"{m} {totals[(d, 'all', m)]} (Table 5: {v})" for m, v in e.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GO enrichment on filtered genes.")
    ap.add_argument("--gmt", default=os.path.join(
        REPO, "data", "GO_Biological_Process_2023.gmt"))
    ap.add_argument("--modules", choices=["component", "community"],
                    default="component",
                    help="component: connected components (greedy modularity "
                         "only when the network is one component), as in the "
                         "primary analysis; community: greedy modularity "
                         "communities for every network")
    a = ap.parse_args()
    main(a.gmt, a.modules)
