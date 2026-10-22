"""
Paul15 enrichment on genes detected in >= 10% of cells under three
mouse-to-human orthologue rules (Supplementary Section S8).

Networks, community modules (five largest greedy-modularity communities
with >= 10 genes), gene-set library and test are those of
filtered_go_enrichment.py --modules community; only the orthologue rule
differs. Each rule is applied to both module genes and background.

  one2one     Ensembl ortholog_one2one (primary analysis)
  one2one_hc  one2one with orthology confidence = 1
  last_entry  all orthologues; a mouse gene with several human orthologues
              keeps the final entry of the retrieved BioMart table

Usage
-----
    python src/enrichment/filtered_orthologue_rules.py --min-detection 0.10
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import filtered_go_enrichment as fge  # noqa: E402

LEGACY = os.path.join(REPO, "data", "processed", "mouse_human_orthologs.csv")


def rules():
    t = pd.read_csv(fge.TYPED_TABLE)
    o2o = t[t["hsapiens_homolog_orthology_type"] == "ortholog_one2one"]

    def uniq(df):
        g = df.groupby("external_gene_name")[
            "hsapiens_homolog_associated_gene_name"].nunique()
        df = df[df["external_gene_name"].isin(set(g.index[g == 1]))]
        return dict(zip(df["external_gene_name"],
                        df["hsapiens_homolog_associated_gene_name"]))

    conf = pd.to_numeric(o2o["hsapiens_homolog_orthology_confidence"],
                         errors="coerce")
    legacy = pd.read_csv(LEGACY).dropna()
    return {"one2one": uniq(o2o),
            "one2one_hc": uniq(o2o[conf == 1]),
            "last_entry": dict(zip(legacy["external_gene_name"],
                                   legacy["hsapiens_homolog_associated_gene_name"]))}


def main(thr, gmt):
    out = os.path.join(REPO, "results", "filtered_orthologue_rules")
    os.makedirs(out, exist_ok=True)
    X_full = np.load(os.path.join(fge.DATA_DIR, "X_paul15.npy"))
    names_full = fge.hvg_names(sc.datasets.paul15, X_full.shape[0])
    det = (X_full > 0).mean(axis=1)
    keep = det >= thr
    X = X_full[keep]
    names = [n for n, k in zip(names_full, keep) if k]
    mu = float(X.mean())
    mods = {m: fge.modules_of(fge.similarity(m, X, mu), names, "community")[0]
            for m in fge.METHODS}
    rows = []
    for r, mp in rules().items():
        conv = lambda gl: sorted({mp[g] for g in gl if g in mp and mp[g] != ""})
        bg = conv(names)
        for m, ms in mods.items():
            n_sig, mapped, top = 0, 0, ""
            for k, mod in enumerate(ms, 1):
                genes = conv(mod)
                mapped += len(genes)
                if len(genes) >= fge.MIN_CONVERTED:
                    df = fge.enrich(genes, bg, gmt)
                    if df is not None:
                        n_sig += len(df)
                        top = top or df.iloc[0]["Term"]
            mg = sum(len(x) for x in ms)
            rows.append([r, len(bg), m, len(ms), mg, mapped,
                         f"{100 * mapped / mg:.1f}", n_sig, top])
            print(f"{r:<11} bg {len(bg):>5}  {m:<9} mapped {mapped}/{mg}  "
                  f"terms {n_sig:>3}  {top[:50]}")
    with open(os.path.join(out, f"orthologue_rules_det{int(thr * 100)}.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "background", "estimator", "modules", "module_genes",
                    "mapped", "mapping_rate_pct", "significant_terms",
                    "leading_term"])
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--min-detection", type=float, default=0.10)
    ap.add_argument("--gmt", default=os.path.join(
        REPO, "data", "GO_Biological_Process_2023.gmt"))
    a = ap.parse_args()
    main(a.min_detection, a.gmt)
