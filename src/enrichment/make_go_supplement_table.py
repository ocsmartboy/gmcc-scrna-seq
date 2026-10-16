r"""
Build the per-term GO enrichment table for the Supplementary Material.

Reads the PBMC 3K files *_hvgbg.csv in results/go_enrichment/ and the
Paul15 files Paul15_one2one_*.csv in
results/go_enrichment_orthologue_sensitivity/ (one-to-one orthologues, the
primary rule), and writes a LaTeX longtable to
results/go_enrichment/supplementary_go_terms.tex (Supplementary Table S6).

Why the Overlap column is reconstructed
---------------------------------------
gp.enrich() (the local hypergeometric test used with a custom background)
does not return an "Overlap" column, unlike the Enrichr web API. Overlap
is therefore rebuilt as k/K:
    k = number of module genes annotated to the term (from the "Genes" column)
    K = number of background genes annotated to the term
K is obtained by intersecting each term in the GO library with the exact
background used for enrichment (2,000 HVGs for PBMC 3K; the 1,610 human
one-to-one orthologues of the 2,000 HVGs for Paul15).

Verification: the hypergeometric p-value is recomputed from (N, K, n, k)
and compared with the P-value in the CSV. For PBMC 3K, n is the module size
in the file name; for Paul15, n is the number of mapped module genes taken
from orthologue_rule_modules.csv. Agreement confirms that K was
reconstructed correctly.

Output macros: \GOterm{digits} and \geneList{...} must be defined in the
supplementary preamble, e.g.
    \newcommand{\GOterm}[1]{{\scriptsize(GO:#1)}}
    \newcommand{\geneList}[1]{{\scriptsize\itshape #1}}

Usage (from anywhere; needs internet on first run for the GO library):
    python src/enrichment/make_go_supplement_table.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import scanpy as sc
import gseapy as gp
from scipy.stats import hypergeom

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SAVE_DIR = os.path.join(REPO, "data", "processed")
GO_DIR = os.path.join(REPO, "results", "go_enrichment")
ORTH_DIR = os.path.join(REPO, "results", "go_enrichment_orthologue_sensitivity")
OUT = os.path.join(GO_DIR, "supplementary_go_terms.tex")

GENE_SETS = "GO_Biological_Process_2023"
ESTIMATOR_ORDER = ["GMCC", "Pearson", "Cosine", "Spearman"]
DATASET_ORDER = ["PBMC 3K", "Paul15"]
PATTERN = re.compile(
    r"^(?:(Paul15)_)?([A-Za-z\-]+)_module(\d+)_(\d+)genes_hvgbg\.csv$")
PATTERN_PAUL = re.compile(
    r"^(Paul15)_one2one_([A-Za-z\-]+)_module(\d+)_(\d+)genes\.csv$")


# ── Backgrounds: rebuilt exactly as in the enrichment scripts ────────
def hvg_names(loader):
    adata = loader()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    return list(adata[:, adata.var.highly_variable].var_names)


print("Rebuilding enrichment backgrounds...")
bg_pbmc = set(hvg_names(sc.datasets.pbmc3k))
print(f"  PBMC 3K background: {len(bg_pbmc)} genes")

orth_path = os.path.join(SAVE_DIR, "mouse_human_orthologs_typed.csv")
if not os.path.exists(orth_path):
    raise SystemExit(f"Typed ortholog table not found: {orth_path}\n"
                     "Run src/enrichment/paul15_go_enrichment_one2one.py first.")
orth = pd.read_csv(orth_path)
orth = orth[orth["hsapiens_homolog_orthology_type"] == "ortholog_one2one"]
n_sym = orth.groupby("external_gene_name")[
    "hsapiens_homolog_associated_gene_name"].nunique()
orth = orth[orth["external_gene_name"].isin(n_sym.index[n_sym == 1])]
m2h = dict(zip(orth["external_gene_name"],
               orth["hsapiens_homolog_associated_gene_name"]))
bg_paul = {m2h[g] for g in hvg_names(sc.datasets.paul15)
           if g in m2h and m2h[g] != ""}
print(f"  Paul15 background:  {len(bg_paul)} genes (expected 1610)")

mods_path = os.path.join(ORTH_DIR, "orthologue_rule_modules.csv")
paul_mapped = {}
if os.path.exists(mods_path):
    mods = pd.read_csv(mods_path)
    mods = mods[mods["rule"] == "one2one"]
    paul_mapped = {(r.estimator, int(r.module)): int(r.mapped)
                   for r in mods.itertuples()}

BACKGROUND = {"PBMC 3K": bg_pbmc, "Paul15": bg_paul}

# ── GO library ───────────────────────────────────────────────────────
print(f"\nLoading {GENE_SETS}...")
library = gp.get_library(name=GENE_SETS, organism="Human")
library = {term: set(genes) for term, genes in library.items()}
print(f"  {len(library):,} terms")


def term_size_in_bg(term, bg):
    genes = library.get(term)
    return None if genes is None else len(genes & bg)


# ── Formatting helpers ───────────────────────────────────────────────
def tex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s


def split_term(term):
    m = re.match(r"^(.*?)\s*\(GO:(\d+)\)\s*$", str(term))
    return (m.group(1), m.group(2)) if m else (str(term), "")


def fmt_p(p):
    mant, exp = f"{p:.2e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


# ── Read CSVs ────────────────────────────────────────────────────────
rows, checks = [], []
files = sorted(f for f in glob.glob(os.path.join(GO_DIR, "*_hvgbg.csv"))
               if not os.path.basename(f).startswith("Paul15_"))
files += sorted(f for f in glob.glob(os.path.join(ORTH_DIR, "Paul15_one2one_*.csv"))
               if PATTERN_PAUL.match(os.path.basename(f)))   # excludes one2one_hc
if not files:
    raise SystemExit(f"No enrichment CSV files found in {GO_DIR} or {ORTH_DIR}")

print("\nReading enrichment results...")
for path in files:
    m = PATTERN.match(os.path.basename(path)) or \
        PATTERN_PAUL.match(os.path.basename(path))
    if not m:
        print(f"  skipped (name not recognised): {os.path.basename(path)}")
        continue
    dataset = "Paul15" if m.group(1) else "PBMC 3K"
    estimator, module, size = m.group(2), int(m.group(3)), int(m.group(4))
    bg = BACKGROUND[dataset]
    df = pd.read_csv(path).sort_values("Adjusted P-value")

    for _, r in df.iterrows():
        genes = [g for g in str(r["Genes"]).split(";") if g]
        k = len(genes)
        K = term_size_in_bg(r["Term"], bg)
        overlap = f"{k}/{K}" if K is not None else f"{k}/?"
        name, go_id = split_term(r["Term"])
        rows.append({
            "dataset": dataset, "estimator": estimator, "_m": module,
            "module": f"M{module} ({size})", "term": name, "go": go_id,
            "overlap": overlap, "padj": r["Adjusted P-value"],
            "genes": ", ".join(genes),
        })
        # p-value check: n = module size (PBMC 3K) or mapped genes (Paul15)
        n_test = size if dataset == "PBMC 3K" else \
            paul_mapped.get((estimator, module))
        if K is not None and n_test is not None:
            p_re = hypergeom.sf(k - 1, len(bg), K, n_test)
            p_re1 = hypergeom.sf(k - 1, len(bg), K + 1, n_test)
            checks.append((r["P-value"], p_re, p_re1,
                           f"{dataset} {estimator} M{module}: {name}"))

    print(f"  {dataset:<8} {estimator:<9} module {module} ({size} genes): "
          f"{len(df)} terms")

missing = sum(r["overlap"].endswith("/?") for r in rows)
if missing:
    print(f"\n  WARNING: {missing} terms not found in the library; "
          "their Overlap is shown as k/?")

# ── Verification ─────────────────────────────────────────────────────
if checks:
    obs = np.array([c[0] for c in checks])
    rec = np.array([c[1] for c in checks])
    rec1 = np.array([c[2] for c in checks])
    ok = np.isclose(obs, rec, rtol=1e-3, atol=1e-12)
    ok1 = ~ok & np.isclose(obs, rec1, rtol=1e-3, atol=1e-12)
    print(f"\nVerification: recomputed hypergeometric p-values "
          f"match the CSV for {ok.sum()}/{len(ok)} terms")
    for i in np.nonzero(ok1)[0]:
        print(f"  {checks[i][3]}: the CSV p-value ({obs[i]:.3e}) corresponds "
              "to K+1, i.e. gseapy counted one term gene twice (a duplicated "
              "entry in the library); Overlap reports the unique count K. "
              "The gseapy p-value is slightly conservative.")
    bad = ~ok & ~ok1
    if bad.any():
        worst = np.nonzero(bad)[0][np.argmax(np.abs(
            np.log10(obs[bad] + 1e-300) - np.log10(rec[bad] + 1e-300)))]
        print(f"  UNEXPLAINED mismatch for {bad.sum()} terms, e.g. "
              f"{checks[worst][3]}: CSV {obs[worst]:.3e} vs recomputed "
              f"{rec[worst]:.3e}")
        print("  -> K may not match the background gseapy used; "
              "do NOT use the Overlap column until resolved.")

# ── LaTeX longtable ──────────────────────────────────────────────────
tbl = pd.DataFrame(rows)
tbl["_d"] = tbl["dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)})
tbl["_e"] = tbl["estimator"].map({e: i for i, e in enumerate(ESTIMATOR_ORDER)})
tbl = tbl.sort_values(["_d", "_e", "_m", "padj"])

header = (r"Estimator & Module & GO term & Overlap & "
          r"Adj.\ $p$ & Overlap genes \\")
lines = [
    r"\begin{longtable}{@{}l l p{5.2cm} r r p{4.0cm}@{}}",
    r"\caption{All significant Gene Ontology Biological Process terms "
    r"(Benjamini--Hochberg adjusted $p<0.05$), tested against the genes "
    r"eligible for network construction as background. Module size in "
    r"genes is given in parentheses. Overlap is $k/K$, where $k$ is the "
    r"number of module genes annotated to the term and $K$ the number of "
    r"background genes annotated to it. For Paul15, module genes and "
    r"background are human one-to-one orthologues.}\label{tab:go-terms}\\",
    r"\toprule", header, r"\midrule", r"\endfirsthead",
    r"\multicolumn{6}{l}{\small\itshape Table~\thetable{} continued}\\",
    r"\toprule", header, r"\midrule", r"\endhead",
    r"\bottomrule", r"\endlastfoot",
]
for ds in DATASET_ORDER:
    sub = tbl[tbl["dataset"] == ds]
    if sub.empty:
        continue
    lines += [rf"\multicolumn{{6}}{{l}}{{\textit{{{ds}}} "
              rf"({len(sub)} terms)}} \\", r"\midrule"]
    prev = None
    for _, r in sub.iterrows():
        key = (r["estimator"], r["module"])
        if prev is not None and key != prev:
            lines.append(r"\addlinespace")
        est = r["estimator"] if key != prev else ""
        mod = r["module"] if key != prev else ""
        prev = key
        term = tex_escape(r["term"]) + (
            rf" \GOterm{{{r['go']}}}" if r["go"] else "")
        lines.append(f"{est} & {mod} & {term} & {r['overlap']} & "
                     f"{fmt_p(r['padj'])} & "
                     rf"\geneList{{{tex_escape(r['genes'])}}} \\")
    lines.append(r"\midrule")
if lines[-1] == r"\midrule":
    lines.pop()
lines.append(r"\end{longtable}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nTotal terms: {len(tbl)} (expected 182: PBMC 147+10+9, Paul15 9+7)")
for ds in DATASET_ORDER:
    for est in ESTIMATOR_ORDER:
        n = ((tbl["dataset"] == ds) & (tbl["estimator"] == est)).sum()
        if n:
            print(f"  {ds:<8} {est:<9} {n:>4}")
print(f"\nWritten: {OUT}")