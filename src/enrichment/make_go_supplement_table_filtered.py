"""
Supplementary Table S6: all significant GO Biological Process terms for the
community modules of the networks of genes detected in at least 10% of cells.

Input : results/filtered_go_enrichment_community/<dataset>_det10_<estimator>_module<k>_<n>genes.csv
        (written by filtered_go_enrichment.py --modules community)
Output: results/filtered_go_enrichment_community/supplementary_go_terms_det10.tex

Usage
-----
    python src/enrichment/make_go_supplement_table_filtered.py
"""
import glob
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
IN_DIR = os.path.join(REPO, "results", "filtered_go_enrichment_community")
OUT = os.path.join(IN_DIR, "supplementary_go_terms_det10.tex")
DATASETS = {"PBMC3K": "PBMC~3K", "Paul15": "Paul15"}
ESTIMATORS = ["GMCC", "Pearson", "Cosine", "Spearman"]


def esc(t):
    return t.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def main():
    body, total = "", 0
    for ds, label in DATASETS.items():
        files = glob.glob(os.path.join(IN_DIR, f"{ds}_det10_*_module*genes.csv"))
        block, n = "", 0
        for est in ESTIMATORS:
            mine = sorted((f for f in files if f"_{est}_" in f),
                          key=lambda f: int(re.search(r"module(\d+)", f).group(1)))
            for fp in mine:
                k, size = re.search(r"module(\d+)_(\d+)genes", fp).groups()
                d = pd.read_csv(fp).sort_values("Adjusted P-value")
                for i, (_, r) in enumerate(d.iterrows()):
                    g = re.search(r"\(GO:(\d+)\)", r["Term"])
                    name = (esc(r["Term"][:g.start()].strip()) + f" \\GOterm{{{g.group(1)}}}"
                            if g else esc(r["Term"]))
                    mant, ex = f"{r['Adjusted P-value']:.2e}".split("e")
                    genes = ", ".join(r["Genes"].split(";"))
                    block += (f"{est if i == 0 else ''} & {f'M{k} ({size})' if i == 0 else ''} & "
                              f"{name} & {r['Overlap']} & ${mant}\\times10^{{{int(ex)}}}$ & "
                              f"\\geneList{{{genes}}} \\\\\n")
                    n += 1
        body += (f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\textit{{{label} ({n} terms)}}}} \\\\\n"
                 f"\\midrule\n{block}")
        total += n
        print(f"{ds}: {n} terms")
    head = r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\begin{longtable}{@{}l l p{4.6cm} r r p{3.6cm}@{}}
\caption{All significant Gene Ontology Biological Process terms (Benjamini--Hochberg adjusted $p<0.05$) for the community modules of the networks of genes detected in at least 10\% of cells. Module size in genes is given in parentheses. Overlap is $k/K$, where $k$ is the number of module genes annotated to the term and $K$ the number of background genes annotated to it. For Paul15, module genes and background are human orthologues.}\label{tab:go-terms}\\
\toprule
Estimator & Module & GO term & Overlap & Adj.\ $p$ & Overlap genes \\
\endfirsthead
\toprule
Estimator & Module & GO term & Overlap & Adj.\ $p$ & Overlap genes \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    with open(OUT, "w") as f:
        f.write(head + body + "\\end{longtable}\n\\endgroup\n")
    print(f"Total terms: {total}\nSaved: {OUT}")


if __name__ == "__main__":
    main()
