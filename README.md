# Robustness of Global Mean-Centered Correlation under Controlled Observed-Zero Sparsity in Single-Cell Transcriptomics

Code, intermediate mappings and results accompanying the manuscript

> Wiyono, S., Adji, T. B., Nugroho, H. A., & Wibirama, S. *Robustness of Global Mean-Centered Correlation under Controlled Observed-Zero Sparsity in Single-Cell Transcriptomics.* Submitted to *Computational Biology and Chemistry* (Elsevier).

The study evaluates whether **Global Mean-Centered Correlation (GMCC)** — introduced by Wiyono et al. (2026, *Information Sciences*) for collaborative filtering with missing entries — preserves gene–gene similarity rankings and magnitudes when detected expression entries in scRNA-seq data are masked to zero. GMCC is compared with Pearson correlation, Cosine similarity and Spearman rank correlation; a biweight midcorrelation variant with a standard-deviation fallback (Bicor-SD) is reported in the Supplementary Material only.

## Datasets

Both datasets are downloaded automatically by Scanpy on first use.

| Dataset | Loader | Organism / tissue | Cells | Sparsity after preprocessing | Global mean μ |
|---|---|---|---|---|---|
| PBMC 3K | `scanpy.datasets.pbmc3k()` (10x Genomics) | human, peripheral blood | 2,700 | 91.4% | 0.1776 |
| Paul15 | `scanpy.datasets.paul15()` (Paul et al., 2015; GEO GSE72857) | mouse, bone marrow | 2,730 | 77.2% | 0.4980 |

Preprocessing (identical for both): genes detected in ≥ 3 cells; total-count normalisation to 10⁴; `log1p`; 2,000 highly variable genes (Scanpy `seurat` flavour). The resulting gene-by-cell matrices are the common input to all estimators.

## Installation

```
pip install -r requirements.txt
```

Exact package versions used for the manuscript are listed in `requirements-lock.txt`. No GPU is required; all similarity matrices are computed with vectorised NumPy.

## Reproducing the results

Run everything in manuscript order (console output of every step is archived in `results/logs/`):

```
python run_all.py            # all steps
python run_all.py --list     # list steps and the manuscript items they produce
python run_all.py rank       # a single step
```

The full run can take several hours on a laptop, dominated by the 2,000 × 2,000 similarity matrices recomputed for 20 realisations at four masking levels. The exploratory PCA ablation (Supplementary Section S4) uses Scanpy's standard PCA–neighbour–Leiden pipeline, whose number of clusters can differ across Scanpy/igraph versions; the reported values were obtained with the versions in `requirements-lock.txt`. The matrix-building steps never overwrite existing matrices in `data/processed/`; they report whether a rebuilt matrix matches the stored one (use `--overwrite` to replace).

## Mapping to the manuscript

| Manuscript item | Script | Output |
|---|---|---|
| Datasets, μ, sparsity (Methods 2.1) | `src/similarity/stage2_similarity_2000hvg.py`, `src/similarity/stage2_similarity_paul15.py` | `data/processed/X_*.npy`, `global_mean_*.npy`; `results/logs/matrix_*.log` |
| Methods checks: nested masking, realised masking levels, no constant profiles in unperturbed matrices, 99,950 edges, positive MAD | `src/analysis/verify_methods_assumptions.py` | `results/logs/assumptions.log` |
| Table 1; Fig. 1 data; Suppl. Tables S1, S8 | `src/stability/paired_difference_ci.py` | `results/rank_robustness.csv`, `results/paired_tests.csv`, `data/processed/raw_kendall_tau_per_seed.npz` |
| Table 2 (retained vs recomputed global mean) | `src/analysis/mu_fixed_vs_recomputed.py` | `results/logs/centring.log` |
| Figs. 1–2; Suppl. Table S2 (NMAC) | `src/figures/make_figures.py` | `figures/Figure_1.*`, `figures/Figure_2.*`, `results/nmac.csv` |
| Table 3 (unperturbed networks) | `src/network/stage5_network_2000hvg.py`, `src/network/stage5_network_paul15.py` | `results/logs/network_*.log` |
| Suppl. Table S3 (clustering coefficient under masking) | `src/stability/network_topology_masking.py` | `data/processed/clustering_coefficient_masking.npz`, `results/logs/network_masking.log` |
| Suppl. Table S4 (90th / 95th / 97.5th percentile) | `src/analysis/threshold_sensitivity.py` | `results/logs/threshold.log` |
| Table 4 (degree vs detection rate) | `src/analysis/abundance_confounding.py` | `results/logs/degree.log` |
| Table 5, PBMC 3K enrichment | `src/enrichment/stage6_go_enrichment_2000hvg.py` | `results/go_enrichment/*_hvgbg.csv` |
| Table 5, Paul15 enrichment; Suppl. Tables S5, S10 | `src/enrichment/paul15_go_enrichment_one2one.py` | `results/go_enrichment_orthologue_sensitivity/` |
| Suppl. Table S6 (all significant GO terms) | `src/enrichment/make_go_supplement_table.py` | `results/go_enrichment/supplementary_go_terms.tex` |
| Suppl. Section S4, Fig. S1 (PCA ablation) | `src/clustering/stage7_clustering_2000hvg.py` | `figures/Figure_S1.png`, `results/logs/clustering.log` |
| Suppl. Section S6, Tables S7–S8 (Bicor-SD) | `paired_difference_ci.py`, `stage5_network_*.py`, `network_topology_masking.py`, `threshold_sensitivity.py`, `abundance_confounding.py`, `src/analysis/module_check_bicor.py`, `src/analysis/bicor_standard_check.py` | corresponding logs and CSVs |
| Suppl. Section S7, Table S9 (entirely zero profiles) | `src/stability/sensitivity_defined_pairs.py` | `results/sensitivity_coverage.csv`, `results/sensitivity_mean_tau.csv`, `results/sensitivity_paired_tests.csv` |

`src/analysis/module_abundance_check.py` is a module-level diagnostic that is not tabulated in the manuscript. `src/figures/make_graphical_abstract.py` draws an optional graphical abstract.

## Key implementation details

**Masking.** For each dataset and realisation, a uniform random value is drawn for every entry (`numpy.random.seed(seed)` followed by `numpy.random.rand`, seeds 1–20). Detected entries with values below m ∈ {0.1, 0.3, 0.5, 0.7} are set to zero. The same draws are used at every level, so masking is nested within a realisation; each masked matrix is generated directly from the unperturbed matrix.

**Global mean.** In the primary analysis, μ is recomputed from every analysed (masked) matrix. `src/analysis/mu_fixed_vs_recomputed.py` compares this with retaining μ from the unperturbed matrix.

**Undefined similarities.** When masking produces an entirely zero profile, Pearson, Spearman, Cosine and Bicor-SD are undefined for pairs involving it; these values are set to zero. GMCC remains defined. `sensitivity_defined_pairs.py` repeats the rank analysis on pairs defined for all estimators.

**Statistics.** Masking realisations are the units of inference. Paired differences in Kendall's τ_b (GMCC − comparator) are summarised by Hodges–Lehmann estimates with exact distribution-free 95% confidence intervals from the ordered Walsh averages, and tested with exact two-sided Wilcoxon signed-rank tests; Holm correction is applied across Pearson, Cosine and Spearman within each dataset and masking level (Bicor-SD p-values are unadjusted).

**Networks.** An edge is retained when the signed similarity strictly exceeds the 95th percentile of the estimator's upper-triangular similarity distribution (99,950 of 1,999,000 gene pairs in every unperturbed network). Modules are connected components with ≥ 10 genes; when a network forms a single component (Pearson and Spearman in PBMC 3K), it is partitioned by greedy modularity.

**Functional enrichment.** GO Biological Process (`GO_Biological_Process_2023`, Enrichr library) over-representation is tested with the local hypergeometric test in GSEApy against an explicit background: the 2,000 HVGs for PBMC 3K, and their human one-to-one orthologues (1,610 genes) for Paul15. Benjamini–Hochberg correction is applied within each module (adjusted p < 0.05). Cell clustering uses Leiden with `random_state=42`.

**Orthologue mapping (Paul15).** Mouse-to-human orthologues were retrieved from Ensembl BioMart (`mmusculus_gene_ensembl`, Ensembl release 116, accessed 14 October 2026) and only `ortholog_one2one` pairs are used. Archived tables in `data/processed/`:

| File | Content |
|---|---|
| `mouse_human_orthologs_typed.csv` | all mouse→human homologues with orthology type and confidence (primary analysis) |
| `mouse_human_orthologs_typed.json` | retrieval date, host and attributes |
| `mouse_human_orthologs.csv` | earlier two-column table used by the alternative "final entry" rule (Supplementary Table S10) |

`src/enrichment/paul15_go_enrichment_ortholog.py` implements that earlier rule and is retained for comparison only.

## Repository structure

```
run_all.py            pipeline runner (writes results/logs/)
src/
  similarity/         preprocessing and matrix construction (both datasets)
  stability/          masking, rank robustness, paired statistics, sensitivity analyses
  analysis/           centring reference, thresholds, degree-detection, Bicor diagnostics
  network/            unperturbed co-expression networks
  enrichment/         GO enrichment and the supplementary GO-term table
  clustering/         exploratory PCA ablation (Supplementary Section S4)
  figures/            manuscript figures
data/processed/       orthologue tables (tracked); matrices *.npy (regenerated, not tracked)
results/              tables underlying the manuscript and step logs
figures/              manuscript figures
archive/              superseded scripts (not used for the manuscript; see archive/README.md)
```

## Citation

If you use this code, please cite the manuscript above (full reference to be added upon publication) and the original GMCC article:

```bibtex
@article{wiyono2026gmcc,
  author  = {Wiyono, Slamet and Adji, Teguh Bharata and Nugroho, Hanung Adi and Wibirama, Sunu},
  title   = {Global mean-centered correlation: A variance-reduced and theoretically grounded similarity measure for sparse data environments},
  journal = {Information Sciences},
  volume  = {749},
  pages   = {123502},
  year    = {2026},
  doi     = {10.1016/j.ins.2026.123502}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Contact

Slamet Wiyono — Department of Electrical and Information Engineering, Universitas Gadjah Mada, Yogyakarta, Indonesia; Faculty of Science and Technology, Universitas Harkat Negeri, Tegal, Indonesia.
Corresponding author: Teguh Bharata Adji — adji@ugm.ac.id
