# Global Mean-Centered Correlation for Gene Co-Expression Network Estimation in Sparse Single-Cell Transcriptomics

Code repository accompanying the manuscript:

> Wiyono, S., Adji, T. B., Nugroho, H. A., & Wibirama, S. "Global Mean-Centered Correlation for Gene Co-Expression Network Estimation in Sparse Single-Cell Transcriptomics." *Computer Methods and Programs in Biomedicine* (Elsevier), under review.

This work extends **Global Mean-Centered Correlation (GMCC)** — introduced in Wiyono et al. (2026), *Information Sciences* — to single-cell RNA sequencing (scRNA-seq) data, evaluating similarity/correlation estimator stability under sparsity (dropout) on two independent datasets.

## Overview

The pipeline evaluates five similarity/correlation measures across two scRNA-seq datasets:

| Measure | Centering reference |
|---|---|
| GMCC | Single dataset-level global mean |
| Pearson (PCC) | Local arithmetic mean (per gene pair) |
| Cosine | None |
| Spearman | Local mean of ranks |
| **Bicor-SD** | Local median + MAD, **with standard-deviation fallback** |

**Note on Bicor-SD.** Under the sparsity levels evaluated here, the median absolute deviation (MAD) equals zero for 97.4% of gene profiles in PBMC 3K and 86.9% in Paul15, leaving standard biweight midcorrelation undefined for 99.93% and 98.28% of gene pairs respectively. All results reported as "Bicor-SD" therefore use a fallback variant in which the local standard deviation replaces MAD when MAD equals zero. **Bicor-SD is a distinct estimator from standard Bicor** and should not be interpreted as the WGCNA-standard implementation. See `src/analysis/bicor_standard_check.py` and `src/analysis/module_check_bicor.py` for the scripts that quantify this.

Datasets:

- **PBMC 3K** (10x Genomics; human, peripheral blood; sparsity ≈ 91.4%, global mean μ = 0.1776)
- **Paul et al. 2015** (mouse bone marrow; sparsity ≈ 77.2%, global mean μ = 0.4980)

Evaluation dimensions:

1. Rank stability under simulated dropout (Kendall's τ; 20 seeds per dropout level, Hodges–Lehmann estimators with distribution-free 95% CIs, exact Wilcoxon signed-rank tests with Holm correction)
2. Gene co-expression network topology (clustering coefficient, connected components) across datasets, dropout realizations, and percentile thresholds
3. Biological validation (GO Biological Process enrichment via Enrichr, including mouse–human ortholog conversion for Paul15)
4. Cell clustering quality (Leiden clustering; Silhouette, ARI, NMI)
5. PCA-ablation study (GMCC with vs. without PCA preprocessing)

## Repository Structure

```
src/
├── preprocessing/   Data loading and standard scRNA-seq preprocessing (Scanpy)
├── similarity/      Vectorized implementations of all five similarity measures
├── stability/       Multi-seed dropout simulation and statistical testing
├── network/         Gene co-expression network construction and topology analysis
├── enrichment/      GO Biological Process enrichment (incl. ortholog conversion)
├── clustering/      Cell clustering and PCA-ablation study
├── analysis/        Supplementary analyses (threshold sensitivity, module sizes,
│                    Bicor degeneracy checks)
├── figures/         Figure generation for the manuscript
└── paul15_pipeline.py   Consolidated pipeline for the Paul et al. 2015 dataset

figures/             Generated figures (manuscript and exploratory)
data/processed/      Intermediate arrays (not version-controlled; regenerate by
                     re-running the scripts)
```

## Installation

```
pip install -r requirements.txt
```

Tested with Python 3.12. No GPU required — all similarity computations are vectorized with NumPy.

## Usage

Datasets are downloaded automatically via `scanpy.datasets` on first run (no manual download needed). Run scripts from the repository root.

**Main pipeline:**

```
python src/preprocessing/stage1_preprocessing.py
python src/similarity/stage2_similarity_2000hvg.py
python src/stability/stage3_4_multiseed_with_topology.py
python src/network/stage5_network_2000hvg.py
python src/enrichment/stage6_go_enrichment_2000hvg.py
python src/clustering/stage7_clustering_2000hvg.py
python src/paul15_pipeline.py
python src/network/stage5_network_paul15.py
python src/enrichment/paul15_go_enrichment_ortholog.py
```

**Statistical analysis and supplementary checks** (required to reproduce several
values reported in the manuscript):

```
python src/stability/paired_difference_ci.py      # Hodges-Lehmann estimates + 95% CIs
python src/analysis/threshold_sensitivity.py      # P90 / P95 / P97.5 comparison
python src/analysis/module_sizes.py               # module sizes per method
python src/analysis/module_check_bicor.py         # MAD = 0 rate per gene profile
python src/analysis/bicor_standard_check.py       # undefined rate per gene pair
```

**Figure generation:**

```
python src/figures/make_figures.py                # Figures 1 and 2
```

`make_figures.py` reads the per-seed Kendall's τ values written by
`paired_difference_ci.py` and derives every plotted quantity from them; no values
are hardcoded. Run `paired_difference_ci.py` first.

Each script prints intermediate results to stdout and saves processed arrays to
`data/processed/` (created automatically; not version-controlled).

## Mapping to the Manuscript

| Manuscript item | Script |
|---|---|
| Table 1 (average Kendall's τ, degradation) | `src/stability/stage3_4_multiseed_with_topology.py` |
| Table 2 (Hodges–Lehmann paired differences) | `src/stability/paired_difference_ci.py` |
| Table 3 (network topology, 95th percentile) | `src/network/stage5_network_2000hvg.py`, `src/network/stage5_network_paul15.py` |
| Table 4 (cross-dataset gap under dropout) | `src/stability/stage3_4_multiseed_with_topology.py` |
| Table 5 (threshold sensitivity) | `src/analysis/threshold_sensitivity.py` |
| Table 6 (GO enrichment, PBMC 3K) | `src/enrichment/stage6_go_enrichment_2000hvg.py`, `src/analysis/module_sizes.py` |
| Table 7 (GO enrichment, Paul15) | `src/enrichment/paul15_go_enrichment_ortholog.py` |
| Figure 1 (stability trajectories) | `src/figures/make_figures.py` |
| Figure 2 (GMCC vs. Cosine difference) | `src/figures/make_figures.py` |
| Figure 3 (PCA ablation, UMAP) | `src/clustering/stage7_clustering_2000hvg.py` |
| MAD = 0 rates (Section 2.3) | `src/analysis/module_check_bicor.py` |
| Undefined gene-pair rates (Section 2.3) | `src/analysis/bicor_standard_check.py` |

The `figures/` directory also contains exploratory plots that are not included in
the manuscript.

## Reproducibility Chain for Figures 1–2

```
scanpy.datasets.pbmc3k() / paul15()
  -> src/preprocessing, src/similarity        (preprocessed matrices)
  -> src/stability/paired_difference_ci.py    (per-seed Kendall's tau)
  -> data/processed/raw_kendall_tau_per_seed.npz
  -> src/figures/make_figures.py
  -> figures/Figure_1.{pdf,png}, figures/Figure_2.{pdf,png}
```

## Datasets

Both datasets are public benchmarks accessed via Scanpy's built-in loaders:

- `scanpy.datasets.pbmc3k()` — 10x Genomics PBMC 3K
- `scanpy.datasets.paul15()` — Paul et al. (2015), *Cell*, mouse bone marrow (GEO: GSE72857)

## Citation

If you use this code, please cite both entries below.

**1. This work** (manuscript currently under review — DOI, volume, and page numbers will be added upon acceptance):

```bibtex
@article{wiyono2026gmccscrna,
  author  = {Wiyono, Slamet and Adji, Teguh Bharata and Nugroho, Hanung Adi and Wibirama, Sunu},
  title   = {Global Mean-Centered Correlation for Gene Co-Expression Network Estimation in Sparse Single-Cell Transcriptomics},
  journal = {Computer Methods and Programs in Biomedicine},
  year    = {2026},
  note    = {Manuscript under review; full citation to be updated upon acceptance}
}
```

**2. The foundational GMCC method** (already published; introduces and theoretically establishes GMCC on recommender-system data):

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

Slamet Wiyono — Department of Electrical and Information Engineering, Universitas Gadjah Mada, Yogyakarta, Indonesia.

Corresponding author: Teguh Bharata Adji — <adji@ugm.ac.id>
