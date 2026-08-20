# Global Mean-Centered Correlation for Gene Co-Expression Network Estimation in Sparse Single-Cell Transcriptomics

Code repository accompanying the manuscript:

> Wiyono, S., Adji, T. B., Nugroho, H. A., & Wibirama, S. "Global Mean-Centered Correlation for Gene Co-Expression Network Estimation in Sparse Single-Cell Transcriptomics." *Computer Methods and Programs in Biomedicine* (Elsevier), under review.

This work extends **Global Mean-Centered Correlation (GMCC)** — introduced in Wiyono et al. (2026), *Information Sciences* — to single-cell RNA sequencing (scRNA-seq) data, evaluating similarity/correlation estimator stability under sparsity (dropout) on two independent datasets.

## Overview

The pipeline evaluates five similarity/correlation measures (GMCC, Pearson, Cosine, Spearman, Bicor) across two scRNA-seq datasets:

- **PBMC 3K** (10x Genomics; human, peripheral blood; sparsity ≈ 91.4%)
- **Paul et al. 2015** (mouse bone marrow; sparsity ≈ 77.2%)

on five dimensions:
1. Rank stability under simulated dropout (Kendall's τ, multi-seed Wilcoxon signed-rank testing)
2. Gene co-expression network topology (clustering coefficient, connected components), with multi-seed stability testing
3. Biological validation (GO Biological Process enrichment via Enrichr, including mouse–human ortholog conversion for Paul et al.)
4. Cell clustering quality (Leiden clustering, Silhouette/ARI/NMI)
5. PCA-ablation study (GMCC with vs. without PCA preprocessing)

## Repository Structure

```
src/
├── preprocessing/    Data loading and standard scRNA-seq preprocessing (Scanpy)
├── similarity/        Vectorized implementations of all five similarity measures
├── stability/          Multi-seed dropout simulation and statistical testing
│                        (Kendall's tau and network clustering coefficient)
├── network/            Gene co-expression network construction and topology analysis
├── enrichment/         GO Biological Process enrichment (incl. mouse ortholog conversion)
├── clustering/         Cell clustering and PCA-ablation study
└── paul15_pipeline.py  Consolidated pipeline for the Paul et al. 2015 dataset
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12. No GPU required — all similarity computations are vectorized with NumPy/SciPy sparse operations.

## Usage

Datasets are downloaded automatically via `scanpy.datasets` on first run (no manual download needed). Run scripts in order from the repository root, e.g.:

```bash
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

Each script prints intermediate results to stdout and saves processed arrays to `data/processed/` (created automatically; not version-controlled — regenerate by re-running the scripts).

## Datasets

Both datasets are public benchmarks accessed via Scanpy's built-in loaders:
- `scanpy.datasets.pbmc3k()` — 10x Genomics PBMC 3K
- `scanpy.datasets.paul15()` — Paul et al. (2015), *Cell*, mouse bone marrow

## Citation

If you use this code, please cite both entries below.

**1. This work** (manuscript currently under review — DOI, volume, and page numbers will be added here upon acceptance):

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
  title   = {Global mean--centered correlation: A variance-reduced and theoretically grounded similarity measure for sparse data environments},
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

Slamet Wiyono — Department of Electrical and Information Engineering, Universitas Gadjah Mada, Yogyakarta, Indonesia
Corresponding author (Paper 1 & 2): Teguh Bharata Adji — adji@ugm.ac.id

Repository: `https://github.com/ocsmartboy/gmcc-scrna-seq`
