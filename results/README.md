# Results

Tables, per-condition summaries and console logs underlying the manuscript.
Run `python run_all.py --list` to see which step produces which item.

| File / folder | Content |
|---|---|
| `rank_robustness.csv` | Mean (SD) Kendall's τ_b per masking level, average τ_b and relative decrease (Table 1) |
| `paired_tests.csv` | Hodges–Lehmann estimates, exact 95% CIs, Wilcoxon W and p-values (Supplementary Table S1; Bicor-SD in Table S8) |
| `nmac.csv` | NMAC mean and sample SD (Supplementary Table S2) |
| `sensitivity_*.csv`, `sensitivity_tau_per_seed.npz` | Analysis restricted to gene pairs defined for all estimators (Supplementary Section S7, Table S9) |
| `low_detection_control/` | Kendall's τ_b within detection-rate pair sets (Table 3; Supplementary Tables S11–S13), shared-zero correlations, edge co-detection of the 2,000-gene networks |
| `detection_filter/` | Rank and magnitude robustness on genes detected in ≥ 1%, 5% and 10% of cells (Supplementary Table S14) |
| `filtered_networks/` | Network characteristics, degree–detection correlations and edge co-detection for all 2,000 genes and for genes detected in ≥ 5% and ≥ 10% of cells (Tables 4–5; Supplementary Tables S3–S4, S15–S16) |
| `filtered_go_enrichment_community/` | GO enrichment of community modules (Table 6; Supplementary Tables S5–S6, S17) |
| `filtered_go_enrichment/` | GO enrichment under the component-based module rule (Supplementary Table S17) |
| `filtered_orthologue_rules/` | Paul15 enrichment under three orthologue rules (Supplementary Table S10) |
| `go_enrichment/`, `go_enrichment_orthologue_sensitivity/` | Enrichment of the 2,000-gene networks from an earlier version of the manuscript |
| `logs/` | Console output of each pipeline step, including tables printed only to the console (Table 2; Supplementary Table S7) |

## Provenance

The numerical steps of the 14 October 2026 version (`matrix_*`, `assumptions`, `rank`, `centring`, `defined_pairs`, `figures`, `network_*`, `threshold`, `degree`, `module_check`, `bicor_*`) were re-run from the archived matrices in an independent Linux environment (Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, Scanpy 1.12.4, NetworkX 3.6.1). Their outputs reproduce the values reported in the manuscript.

The gene-detection, filtered-network and filtered-enrichment steps (`low_detection*`, `edge_codetection`, `detection_filter`, `filtered_*`, `go_table`) were run on 22 October 2026 in the same environment, with GSEApy 1.2.1 and the local GO library `data/GO_Biological_Process_2023.gmt`. With all 2,000 genes, these scripts reproduce the earlier Tables 3–4 exactly, and 181 of the 182 earlier significant GO terms; the one additional term (Spearman, PBMC 3K, adjusted p = 0.046) reflects a difference between library downloads (Supplementary Section S10).

The earlier 2,000-gene enrichment steps (`enrich_pbmc`, `enrich_paul15`, `go_table_2000`) require internet access to the Enrichr gene-set library, and the exploratory PCA-ablation step (`clustering`) depends on the Scanpy/igraph versions used. These were run in the authors' environment (`requirements-lock.txt`).
