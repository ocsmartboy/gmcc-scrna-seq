# Results

Tables, per-condition summaries and console logs underlying the manuscript.
Run `python run_all.py --list` to see which step produces which item.

| File | Content |
|---|---|
| `rank_robustness.csv` | Mean (SD) Kendall's τ_b per masking level, average τ_b and relative decrease (Table 1; Supplementary Table S7) |
| `paired_tests.csv` | Hodges–Lehmann estimates, exact 95% CIs, Wilcoxon W and p-values (Supplementary Tables S1, S8) |
| `nmac.csv` | NMAC mean and sample SD (Supplementary Table S2) |
| `sensitivity_*.csv`, `sensitivity_tau_per_seed.npz` | Analysis restricted to gene pairs defined for all estimators (Supplementary Section S7, Table S9) |
| `go_enrichment/` | Significant GO terms per module (*_hvgbg.csv, PBMC 3K) and Supplementary Table S6 |
| `go_enrichment_orthologue_sensitivity/` | Paul15 enrichment under three orthologue rules (Table 5; Supplementary Tables S5, S10) |
| `logs/` | Console output of each pipeline step, including tables printed only to the console (Table 2, 3, 4; Supplementary Tables S3, S4, S7) |

## Provenance

The numerical steps (`matrix_*`, `assumptions`, `rank`, `centring`, `defined_pairs`, `figures`, `network_*`, `threshold`, `degree`, `module_check`, `bicor_*`) were re-run on 14 October 2026 from the archived matrices in an independent Linux environment (Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, Scanpy 1.12.4, NetworkX 3.6.1). Their outputs reproduce the values reported in the manuscript.

The enrichment steps (`enrich_pbmc`, `enrich_paul15`, `go_table`) require internet access to the Enrichr gene-set library, and the exploratory PCA-ablation step (`clustering`) depends on the Scanpy/igraph versions used. These were run in the authors' environment (`requirements-lock.txt`).
