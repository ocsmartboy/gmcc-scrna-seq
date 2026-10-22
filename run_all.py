"""
Run the full analysis pipeline in manuscript order and archive the console
output of every step in results/logs/<step>.log.

Usage (from anywhere):
    python run_all.py                 # all steps
    python run_all.py --list          # show steps
    python run_all.py rank network    # selected steps only
    python run_all.py --from enrich_pbmc

Steps that print tables only to the console (e.g. Table 2 and
Supplementary Table S7) are reproducible from these logs.
Expected total runtime on a laptop: several hours (dominated by the
2,000 x 2,000 similarity computations over 20 realisations x 4 levels).
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO, "results", "logs")

# (step name, script, manuscript items)
STEPS = [
    # ── Matrices and primary rank / magnitude robustness (2,000 genes) ──
    ("matrix_pbmc",   "src/similarity/stage2_similarity_2000hvg.py",
     "PBMC 3K matrix and global mean; Methods 2.1"),
    ("matrix_paul15", "src/similarity/stage2_similarity_paul15.py",
     "Paul15 matrix and global mean; Methods 2.1"),
    ("assumptions",   "src/analysis/verify_methods_assumptions.py",
     "Methods checks (nested masking, realised levels, no ties, 99,950 edges)"),
    ("rank",          "src/stability/paired_difference_ci.py",
     "Table 1, Fig. 1 data, Suppl. Table S1"),
    ("centring",      "src/analysis/mu_fixed_vs_recomputed.py",
     "Table 2"),
    ("defined_pairs", "src/stability/sensitivity_defined_pairs.py",
     "Suppl. Section S7, Table S9"),
    ("figures",       "src/figures/make_figures.py",
     "Figs. 1-2, Suppl. Table S2 (NMAC)"),
    # ── Dependence on gene detection (Section 3.3, Suppl. Section S9) ──
    ("low_detection", "src/analysis/low_detection_control.py",
     "Table 3, Suppl. Tables S11-S12, shared-zero correlations"),
    ("low_detection_q", "src/analysis/low_detection_control.py --split quartile",
     "Suppl. Tables S11, S13 (quartile split)"),
    ("edge_codetection", "src/analysis/edge_codetection_check.py",
     "Section 3.3: co-detection of the 2,000-gene network edges"),
    ("detection_filter", "src/analysis/detection_filter_analysis.py",
     "Suppl. Table S14 (genes detected in >=1/5/10% of cells)"),
    # ── Networks and enrichment of adequately detected genes ──
    ("filtered_networks", "src/network/filtered_networks.py",
     "Tables 4-5, Suppl. Tables S3-S4, S15-S16"),
    ("filtered_enrich_comm", "src/enrichment/filtered_go_enrichment.py --modules community",
     "Table 6, Suppl. Tables S5, S17 (needs data/GO_Biological_Process_2023.gmt)"),
    ("filtered_enrich", "src/enrichment/filtered_go_enrichment.py",
     "Suppl. Table S17 (component-based module rule)"),
    ("filtered_orthologue", "src/enrichment/filtered_orthologue_rules.py",
     "Suppl. Section S8, Tables S5, S10"),
    ("go_table",      "src/enrichment/make_go_supplement_table_filtered.py",
     "Suppl. Table S6"),
    # ── 2,000-gene networks (earlier version; Bicor-SD statistics) ──
    ("network_pbmc",  "src/network/stage5_network_2000hvg.py",
     "2,000-gene network, PBMC 3K (earlier version; Suppl. Table S7)"),
    ("network_paul15", "src/network/stage5_network_paul15.py",
     "2,000-gene network, Paul15 (earlier version; Suppl. Table S7)"),
    ("network_masking", "src/stability/network_topology_masking.py",
     "2,000-gene clustering under masking, incl. Bicor-SD (Suppl. Table S7)"),
    ("threshold",     "src/analysis/threshold_sensitivity.py",
     "2,000-gene percentile sensitivity, incl. Bicor-SD (Suppl. Table S7)"),
    ("degree",        "src/analysis/abundance_confounding.py",
     "2,000-gene degree vs detection rate (Suppl. Table S7)"),
    ("module_check",  "src/analysis/module_abundance_check.py",
     "module-level diagnostic (not tabulated)"),
    ("enrich_pbmc",   "src/enrichment/stage6_go_enrichment_2000hvg.py",
     "2,000-gene enrichment, PBMC 3K (earlier version; needs Enrichr access)"),
    ("enrich_paul15", "src/enrichment/paul15_go_enrichment_one2one.py",
     "2,000-gene enrichment, Paul15 (earlier version; needs Enrichr access)"),
    ("go_table_2000", "src/enrichment/make_go_supplement_table.py",
     "term table of the earlier 2,000-gene version"),
    ("bicor_mad",     "src/analysis/module_check_bicor.py",
     "Suppl. Section S6 (MAD = 0 rates)"),
    ("bicor_undef",   "src/analysis/bicor_standard_check.py",
     "Suppl. Section S6 (undefined pairs)"),
    ("clustering",    "src/clustering/stage7_clustering_2000hvg.py",
     "Suppl. Section S4, Fig. S1"),
]


def run_step(name, script):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    print(f"\n{'#' * 72}\n# {name}: {script}\n{'#' * 72}", flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(REPO, script.split()[0])] + script.split()[1:], cwd=REPO, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    status = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(f"--> {name}: {status} in {(time.time() - t0) / 60:.1f} min; "
          f"log: {os.path.relpath(log_path, REPO)}", flush=True)
    return proc.returncode == 0


def main():
    names = [s[0] for s in STEPS]
    ap = argparse.ArgumentParser(description="Run the analysis pipeline.")
    ap.add_argument("steps", nargs="*", help=f"subset of: {', '.join(names)}")
    ap.add_argument("--from", dest="start", choices=names,
                    help="start from this step")
    ap.add_argument("--list", action="store_true", help="list steps and exit")
    a = ap.parse_args()

    if a.list:
        for n, script, items in STEPS:
            print(f"{n:<16} {script:<52} {items}")
        return

    unknown = [s for s in a.steps if s not in names]
    if unknown:
        sys.exit(f"Unknown step(s): {unknown}. Use --list.")
    selected = [s for s in STEPS if (not a.steps or s[0] in a.steps)]
    if a.start:
        selected = selected[names.index(a.start):] if not a.steps else selected

    failed = [n for n, script, _ in selected if not run_step(n, script)]
    print("\nSummary:", "all steps succeeded" if not failed
          else f"failed steps: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
