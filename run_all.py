"""
Run the full analysis pipeline in manuscript order and archive the console
output of every step in results/logs/<step>.log.

Usage (from anywhere):
    python run_all.py                 # all steps
    python run_all.py --list          # show steps
    python run_all.py rank network    # selected steps only
    python run_all.py --from enrich_pbmc

Steps that print tables only to the console (e.g. Table 2, Table 3,
Supplementary Tables S3, S4 and S7) are reproducible from these logs.
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
    ("matrix_pbmc",   "src/similarity/stage2_similarity_2000hvg.py",
     "PBMC 3K matrix and global mean; Methods 2.1"),
    ("matrix_paul15", "src/similarity/stage2_similarity_paul15.py",
     "Paul15 matrix and global mean; Methods 2.1"),
    ("assumptions",   "src/analysis/verify_methods_assumptions.py",
     "Methods 2.3-2.6 checks (nested masking, realised levels, no ties, 99,950 edges)"),
    ("rank",          "src/stability/paired_difference_ci.py",
     "Table 1, Fig. 1 data, Suppl. Tables S1, S7 (avg tau), S8"),
    ("centring",      "src/analysis/mu_fixed_vs_recomputed.py",
     "Table 2"),
    ("defined_pairs", "src/stability/sensitivity_defined_pairs.py",
     "Suppl. Section S7, Table S9"),
    ("figures",       "src/figures/make_figures.py",
     "Figs. 1-2, Suppl. Table S2 (NMAC)"),
    ("network_pbmc",  "src/network/stage5_network_2000hvg.py",
     "Table 3 (PBMC 3K)"),
    ("network_paul15", "src/network/stage5_network_paul15.py",
     "Table 3 (Paul15)"),
    ("network_masking", "src/stability/network_topology_masking.py",
     "Suppl. Table S3"),
    ("threshold",     "src/analysis/threshold_sensitivity.py",
     "Suppl. Table S4"),
    ("degree",        "src/analysis/abundance_confounding.py",
     "Table 4 (degree vs detection rate)"),
    ("module_check",  "src/analysis/module_abundance_check.py",
     "module-level diagnostic (not tabulated)"),
    ("enrich_pbmc",   "src/enrichment/stage6_go_enrichment_2000hvg.py",
     "Table 5 (PBMC 3K)"),
    ("enrich_paul15", "src/enrichment/paul15_go_enrichment_one2one.py",
     "Table 5 (Paul15), Suppl. Tables S5, S10"),
    ("go_table",      "src/enrichment/make_go_supplement_table.py",
     "Suppl. Table S6"),
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
            [sys.executable, os.path.join(REPO, script)], cwd=REPO, env=env,
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
