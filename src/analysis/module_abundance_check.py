import os
import sys

import numpy as np
import networkx as nx
import scanpy as sc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SAVE_DIR = os.path.join(REPO, "data", "processed")

PERCENTILE = 95
MIN_MODULE_SIZE = 10


def gmcc_vectorized(X, mu):
    X_c = X - mu
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_c / norms) @ (X_c / norms).T, -1.0, 1.0)


def build_network_percentile(sim, gene_names, percentile=PERCENTILE):
    idx = np.triu_indices(len(gene_names), k=1)
    upper = sim[idx]
    threshold = np.percentile(upper, percentile)
    G = nx.Graph()
    G.add_nodes_from(gene_names)
    rows, cols = idx
    mask = upper > threshold
    G.add_edges_from(zip(
        [gene_names[i] for i in rows[mask]],
        [gene_names[j] for j in cols[mask]],
    ))
    return G


def extract_modules(G, min_size=MIN_MODULE_SIZE, max_modules=5):
    """Identical logic to module_sizes.py: connected components if the graph
    is fragmented, otherwise greedy modularity communities."""
    n_comp = nx.number_connected_components(G)
    if n_comp > 1:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        modules = [list(c) for c in components if len(c) >= min_size]
    else:
        communities = nx.community.greedy_modularity_communities(G)
        modules = sorted([list(c) for c in communities if len(c) >= min_size],
                          key=len, reverse=True)
    return modules[:max_modules]


def load(name):
    path = os.path.join(SAVE_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found.")
    return np.load(path)


def main():
    print("=" * 74)
    print("MODULE-LEVEL ABUNDANCE CHECK (PBMC 3K, GMCC)")
    print("=" * 74)

    X = load("X_2000hvg.npy")
    mu = load("global_mean_2000hvg.npy")[0]
    detection_rate = (X > 0).mean(axis=1)

    print("\nRebuilding gene names and network (must match Stage 6 exactly)...")
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    gene_names = list(adata.var_names)

    detection_by_gene = dict(zip(gene_names, detection_rate))

    sim = gmcc_vectorized(X, mu)
    G = build_network_percentile(sim, gene_names)
    modules = extract_modules(G)

    overall_median = np.median(detection_rate)
    overall_q1, overall_q3 = np.quantile(detection_rate, [0.25, 0.75])
    print(f"\nDataset-wide detection rate: median={overall_median:.4f}, "
          f"IQR=[{overall_q1:.4f}, {overall_q3:.4f}]")

    print(f"\nFound {len(modules)} modules (>= {MIN_MODULE_SIZE} genes):\n")
    print(f"{'Module':<8}{'Size':>6}{'Median DR':>12}{'IQR low':>10}"
          f"{'IQR high':>10}{'% below dataset Q1':>20}")
    print("-" * 66)

    summary = []
    for i, mod in enumerate(modules, 1):
        drs = np.array([detection_by_gene[g] for g in mod])
        med = np.median(drs)
        q1, q3 = np.quantile(drs, [0.25, 0.75])
        pct_low = 100 * (drs < overall_q1).mean()
        print(f"{i:<8}{len(mod):>6}{med:>12.4f}{q1:>10.4f}{q3:>10.4f}"
              f"{pct_low:>19.1f}%")
        summary.append((i, len(mod), med, pct_low))

    print("\nInterpretation guide:")
    print("  - A module whose median detection rate is close to the dataset")
    print("    median, with a low '% below dataset Q1', looks like ordinary")
    print("    genes -- consistent with genuine co-expression (scenario A).")
    print("  - A module whose median detection rate sits well below the")
    print("    dataset Q1, with most members in the bottom quartile, overlaps")
    print("    the low-detection cluster identified by")
    print("    abundance_confounding.py (scenario B) -- its GO enrichment")
    print("    should be re-examined before being reported as biological.")
    print("\nCross-reference module sizes above against the enrichment results")
    print("in Table 6 (module sizes 589, 92, 13) to identify which module is")
    print("which.")


if __name__ == "__main__":
    main()