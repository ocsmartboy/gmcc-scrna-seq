"""
Stage 2 (Paul15): build the preprocessed 2,000-HVG gene-by-cell matrix.

Writes
    data/processed/X_paul15.npy             (2,000 genes x 2,730 cells)
    data/processed/global_mean_paul15.npy   (matrix-level mean, mu)

Preprocessing is identical to PBMC 3K (stage2_similarity_2000hvg.py) and to
the gene list rebuilt inside the Paul15 network and enrichment scripts:
genes detected in >= 3 cells, total-count normalisation to 1e4, log1p,
2,000 highly variable genes (Scanpy "seurat" flavour).

Safety: if the two output files already exist, they are NOT overwritten.
The script instead compares the newly built matrix with the stored one and
reports whether they are identical. Use --overwrite to replace them.

Expected values (manuscript): sparsity 77.2%, mu = 0.4980.

Usage (from anywhere):
    python src/similarity/stage2_similarity_paul15.py
    python src/similarity/stage2_similarity_paul15.py --overwrite
"""
import argparse
import os

import numpy as np
import scanpy as sc
from scipy.sparse import issparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SAVE_DIR = os.path.join(REPO, "data", "processed")
X_PATH = os.path.join(SAVE_DIR, "X_paul15.npy")
MU_PATH = os.path.join(SAVE_DIR, "global_mean_paul15.npy")


def build_matrix():
    adata = sc.datasets.paul15()
    n_cells, n_genes_raw = adata.shape
    X_raw = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    sparsity_raw = (X_raw == 0).mean() * 100

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    # Identical to the original Paul15 pipeline (archive/paul15_pipeline.py):
    # keep the AnnData dtype and compute mu with X.mean() on that dtype.
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X = X.T                                # (2,000 genes x cells)
    mu = X.mean()
    info = {
        "cells": n_cells,
        "genes_raw": n_genes_raw,
        "sparsity_raw": sparsity_raw,
        "genes_hvg": X.shape[0],
        "sparsity_hvg": (X == 0).mean() * 100,
        "mu": float(mu),
    }
    return X, mu, info


def main(overwrite):
    print("=" * 64)
    print("STAGE 2 (Paul15): PREPROCESSED 2,000-HVG MATRIX")
    print("=" * 64)
    X, mu, info = build_matrix()
    print(f"  Cells                          : {info['cells']}")
    print(f"  Genes (raw)                    : {info['genes_raw']}")
    print(f"  Sparsity (raw)                 : {info['sparsity_raw']:.1f}%")
    print(f"  Genes after HVG selection      : {info['genes_hvg']}")
    print(f"  Sparsity (after preprocessing) : {info['sparsity_hvg']:.1f}%"
          "   (manuscript: 77.2%)")
    print(f"  Matrix-level mean, mu          : {info['mu']:.4f}"
          "   (manuscript: 0.4980)")

    os.makedirs(SAVE_DIR, exist_ok=True)
    exists = os.path.exists(X_PATH) and os.path.exists(MU_PATH)

    if exists and not overwrite:
        X_old = np.load(X_PATH)
        mu_old = float(np.load(MU_PATH)[0])
        print("\n  Existing files found; comparing instead of overwriting.")
        print(f"    stored shape {X_old.shape}, new shape {X.shape}")
        if X_old.shape != X.shape:
            print("    RESULT: DIFFERENT (shape mismatch)")
            return
        identical = np.array_equal(X_old, X)
        max_diff = float(np.max(np.abs(X_old.astype(np.float64)
                                       - X.astype(np.float64))))
        print(f"    identical (bit-for-bit) : {identical}")
        print(f"    max |difference|        : {max_diff:.3e}")
        print(f"    stored mu {mu_old:.6f}  vs  new mu {info['mu']:.6f}")
        print(f"    dtype stored/new        : {X_old.dtype} / {X.dtype}")
        if identical or max_diff < 1e-6:
            print("    RESULT: MATCH - this script reproduces the stored matrix.")
        else:
            print("    RESULT: DIFFERENT - do not overwrite; report to author.")
        return

    np.save(X_PATH, X)
    np.save(MU_PATH, np.array([mu]))       # same dtype as the original
    print(f"\n  Saved: {X_PATH}")
    print(f"  Saved: {MU_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build the Paul15 2,000-HVG matrix and global mean.")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing X_paul15.npy / global_mean_paul15.npy")
    main(ap.parse_args().overwrite)
