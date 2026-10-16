import scanpy as sc
import numpy as np
import pandas as pd

print("=" * 50)
print("STAGE 1: DATA LOADING & PREPROCESSING")
print("=" * 50)

# ── 1. LOAD DATA ──────────────────────────────────────
# sc.datasets.pbmc3k() akan otomatis download dataset
# PBMC 3K dari server 10x Genomics (~21MB)
print("\n[1/5] Loading PBMC 3K dataset...")
adata = sc.datasets.pbmc3k()

print(f"      Ukuran data awal: {adata.shape[0]} sel × {adata.shape[1]} gen")

# ── 2. CEK SPARSITY AWAL ──────────────────────────────
# Ini penting: kita dokumentasikan seberapa sparse data aslinya
total_values   = adata.X.shape[0] * adata.X.shape[1]
nonzero_values = adata.X.nnz  # nnz = number of non-zeros
zero_values    = total_values - nonzero_values
sparsity       = zero_values / total_values * 100

print(f"\n[2/5] Cek sparsity data awal:")
print(f"      Total nilai        : {total_values:,}")
print(f"      Nilai non-zero     : {nonzero_values:,}")
print(f"      Nilai zero (0)     : {zero_values:,}")
print(f"      Tingkat sparsity   : {sparsity:.1f}%")

# ── 3. FILTER GEN ─────────────────────────────────────
# Buang gen yang hampir tidak pernah terdeteksi
# min_cells=3 artinya: gen harus muncul di minimal 3 sel
print(f"\n[3/5] Filter gen (min_cells=3)...")
sc.pp.filter_genes(adata, min_cells=3)
print(f"      Gen tersisa: {adata.shape[1]}")

# ── 4. NORMALISASI & LOG TRANSFORM ───────────────────
# Setiap sel punya jumlah total RNA berbeda-beda
# (bukan karena biologi, tapi karena teknis sequencing)
# Normalisasi menyamakan "total RNA" tiap sel ke 10.000
print(f"\n[4/5] Normalisasi dan log-transform...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(f"      Selesai.")

# ── 5. PILIH HIGHLY VARIABLE GENES (HVG) ─────────────
# Dari ribuan gen, pilih 2000 yang paling "informatif"
# (variasinya tinggi antar sel → lebih bermakna biologis)
print(f"\n[5/5] Memilih 2000 Highly Variable Genes (HVG)...")
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]
print(f"      Ukuran data final: {adata.shape[0]} sel × {adata.shape[1]} gen")

# ── RINGKASAN AKHIR ───────────────────────────────────
print("\n" + "=" * 50)
print("RINGKASAN PREPROCESSING")
print("=" * 50)
print(f"  Sel   : {adata.shape[0]}")
print(f"  Gen   : {adata.shape[1]}")

X = adata.X.toarray()  # konversi ke format array biasa
print(f"  Sparsity setelah preprocessing: "
      f"{(X == 0).sum() / X.size * 100:.1f}%")
print(f"  Min ekspresi : {X.min():.4f}")
print(f"  Max ekspresi : {X.max():.4f}")
print(f"  Mean ekspresi: {X.mean():.4f}")
print("\n✓ Stage 1 selesai. Data siap untuk Stage 2.")