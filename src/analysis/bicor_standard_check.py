import numpy as np

print("=" * 60)
print("STANDARD BICOR (NO FALLBACK) — DEGENERACY CHECK")
print("=" * 60)

save_dir = "d:/S3/Eksperimen/Bioinformatika/gmcc_biomedical/data/processed"

def bicor_standard_vectorized(X):
    """Bicor MURNI — tanpa fallback ke std saat MAD=0."""
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True)
    # TIDAK ada fallback — biarkan MAD=0 apa adanya
    mad_safe = np.where(mad == 0, np.nan, mad)  # tandai eksplisit sebagai undefined
    u = (X - med) / (9.0 * mad_safe)
    w = np.where(np.abs(u) < 1, (1 - u**2)**2, 0.0)
    X_t = w * (X - med)
    norms = np.linalg.norm(X_t, axis=1, keepdims=True)
    sim = (X_t / norms) @ (X_t / norms).T
    return sim

def check_degeneracy(X, name):
    sim = bicor_standard_vectorized(X)
    n_total    = sim.size
    n_nan      = int(np.isnan(sim).sum())
    n_valid    = n_total - n_nan
    print(f"\n{name}:")
    print(f"  Total entri similarity matrix : {n_total:,}")
    print(f"  NaN (undefined)               : {n_nan:,} ({n_nan/n_total*100:.2f}%)")
    print(f"  Valid (defined)               : {n_valid:,} ({n_valid/n_total*100:.2f}%)")

X_pbmc = np.load(f"{save_dir}/X_2000hvg.npy")
X_paul = np.load(f"{save_dir}/X_paul15.npy")

check_degeneracy(X_pbmc, "PBMC3K")
check_degeneracy(X_paul, "Paul15")

print("\nSELESAI")