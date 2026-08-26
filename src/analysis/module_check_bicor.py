import numpy as np

print("=" * 60)
print("BICOR FALLBACK PROPORTION CHECK")
print("=" * 60)

save_dir = "d:/S3/Eksperimen/Bioinformatika/gmcc_biomedical/data/processed"

def check_fallback(X, name):
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True)
    n_fallback = int(np.sum(mad.flatten() < 1e-8))
    n_total    = X.shape[0]
    pct        = n_fallback / n_total * 100
    print(f"{name}: {n_fallback}/{n_total} genes ({pct:.1f}%) required MAD fallback")
    return n_fallback, n_total

X_pbmc = np.load(f"{save_dir}/X_2000hvg.npy")
X_paul = np.load(f"{save_dir}/X_paul15.npy")

check_fallback(X_pbmc, "PBMC3K")
check_fallback(X_paul, "Paul15")

print("\nSELESAI")