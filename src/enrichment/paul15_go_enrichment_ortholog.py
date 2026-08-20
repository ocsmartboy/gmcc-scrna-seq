import scanpy as sc
import numpy as np
import networkx as nx
import gseapy as gp
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("PAUL ET AL. 2015: GO ENRICHMENT VIA ORTHOLOG CONVERSION")
print("=" * 65)

save_dir = ("data/processed")

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("\n[1/5] Memuat data...")
X           = np.load(f"{save_dir}/X_paul15.npy")
global_mean = np.load(f"{save_dir}/global_mean_paul15.npy")[0]

adata = sc.datasets.paul15()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata      = adata[:, adata.var.highly_variable]
gene_names = list(adata.var_names)

print(f"      Gen tersedia (mouse symbol): {len(gene_names)}")

# ── 2. BANGUN JARINGAN & EKSTRAK MODUL (GMCC) ──────────────────
print("\n[2/5] Membangun jaringan GMCC dan ekstraksi modul...")

def gmcc_vectorized(X, mu):
    X_c   = X - mu
    norms = np.linalg.norm(X_c, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return np.clip((X_c/norms) @ (X_c/norms).T, -1.0, 1.0)

sim_gmcc = gmcc_vectorized(X, global_mean)

idx       = np.triu_indices(len(gene_names), k=1)
upper     = sim_gmcc[idx]
threshold = np.percentile(upper, 95)

G = nx.Graph()
G.add_nodes_from(gene_names)
rows, cols = idx
mask       = upper > threshold
for i, j, s in zip(rows[mask], cols[mask], upper[mask]):
    G.add_edge(gene_names[i], gene_names[j], weight=float(s))

components = sorted(nx.connected_components(G), key=len, reverse=True)
modules    = [list(c) for c in components if len(c) >= 10][:5]

print(f"      Threshold: {threshold:.4f}")
print(f"      Modul ditemukan (≥10 gen): {len(modules)}")
for i, mod in enumerate(modules):
    print(f"        Modul {i+1}: {len(mod)} gen — "
          f"{', '.join(mod[:4])}...")

# ── 3. KONVERSI ORTHOLOG MOUSE → HUMAN ─────────────────────────
print("\n[3/5] Konversi ortholog mouse → human via Biomart...")
print("      (memerlukan koneksi internet ke Ensembl Biomart)")

try:
    from gseapy import Biomart
    bm = Biomart()

    print("      Mengambil tabel ortholog mouse-human...",
          end="", flush=True)
    ortholog_table = bm.query(
        dataset="mmusculus_gene_ensembl",
        attributes=[
            "external_gene_name",
            "hsapiens_homolog_associated_gene_name",
        ],
    )
    print(" ✓")

    # Bangun dictionary mouse_symbol -> human_symbol
    ortholog_table = ortholog_table.dropna()
    mouse_to_human = dict(zip(
        ortholog_table["external_gene_name"],
        ortholog_table["hsapiens_homolog_associated_gene_name"],
    ))
    print(f"      Total pasangan ortholog tersedia: "
          f"{len(mouse_to_human):,}")

    ORTHOLOG_AVAILABLE = True

except Exception as e:
    print(f"\n      [GAGAL] {e}")
    print("      Biomart tidak dapat diakses. "
          "Coba install: pip install pybiomart")
    ORTHOLOG_AVAILABLE = False
    mouse_to_human = {}

# ── 4. KONVERSI SETIAP MODUL & JALANKAN GO ENRICHMENT ──────────
print("\n[4/5] Konversi modul dan GO Enrichment...")

def run_go_enrichment(gene_list):
    try:
        result = gp.enrichr(
            gene_list  = gene_list,
            gene_sets  = "GO_Biological_Process_2023",
            outdir     = None,
            verbose    = False,
        )
        df = result.results
        if df is not None and len(df) > 0:
            df_sig = df[df["Adjusted P-value"] < 0.05]
            df_sig = df_sig.sort_values("Adjusted P-value")
            return df_sig
    except Exception as e:
        print(f" [error: {e}]", end="")
    return None

module_results = []

for i, module in enumerate(modules):
    print(f"\n  ── Modul {i+1} ({len(module)} gen mouse) ──")

    if ORTHOLOG_AVAILABLE:
        # Konversi: uppercase mouse symbol untuk matching,
        # lalu translasi via tabel ortholog
        converted = [
            mouse_to_human[g] for g in module
            if g in mouse_to_human and mouse_to_human[g] != ""
        ]
        converted = list(set(converted))  # buang duplikat

        mapping_rate = len(converted) / len(module) * 100
        print(f"    Berhasil dikonversi: {len(converted)}/"
              f"{len(module)} gen ({mapping_rate:.1f}%)")

        if len(converted) >= 5:
            df_sig = run_go_enrichment(converted)
            n_sig  = len(df_sig) if df_sig is not None else 0
            print(f"    GO terms signifikan: {n_sig}")

            if df_sig is not None and len(df_sig) > 0:
                for _, row in df_sig.head(3).iterrows():
                    print(f"      • {row['Term'][:50]}")
                    print(f"        p-adj={row['Adjusted P-value']:.2e}")

            module_results.append({
                "module_id"    : i + 1,
                "n_genes_mouse": len(module),
                "n_converted"  : len(converted),
                "mapping_rate" : mapping_rate,
                "n_sig_go"     : n_sig,
            })
        else:
            print(f"    [Dilewati] Gen terkonversi terlalu sedikit "
                  f"(<5) untuk enrichment valid")
            module_results.append({
                "module_id"    : i + 1,
                "n_genes_mouse": len(module),
                "n_converted"  : len(converted),
                "mapping_rate" : mapping_rate,
                "n_sig_go"     : 0,
            })
    else:
        print("    [Dilewati] Ortholog conversion tidak tersedia")

# ── 5. RINGKASAN ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RINGKASAN GO ENRICHMENT PAUL ET AL. (via ortholog conversion)")
print("=" * 65)

if module_results:
    total_sig       = sum(m["n_sig_go"] for m in module_results)
    total_genes     = sum(m["n_genes_mouse"] for m in module_results)
    total_converted = sum(m["n_converted"] for m in module_results)
    overall_rate    = total_converted / total_genes * 100

    print(f"\n  Jumlah modul dianalisis   : {len(module_results)}")
    print(f"  Total gen mouse           : {total_genes}")
    print(f"  Total berhasil dikonversi : {total_converted} "
          f"({overall_rate:.1f}%)")
    print(f"  Total GO terms signifikan : {total_sig}")

    print(f"\n  {'Modul':<8}{'Gen(mouse)':>12}{'Konversi':>10}"
          f"{'Rate':>8}{'GO terms':>10}")
    print(f"  {'-'*48}")
    for m in module_results:
        print(f"  {m['module_id']:<8}{m['n_genes_mouse']:>12}"
              f"{m['n_converted']:>10}{m['mapping_rate']:>7.1f}%"
              f"{m['n_sig_go']:>10}")

    print(f"\n  Perbandingan dengan PBMC 3K:")
    print(f"    PBMC 3K   : 243 GO terms signifikan (3 modul)")
    print(f"    Paul et al: {total_sig} GO terms signifikan "
          f"({len(module_results)} modul)")
else:
    print("\n  Tidak ada hasil — ortholog conversion gagal total.")
    print("  Rekomendasi: gunakan Opsi A (native-mouse pathway)")
    print("  sebagai fallback, atau laporkan sebagai limitasi teknis.")

print("\n✓ Selesai.")
