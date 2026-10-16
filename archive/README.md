# Archived scripts (not used for the manuscript)

These scripts are kept only for transparency about earlier development stages.
They do **not** reproduce any value reported in the manuscript.

| Script | Why archived |
|---|---|
| `stage1_preprocessing.py` | Exploratory check of PBMC 3K preprocessing; saves nothing. The same preprocessing is performed by `src/similarity/stage2_similarity_2000hvg.py`. |
| `stage3_4_multiseed_with_topology.py` | Earlier masking protocol: the GMCC global mean was **retained** from the unperturbed matrix and masks were drawn with a different random generator (`numpy.random.default_rng`). Superseded by `src/stability/paired_difference_ci.py` (global mean recomputed from each masked matrix; `numpy.random.seed`, seeds 1–20) and `src/stability/network_topology_masking.py`. |
| `paul15_pipeline.py` | Original script that created `data/processed/X_paul15.npy` and `global_mean_paul15.npy` (its Stage 1). Kept as the historical source of these files. Its preprocessing is reproduced line for line by `src/similarity/stage2_similarity_paul15.py`, which also verifies the stored matrix. Its Stages 2–4 use an earlier protocol (a single masking seed, 42; global mean retained from the unperturbed matrix; mean absolute change divided by the standard deviation instead of NMAC; PBMC 3K values hard-coded) and do **not** reproduce manuscript values. The save path points to the author's earlier local folder. |
