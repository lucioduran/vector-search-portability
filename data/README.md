# Fixed synthetic embeddings

`fixed.npz` is the original MIT-licensed generated dataset for this experiment.
It contains base, meta, queries, update_ids, update_vectors and update_meta arrays.
No private records, scraped text, external model or third-party corpus is included.
Generation uses the defaults in vector_portability/core.py, seed 1729, NumPy
2.2.6. The raw NPZ is committed so both engines consume identical float32 values.
Each report and manifest records its SHA-256. Regeneration can change ZIP bytes
without changing arrays, so compare arrays when studying generator reproducibility.
