# Recorded evidence

The six reference reports were produced on 2026-10-21 from clean source commit
`7ee57a5af5e0f5bf4b46caf7eb55f0e1abd9fcf1`. Each report records source file hashes,
versions, hardware, UTC time and thread settings. Later commits add results and
documentation; the measured implementation remains identifiable independently.

Command executed from the repository root:

```sh
python3 /Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py .venv/bin/python -m vector_portability run --output results/reference
```

The shared wrapper acquired its exclusive lock before the six trials. Each engine
ran in a fresh subprocess in alternating order. All reports have dirty=false.
The manifest is written only after all trials succeed. Binary index snapshots are
disposable and are not published; their byte counts are recorded in the reports.
Input vectors, updates, query vectors and metadata are in data/fixed.npz.

Verification recomputed 4,800 quality search records against exact references
and validated 4,800 load search records for distances, filters and scheduling.
Quality records include both native filtering and the postfilter diagnostic.
Read load is native filtering only and follows updates. Load records preserve
rankings but do not add a separate aggregate load recall claim.

The same inputs are reused across trials. Three repetitions are not three
independent datasets. No human review, external audit, semantic retrieval result,
network load, paid compute or hosted inference result is claimed.
