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

## Confirmation after CPU accounting fix

Linux CI exposed floating-point cancellation when unchanged process CPU samples
were subtracted one component at a time. Commit
`a416742cb7df20476b67c75dc4c13fe67c2d7ead` groups each sample's user + system sum
before subtraction. A regression test covers unchanged samples. The reference
reports are preserved verbatim; none had an invalid negative phase timing.

A separate three-repeat run in `results/confirmation` used that clean commit,
the same input bytes and the same exclusive benchmark wrapper. All six source
manifests identify the corrected implementation, and all 9,600 searches verify.
CI verifies both evidence sets. The original tables remain the original run's
measurements, so no favorable replacement of timing observations is hidden.

| Engine | Ingestion median (s) | Update median (s) | 1-client requests/s | 4-client requests/s | 4-client P95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| hnswlib | 0.282 | 0.236 | 1494.4 | 100.0 | 150.289 |
| sqlite-vec | 0.116 | 0.021 | 1528.3 | 2695.9 | 1.626 |

Native recall remains 1.0 and native result loss remains zero. Selective
postfilter recall remains 0.0275 with 0.9725 result loss after updates. Variability
in timing, including SQLite update latency, is retained in the raw reports.
