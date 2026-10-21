# Vector Search Portability

[![CI](https://github.com/lucioduran/vector-search-portability/actions/workflows/ci.yml/badge.svg)](https://github.com/lucioduran/vector-search-portability/actions/workflows/ci.yml)

Compare real hnswlib and sqlite-vec search with frozen vectors, shared filters,
an independent exact reference and inspectable raw results.

## Problem

Moving vector search between engines can change recall, filtering behavior,
update cost and latency. A count of ten results does not guarantee the right ten.
This project measures these differences with an explicit common contract.

## Scope

Two free local native engines: approximate HNSW and exact SQLite vec0 search.
The original synthetic fixture has 5,000 vectors, 64 dimensions, 40 queries and
500 vector plus metadata updates. It tests geometry and adapter behavior, not
language understanding or production database scalability.

## Quick start

Python 3.11 and a C++ compiler are required for hnswlib. On macOS install Xcode
Command Line Tools; on Linux install a C++ build toolchain.

```sh
git clone https://github.com/lucioduran/vector-search-portability.git
cd vector-search-portability
python3 scripts/setup.py
.venv/bin/python -m vector_portability verify --output results/reference
```

On Windows use `.venv\Scripts\python.exe` and an appropriate C++ compiler.
SQLite must support loadable extensions. No server, Docker, model download,
API key or paid service is required.

## Usage

```sh
.venv/bin/python -m vector_portability generate --output build/custom.npz --count 1000
.venv/bin/python -m vector_portability run --dataset data/fixed.npz --output build/run --repeats 3
.venv/bin/python -m vector_portability verify --dataset data/fixed.npz --output build/run
```

Create the parent output directory first. Existing datasets and run directories
are never overwritten. A failed run has no final manifest and cannot pass
verification. Commands exit 2 on invalid input. Inputs are trusted local files.
The internal `trial` command runs one engine for subprocess isolation.

## Architecture

```text
Saved float32 vectors + integer metadata + saved updates
             |                       |
      hnswlib HNSW              sqlite-vec vec0
             |                       |
             +--- common filters ----+
                         |
       raw IDs, distances, timings and resources
                         |
           float64 exact reference + verifier
```

See [methodology](docs/methodology.md) for filter semantics, metric formulas,
threading, timing boundaries, resource accounting and durability differences.

## Reproduction

The committed NPZ fixture is the authoritative fixed input. Generate a new
fixture only for a separately named experiment. Run the three paired trials:

```sh
.venv/bin/python -m vector_portability run --output build/reproduced
.venv/bin/python -m vector_portability verify --output build/reproduced
```

When participating in the shared project batch, every benchmark must instead run
under the coordinator's exclusive lock wrapper:

```sh
python3 /Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py .venv/bin/python -m vector_portability run --output build/reproduced
```

The runner uses fresh processes and one internal CPU thread per engine. Read
load uses one and four clients. Engine order alternates across repetitions.
`requirements.txt` pins runtime packages; `requirements-local.lock` records the
complete measured environment. Clean disposable work with
`python3 scripts/cleanup.py`; this only removes this repository's `build/` tree.
Recorded results are never removed by that script.

## Results

Measured on 2026-10-21: Intel Xeon W-3223, 16 logical CPUs, 48 GiB RAM,
Intel macOS 26.7.1, Python 3.11.13. Values below are medians of three trial
measurements, including the median of each trial's percentile, not pooled
percentiles. All 9,600 saved searches pass independent evidence verification.

| Engine | Ingest 5,000 (s) | Update 500 (s) | Sampled peak RSS (MiB) | Snapshot (MiB) |
| --- | ---: | ---: | ---: | ---: |
| hnswlib | 0.287 | 0.236 | 95.5 | 2.04 |
| sqlite-vec | 0.116 | 0.031 | 104.5 | 1.52 |

| Engine | Read clients | Requests/s | P50 (ms) | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| hnswlib | 1 | 1503.6 | 0.363 | 2.295 |
| hnswlib | 4 | 100.3 | 9.402 | 149.332 |
| sqlite-vec | 1 | 1412.7 | 0.666 | 0.801 |
| sqlite-vec | 4 | 2637.5 | 1.476 | 1.660 |

Each load trial has 400 requests: 120 unrestricted, 120 tenant-filtered,
80 active-filtered and 80 selectively filtered. Both engines achieve strict
recall@10 = 1.0 with native filtering before and after updates on all nonempty
query groups. Native result loss is zero. This small fixture did not expose
approximation error at the frozen HNSW settings.

After updates, both engines' diagnostic postfilter mode has recall 0.4725 and
result loss 0.5275 for tenant filtering. For the selective filter, recall is
0.0275 and result loss is 0.9725, despite 14 eligible vectors. Retrieving 40 global
neighbors is insufficient for these predicates.

Four clients sharply degrade this HNSW Python adapter on the mixed workload.
The adapter combines a Python predicate, GIL contention and O(N) mask creation;
this experiment does not isolate their individual contributions. These results
are not a general ranking of HNSW against SQLite, nor a production scaling claim.
Ingestion timing also compares different durability semantics.

- [Manifest and report hashes](results/reference/manifest.json)
- [hnswlib trial 1](results/reference/hnswlib-0.json), [trial 2](results/reference/hnswlib-1.json), [trial 3](results/reference/hnswlib-2.json)
- [sqlite-vec trial 1](results/reference/sqlite-vec-0.json), [trial 2](results/reference/sqlite-vec-1.json), [trial 3](results/reference/sqlite-vec-2.json)
- [Evidence notes](results/README.md) and [methodology](docs/methodology.md)

Run `.venv/bin/python scripts/report.py` to regenerate the table values.

## Testing

```sh
.venv/bin/python -m pytest --cov=vector_portability --cov-report=term-missing
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy vector_portability
.venv/bin/python -m vector_portability verify --output results/reference
```

Tests use both real engines, hand-calculated rankings, metadata updates, empty
and underfull filters, concurrent reads, malformed inputs, isolated tiny trials
and tampered reports. CI repeats native integration tests and evidence validation
on Linux and macOS. Tiny test trials are correctness checks, not benchmarks.

## Limitations

- Synthetic embeddings and one local machine cannot establish a production winner.
- The exact engine and approximate engine have different algorithms and durability.
- HNSW filtering includes a Python callback and a full metadata mask per request.
- Closed-loop thread load does not measure network, overload or queueing latency.
- Peak RSS includes Python, reference work and traces, and may miss short peaks.
- Strict ID recall may penalize ties. Empty filters have null scores.
- Three repetitions describe this machine; no general confidence claim is made.
- Deletes, crash recovery, persisted-index reload, service isolation, access control,
  hosted engines and million-vector scale are not evaluated.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Preserve inconvenient outcomes and add
contract tests when introducing an engine, filter or metric. Report security
issues as described in [SECURITY.md](SECURITY.md).

## License

Original code, synthetic data, documentation and recorded results use the
[MIT license](LICENSE). Dependencies retain their own licenses.
