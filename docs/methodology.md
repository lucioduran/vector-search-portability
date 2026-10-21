# Experiment design

The unit of comparison is an in-process Python adapter, not a network service.
Two distinct native engines run locally: hnswlib 0.8.0 (C++ approximate HNSW)
and sqlite-vec 0.1.6 (C vec0 exact KNN). NumPy float64 exhaustive search is the
independent reference. No engine is implemented by calling that reference.

## Frozen input and semantics

The original MIT-licensed synthetic fixture contains 5,000 float32 vectors of
64 dimensions, 40 independent query vectors and 500 vector plus metadata updates.
Twenty Gaussian centers plus Gaussian noise provide clustered geometry. A NumPy
PCG64 generator uses seed 1729. These are synthetic embeddings, not neural text
embeddings, and do not test semantic relevance. The committed NPZ bytes, not a
promise of random generator compatibility, are the portable input contract.

IDs are dense zero-based integers. Metadata consists of tenant (0..9), active
(0 or 1) and bucket (0..99). All clauses are ANDed. Filters are: unrestricted;
tenant = 0; active = 1; tenant = 0 AND active = 1 AND bucket >= 95; and tenant =
999 (no matches). Both native adapters apply the same predicates during search.
SQLite uses vec0 metadata constraints without partitioning. HNSW uses a Python
label predicate and counts eligible rows before requesting min(k, eligible).
This O(N) mask and Python callback cost is included in HNSW search timing.

Distances are squared Euclidean L2. sqlite-vec emits L2 distances, which the
adapter squares. HNSW emits squared L2 directly. The reference accumulates in
float64. Result order is distance then ID within the returned set. At a tied
cutoff native engines may select a different tied ID; strict ID recall can
penalize this. The benchmark does not redefine ties to inflate recall.

Recall@10 = intersection(returned IDs, exact top IDs) / exact top ID count.
Result loss = 1 - returned count / min(10, eligible count). Empty candidate sets
have null metrics and must return no IDs. Zero result loss is not perfect recall.
The diagnostic postfilter mode retrieves 40 unrestricted neighbors, then applies
the same predicate and keeps at most ten. Its loss quantifies candidate-budget
failure; it is explicitly separate from native filtered search.

## Trials and load

Three paired repetitions use fresh subprocesses, with engine order alternating.
HNSW settings are M=16, construction ef=100, search ef=64, random_seed=1729 and
one internal thread. SQLite uses its default rollback journal and synchronous
settings, with one transaction per build or update. No tuning follows results.

Build timing includes adapter validation, native insertion and SQLite commit.
It excludes engine object creation and Python import. Update timing includes
validation, replacement of the first 500 vectors, metadata replacement and
SQLite commit. HNSW updates have no crash durability guarantee. Disk sizes are
explicit snapshots: HNSW graph plus metadata NPY versus SQLite database. Snapshot
serialization is outside ingestion/update time, so timings are not comparisons
of equivalent durability. Index reload and crash recovery are outside scope.

Quality is measured before and after updates for each query, filter and mode.
Exact-reference computation is outside latency timing. Quality searches have no
explicit warmup and share progressively warmed indexes. Load follows updates and
uses native filters only. Each client warms one search before a barrier, then
issues one synchronous request at a time. One and four client threads run 400
requests each; native thread limits remain one. Query/filter schedule is saved.
This is a finite closed-loop saturation workload with no think time, arrival
rate, network or external queue. Per-request latency covers adapter calls;
throughput starts just before the main thread waits at the barrier and ends
after all completed clients, including any remaining warmup and Python bookkeeping. Thread-local SQLite connections avoid serializing all reads
through one connection. HNSW Python predicates contend on the GIL.

RSS is sampled every 10 ms from the isolated process, after dataset loading and
before index creation work. Baseline and peak include the dataset, interpreter,
reference calculation, Python adapter and saved trace objects. Sampling can miss
short peaks. CPU seconds use process user + system deltas; load CPU includes
client bookkeeping and sampling. Disk is file size, not physical allocated bytes.
No energy, cost, GPU, server, distributed or production workload is measured.

## Evidence and verification boundaries

Each report records versions, OS, CPU, logical cores, RAM, source hashes, source
commit, dirty flag, thread settings and UTC timestamp. Each manifest hashes the
NPZ input and raw JSON reports. Verification recomputes exact neighbors, validates
filter membership and distances, checks row coverage and load schedules, and
recomputes scores, percentiles and throughput. Hashes detect accidental changes;
they do not authenticate who measured a run. Timings and memory cannot be replayed
from rankings. Verification does not independently prove hardware measurements.

Sources: [hnswlib API](https://github.com/nmslib/hnswlib/tree/v0.8.0),
[sqlite-vec metadata](https://alexgarcia.xyz/sqlite-vec/features/vec0.html), and
[sqlite-vec v0.1.6](https://github.com/asg017/sqlite-vec/releases/tag/v0.1.6).
The adapters are tested against pinned packages, since current docs may describe
newer versions. Dependencies retain their upstream licenses.
