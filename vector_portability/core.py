"""Shared input, predicate and exact ranking contracts."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Filter:
    tenant: int | None = None
    active: int | None = None
    min_bucket: int | None = None

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            if value is not None and type(value) is not int:
                raise ValueError("filter values must be integers or None")
        if self.active not in (None, 0, 1):
            raise ValueError("active must be 0 or 1")

    def mask(self, metadata: IntArray) -> npt.NDArray[np.bool_]:
        selected = np.ones(len(metadata), dtype=bool)
        for column, value in enumerate((self.tenant, self.active, self.min_bucket)):
            if value is not None:
                selected &= (
                    metadata[:, column] >= value if column == 2 else metadata[:, column] == value
                )
        return selected

    def sql(self) -> tuple[str, list[int]]:
        clauses, values = [], []
        for column, op, value in (
            ("tenant", "=", self.tenant),
            ("active", "=", self.active),
            ("bucket", ">=", self.min_bucket),
        ):
            if value is not None:
                clauses.append(f"{column} {op} ?")
                values.append(value)
        return "".join(f" AND {clause}" for clause in clauses), values


FILTERS = {
    "all": Filter(),
    "tenant": Filter(tenant=0),
    "active": Filter(active=1),
    "selective": Filter(tenant=0, active=1, min_bucket=95),
    "empty": Filter(tenant=999),
}


def vectors(value: npt.ArrayLike, dimension: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError("vectors must be a finite two-dimensional matrix")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError("vector dimension mismatch")
    return np.ascontiguousarray(array)


def metadata(value: npt.ArrayLike, count: int) -> IntArray:
    array = np.asarray(value)
    if array.shape != (count, 3) or array.dtype.kind not in "iu":
        raise ValueError("metadata must be an integer N by 3 matrix")
    if not np.isin(array[:, 1], [0, 1]).all():
        raise ValueError("active metadata must be 0 or 1")
    return array.astype(np.int64)


def query_vector(query: npt.ArrayLike, dimension: int, k: int) -> FloatArray:
    if type(k) is not int or k < 1 or k > 4096:
        raise ValueError("k must be an integer between 1 and 4096")
    array = np.asarray(query, dtype=np.float32)
    if array.shape != (dimension,) or not np.isfinite(array).all():
        raise ValueError("query must be a finite vector of the index dimension")
    return array


def exact(x: FloatArray, meta: IntArray, query: FloatArray, k: int, where: Filter) -> list[int]:
    query_vector(query, x.shape[1], k)
    ids = np.flatnonzero(where.mask(meta))
    # Float64 reference avoids cancellation and engine-specific accumulation order.
    distance = np.sum((x[ids].astype(np.float64) - query.astype(np.float64)) ** 2, axis=1)
    return ids[np.lexsort((ids, distance))[:k]].tolist()


def metrics(ids: list[int], truth: list[int], eligible: int, k: int) -> dict[str, float | None]:
    if len(ids) != len(set(ids)) or len(ids) > k:
        raise ValueError("duplicate or excess result IDs")
    target = min(k, eligible)
    return {
        "recall": len(set(ids) & set(truth)) / len(truth) if truth else None,
        "result_loss": 1 - len(ids) / target if target else None,
    }


@dataclass
class Dataset:
    base: FloatArray
    meta: IntArray
    queries: FloatArray
    update_ids: IntArray
    update_vectors: FloatArray
    update_meta: IntArray

    def __post_init__(self) -> None:
        self.base = vectors(self.base)
        if not len(self.base):
            raise ValueError("corpus must not be empty")
        self.meta = metadata(self.meta, len(self.base))
        self.queries = vectors(self.queries, self.base.shape[1])
        if not len(self.queries):
            raise ValueError("queries must not be empty")
        ids = np.asarray(self.update_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(set(ids)) != len(ids):
            raise ValueError("update IDs must be unique integers")
        if np.any(ids < 0) or np.any(ids >= len(self.base)):
            raise ValueError("update ID outside corpus")
        self.update_ids = ids.astype(np.int64)
        self.update_vectors = vectors(self.update_vectors, self.base.shape[1])
        if len(self.update_vectors) != len(ids):
            raise ValueError("update length mismatch")
        self.update_meta = metadata(self.update_meta, len(ids))

    def state(self, updated: bool) -> tuple[FloatArray, IntArray]:
        x, meta = self.base.copy(), self.meta.copy()
        if updated:
            x[self.update_ids] = self.update_vectors
            meta[self.update_ids] = self.update_meta
        return x, meta

    def save(self, path: Path) -> None:
        if path.exists():
            raise ValueError("dataset already exists")
        np.savez_compressed(path, **asdict(self))

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        with np.load(path, allow_pickle=False) as source:
            return cls(**dict(source))


def generate(
    count: int = 5000, dimension: int = 64, queries: int = 40, seed: int = 1729
) -> Dataset:
    if count < 10 or dimension < 2 or queries < 1:
        raise ValueError("count >= 10, dimension >= 2 and queries >= 1 required")
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(20, dimension))

    def sample(n: int) -> FloatArray:
        return (centers[rng.integers(20, size=n)] + rng.normal(size=(n, dimension))).astype(
            np.float32
        )

    def tags(n: int) -> IntArray:
        return np.column_stack(
            (rng.integers(10, size=n), rng.integers(2, size=n), rng.integers(100, size=n))
        )

    return Dataset(
        sample(count),
        tags(count),
        sample(queries),
        np.arange(count // 10),
        sample(count // 10),
        tags(count // 10),
    )
