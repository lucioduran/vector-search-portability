"""Real local engines implementing one filter and squared L2 contract."""

import sqlite3
import threading
from pathlib import Path
from typing import Protocol

import hnswlib
import numpy as np
import sqlite_vec

from .core import Filter, FloatArray, IntArray, metadata, query_vector, vectors


class Engine(Protocol):
    def build(self, x: FloatArray, meta: IntArray) -> None: ...
    def update(self, ids: IntArray, x: FloatArray, meta: IntArray) -> None: ...
    def search(self, q: FloatArray, k: int, where: Filter) -> tuple[list[int], list[float]]: ...
    def snapshot_bytes(self) -> int: ...
    def close(self) -> None: ...


class Base:
    def __init__(self, path: Path, dimension: int) -> None:
        self.path, self.dimension = path, dimension
        self.meta: IntArray = np.empty((0, 3), dtype=np.int64)
        self.ready = False

    def validate_build(self, x: FloatArray, meta: IntArray) -> tuple[FloatArray, IntArray]:
        if self.ready:
            raise ValueError("index already built")
        x = vectors(x, self.dimension)
        if not len(x):
            raise ValueError("cannot build empty index")
        return x, metadata(meta, len(x))

    def validate_update(
        self, ids: IntArray, x: FloatArray, meta: IntArray
    ) -> tuple[FloatArray, IntArray]:
        if not self.ready:
            raise ValueError("index is not ready")
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) != len(set(ids)):
            raise ValueError("update IDs must be unique integers")
        if np.any(ids < 0) or np.any(ids >= len(self.meta)):
            raise ValueError("unknown update ID")
        x = vectors(x, self.dimension)
        if len(x) != len(ids):
            raise ValueError("update length mismatch")
        return x, metadata(meta, len(ids))

    def validate_query(self, q: FloatArray, k: int) -> FloatArray:
        if not self.ready:
            raise ValueError("index is not ready")
        return query_vector(q, self.dimension, k)


class Hnsw(Base):
    def __init__(self, path: Path, dimension: int, ef: int = 64) -> None:
        super().__init__(path, dimension)
        if type(ef) is not int or ef < 1:
            raise ValueError("ef must be positive")
        self.ef = ef
        self.index = hnswlib.Index(space="l2", dim=dimension)

    def build(self, x: FloatArray, meta: IntArray) -> None:
        x, meta = self.validate_build(x, meta)
        self.index.init_index(max_elements=len(x), M=16, ef_construction=100, random_seed=1729)
        self.index.set_num_threads(1)
        self.index.set_ef(self.ef)
        self.index.add_items(x, np.arange(len(x)), num_threads=1)
        self.meta, self.ready = meta.copy(), True

    def update(self, ids: IntArray, x: FloatArray, meta: IntArray) -> None:
        x, meta = self.validate_update(ids, x, meta)
        self.index.add_items(x, ids, num_threads=1)
        self.meta[ids] = meta

    def search(self, q: FloatArray, k: int, where: Filter) -> tuple[list[int], list[float]]:
        q = self.validate_query(q, k)
        allowed = where.mask(self.meta)
        count = min(k, int(allowed.sum()))
        if not count:
            return [], []
        predicate = None if where == Filter() else lambda label: bool(allowed[label])
        ids, distances = self.index.knn_query(q, k=count, num_threads=1, filter=predicate)
        pairs = sorted(
            zip(ids[0].tolist(), distances[0].tolist(), strict=True), key=lambda p: (p[1], p[0])
        )
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def snapshot_bytes(self) -> int:
        self.index.save_index(str(self.path / "hnsw.bin"))
        np.save(self.path / "metadata.npy", self.meta)
        return sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())

    def close(self) -> None:
        self.ready = False
        self.index = None


class SqliteVec(Base):
    def __init__(self, path: Path, dimension: int) -> None:
        super().__init__(path, dimension)
        self.local = threading.local()
        self.connections: list[sqlite3.Connection] = []
        self.lock = threading.Lock()

    def connection(self) -> sqlite3.Connection:
        if not hasattr(self.local, "connection"):
            connection = sqlite3.connect(self.path / "vectors.db", check_same_thread=False)
            connection.enable_load_extension(True)
            sqlite_vec.load(connection)
            connection.enable_load_extension(False)
            self.local.connection = connection
            with self.lock:
                self.connections.append(connection)
        return self.local.connection

    def build(self, x: FloatArray, meta: IntArray) -> None:
        x, meta = self.validate_build(x, meta)
        connection = self.connection()
        connection.execute(
            f"CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{self.dimension}], "
            "tenant integer, active integer, bucket integer)"
        )
        with connection:
            connection.executemany(
                "INSERT INTO vectors(rowid, embedding, tenant, active, bucket) VALUES (?, ?, ?, ?, ?)",
                [(i, row.astype("<f4").tobytes(), *map(int, meta[i])) for i, row in enumerate(x)],
            )
        self.meta, self.ready = meta.copy(), True

    def update(self, ids: IntArray, x: FloatArray, meta: IntArray) -> None:
        x, meta = self.validate_update(ids, x, meta)
        with self.connection() as connection:
            connection.executemany(
                "UPDATE vectors SET embedding=?, tenant=?, active=?, bucket=? WHERE rowid=?",
                [
                    (row.astype("<f4").tobytes(), *map(int, meta[j].tolist()), int(i))
                    for j, (i, row) in enumerate(zip(ids, x, strict=True))
                ],
            )
        self.meta[ids] = meta

    def search(self, q: FloatArray, k: int, where: Filter) -> tuple[list[int], list[float]]:
        q = self.validate_query(q, k)
        clause, parameters = where.sql()
        rows = (
            self.connection()
            .execute(
                "SELECT rowid, distance FROM vectors WHERE embedding MATCH ? AND k = ?"
                + clause
                + " ORDER BY distance",
                [q.astype("<f4").tobytes(), k, *parameters],
            )
            .fetchall()
        )
        rows.sort(key=lambda row: (row[1], row[0]))
        return [int(row[0]) for row in rows], [float(row[1]) ** 2 for row in rows]

    def snapshot_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())

    def close(self) -> None:
        for connection in self.connections:
            connection.close()
        self.connections.clear()
        self.ready = False


def create(name: str, path: Path, dimension: int) -> Engine:
    path.mkdir(parents=True, exist_ok=False)
    if name == "hnswlib":
        return Hnsw(path, dimension)
    if name == "sqlite-vec":
        return SqliteVec(path, dimension)
    raise ValueError(f"unknown engine: {name}")
