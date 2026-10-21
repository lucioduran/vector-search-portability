"""Isolated engine trials and closed-loop read load measurements."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .core import FILTERS, Dataset, Filter, FloatArray, IntArray, exact, metrics
from .engines import Engine, create


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance() -> dict[str, Any]:
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    elif sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "sqlite": sqlite3.sqlite_version,
        "packages": {
            p: importlib.metadata.version(p) for p in ("numpy", "hnswlib", "sqlite-vec", "psutil")
        },
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        ),
        "source_sha256": {
            str(p): digest(p) for p in sorted(Path("vector_portability").glob("*.py"))
        },
    }


class Resources:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "Resources":
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()

    def measure(self, operation: Callable[[], Any]) -> dict[str, float]:
        before = self.process.cpu_times()
        start = time.perf_counter()
        operation()
        seconds = time.perf_counter() - start
        after = self.process.cpu_times()
        return {
            "wall_seconds": seconds,
            "cpu_seconds": after.user + after.system - before.user - before.system,
        }


def percentiles(values: list[float]) -> dict[str, float]:
    return {f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 95, 99)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for phase in ("before", "after"):
        for mode in ("native", "postfilter"):
            for name in FILTERS:
                group = [
                    r for r in rows if (r["phase"], r["mode"], r["filter"]) == (phase, mode, name)
                ]
                if not group:
                    continue
                scores = [r["recall"] for r in group if r["recall"] is not None]
                losses = [r["result_loss"] for r in group if r["result_loss"] is not None]
                result[f"{phase}/{mode}/{name}"] = {
                    "queries": len(group),
                    "recall": float(np.mean(scores)) if scores else None,
                    "result_loss": float(np.mean(losses)) if losses else None,
                    **percentiles([r["latency_ms"] for r in group]),
                }
    return result


def retrieve(
    engine: Engine, q: FloatArray, meta: IntArray, name: str, mode: str, k: int
) -> tuple[list[int], list[float]]:
    if mode == "native":
        return engine.search(q, k, FILTERS[name])
    ids, distances = engine.search(q, min(4 * k, 4096), Filter())
    mask = FILTERS[name].mask(meta)
    pairs = [(i, d) for i, d in zip(ids, distances, strict=True) if mask[i]][:k]
    return [i for i, _ in pairs], [d for _, d in pairs]


def quality(engine: Engine, data: Dataset, updated: bool, k: int) -> list[dict[str, Any]]:
    x, meta = data.state(updated)
    rows = []
    for name, where in FILTERS.items():
        eligible = int(where.mask(meta).sum())
        for qi in range(len(data.queries)):
            q = data.queries[qi]
            truth = exact(x, meta, q, k, where)
            for mode in ("native", "postfilter"):
                start = time.perf_counter()
                ids, distances = retrieve(engine, q, meta, name, mode, k)
                latency = (time.perf_counter() - start) * 1000
                rows.append(
                    {
                        "phase": "after" if updated else "before",
                        "mode": mode,
                        "filter": name,
                        "query": qi,
                        "eligible": eligible,
                        "truth": truth,
                        "ids": ids,
                        "squared_l2": distances,
                        "latency_ms": latency,
                        **metrics(ids, truth, eligible, k),
                    }
                )
    return rows


def load(engine: Engine, data: Dataset, workers: int, requests: int, k: int) -> dict[str, Any]:
    barrier = threading.Barrier(workers + 1)
    names = ("all", "tenant", "active", "selective")

    def client(worker: int) -> list[dict[str, Any]]:
        engine.search(data.queries[0], k, Filter())
        barrier.wait()
        rows = []
        for request in range(worker, requests, workers):
            qi = request % len(data.queries)
            name = names[(request // len(data.queries)) % len(names)]
            start = time.perf_counter()
            ids, distances = engine.search(data.queries[qi], k, FILTERS[name])
            rows.append(
                {
                    "request": request,
                    "query": qi,
                    "filter": name,
                    "ids": ids,
                    "squared_l2": distances,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
            )
        return rows

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(client, worker) for worker in range(workers)]
        start = time.perf_counter()
        barrier.wait(timeout=60)
        rows = [row for future in futures for row in future.result()]
        elapsed = time.perf_counter() - start
    rows.sort(key=lambda row: row["request"])
    return {
        "workers": workers,
        "requests": requests,
        "wall_seconds": elapsed,
        "requests_per_second": requests / elapsed,
        "rows": rows,
        **percentiles([r["latency_ms"] for r in rows]),
    }


def trial(
    dataset: Path, output: Path, engine_name: str, repeat: int, k: int, requests: int
) -> None:
    data = Dataset.load(dataset)
    report: dict[str, Any] = {
        "schema": 1,
        "engine": engine_name,
        "repeat": repeat,
        "dataset_sha256": digest(dataset),
        "k": k,
        "provenance": provenance(),
    }
    engine = create(engine_name, output / "indexes" / f"{engine_name}-{repeat}", data.base.shape[1])
    try:
        with Resources() as resource:
            report["ingestion"] = resource.measure(lambda: engine.build(data.base, data.meta))
            report["ingestion"]["records"] = len(data.base)
            report["disk_before_bytes"] = engine.snapshot_bytes()
            rows = quality(engine, data, False, k)
            report["update"] = resource.measure(
                lambda: engine.update(data.update_ids, data.update_vectors, data.update_meta)
            )
            report["update"]["records"] = len(data.update_ids)
            report["disk_after_bytes"] = engine.snapshot_bytes()
            rows += quality(engine, data, True, k)
            report["load"] = []
            for workers in (1, 4):
                item: dict[str, Any] = {}

                def run_load(item: dict[str, Any] = item, workers: int = workers) -> None:
                    item.update(load(engine, data, workers, requests, k))

                timing = resource.measure(run_load)
                item["cpu_seconds"] = timing["cpu_seconds"]
                report["load"].append(item)
        report["memory"] = {
            "baseline_rss_bytes": resource.baseline,
            "sampled_peak_rss_bytes": resource.peak,
            "sample_interval_seconds": 0.01,
        }
        report["rows"], report["summary"] = rows, summarize(rows)
        write_json(output / f"{engine_name}-{repeat}.json", report)
    finally:
        engine.close()


def run(dataset: Path, output: Path, repeats: int = 3, requests: int = 400, k: int = 10) -> None:
    if repeats < 1 or requests < 4 or not 1 <= k <= 1024:
        raise ValueError("repeats >= 1, requests >= 4 and 1 <= k <= 1024 required")
    Dataset.load(dataset)
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    reports = []
    for repeat in range(repeats):
        order = ("hnswlib", "sqlite-vec") if repeat % 2 == 0 else ("sqlite-vec", "hnswlib")
        for engine in order:
            print(f"Trial {repeat + 1}/{repeats}: {engine}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vector_portability",
                    "trial",
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--engine",
                    engine,
                    "--repeat",
                    str(repeat),
                    "--requests",
                    str(requests),
                    "--k",
                    str(k),
                ],
                check=True,
                env=environment,
            )
            name = f"{engine}-{repeat}.json"
            reports.append({"file": name, "sha256": digest(output / name)})
    write_json(
        output / "manifest.json",
        {
            "schema": 1,
            "dataset_sha256": digest(dataset),
            "repeats": repeats,
            "requests": requests,
            "k": k,
            "reports": reports,
        },
    )
