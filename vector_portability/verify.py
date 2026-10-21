"""Recompute rankings, scores and summaries without invoking either engine."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .core import FILTERS, Dataset, FloatArray, IntArray, exact, metrics
from .experiment import digest, percentiles, summarize


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_results(
    row: dict[str, Any], x: FloatArray, meta: IntArray, data: Dataset, k: int
) -> None:
    qi = row["query"]
    require(type(qi) is int and 0 <= qi < len(data.queries), "invalid query ID")
    ids, distances = row["ids"], row["squared_l2"]
    require(all(type(i) is int and 0 <= i < len(x) for i in ids), "invalid result ID")
    require(len(ids) == len(set(ids)) and len(ids) <= k, "duplicate or excess result IDs")
    require(len(ids) == len(distances), "distance count mismatch")
    require(all(math.isfinite(d) and d >= 0 for d in distances), "invalid distance")
    require(math.isfinite(row["latency_ms"]) and row["latency_ms"] >= 0, "invalid latency")
    mask = FILTERS[row["filter"]].mask(meta)
    require(bool(np.all(mask[ids])), "filter violation")
    expected = np.sum(
        (x[ids].astype(np.float64) - data.queries[qi].astype(np.float64)) ** 2, axis=1
    )
    require(bool(np.allclose(distances, expected, rtol=2e-5, atol=1e-6)), "distance mismatch")
    require(
        all(a <= b for a, b in zip(distances, distances[1:], strict=False)),
        "distance order mismatch",
    )


def verify(dataset: Path, output: Path) -> dict[str, int]:
    data = Dataset.load(dataset)
    manifest = json.loads((output / "manifest.json").read_text())
    require(manifest["schema"] == 1, "unsupported manifest")
    require(manifest["dataset_sha256"] == digest(dataset), "dataset hash mismatch")
    expected_files = {
        f"{engine}-{r}.json"
        for engine in ("hnswlib", "sqlite-vec")
        for r in range(manifest["repeats"])
    }
    files = [r["file"] for r in manifest["reports"]]
    require(
        set(files) == expected_files and len(files) == len(expected_files), "report set mismatch"
    )
    states = {phase: data.state(phase == "after") for phase in ("before", "after")}
    truths = {
        (phase, name, qi): exact(x, meta, data.queries[qi], manifest["k"], where)
        for phase, (x, meta) in states.items()
        for name, where in FILTERS.items()
        for qi in range(len(data.queries))
    }
    total = 0
    for entry in manifest["reports"]:
        path = output / entry["file"]
        require(digest(path) == entry["sha256"], "report hash mismatch")
        report = json.loads(path.read_text())
        require(
            f"{report['engine']}-{report['repeat']}.json" == entry["file"],
            "report identity mismatch",
        )
        require(
            report["schema"] == 1 and report["dataset_sha256"] == digest(dataset),
            "report dataset mismatch",
        )
        require(report["k"] == manifest["k"], "k mismatch")
        expected_rows = {
            (phase, mode, name, qi)
            for phase in states
            for mode in ("native", "postfilter")
            for name in FILTERS
            for qi in range(len(data.queries))
        }
        seen = set()
        for row in report["rows"]:
            key = (row["phase"], row["mode"], row["filter"], row["query"])
            require(key in expected_rows and key not in seen, "duplicate or unexpected quality row")
            seen.add(key)
            x, meta = states[row["phase"]]
            check_results(row, x, meta, data, report["k"])
            truth = truths[(row["phase"], row["filter"], row["query"])]
            eligible = int(FILTERS[row["filter"]].mask(meta).sum())
            require(
                row["truth"] == truth and row["eligible"] == eligible, "exact reference mismatch"
            )
            score = metrics(row["ids"], truth, eligible, report["k"])
            require(all(row[key] == value for key, value in score.items()), "metric mismatch")
            total += 1
        require(seen == expected_rows, "missing quality rows")
        require(report["summary"] == summarize(report["rows"]), "summary mismatch")
        require([item["workers"] for item in report["load"]] == [1, 4], "load worker set mismatch")
        for item in report["load"]:
            require(item["requests"] == manifest["requests"], "load request count mismatch")
            require(
                [r["request"] for r in item["rows"]] == list(range(item["requests"])),
                "missing load rows",
            )
            require(
                item["wall_seconds"] > 0 and math.isfinite(item["wall_seconds"]),
                "invalid load wall time",
            )
            require(
                item["requests_per_second"] == item["requests"] / item["wall_seconds"],
                "throughput mismatch",
            )
            for row in item["rows"]:
                request = row["request"]
                require(row["query"] == request % len(data.queries), "load query schedule mismatch")
                names = ("all", "tenant", "active", "selective")
                require(
                    row["filter"] == names[(request // len(data.queries)) % 4],
                    "load filter schedule mismatch",
                )
                check_results(row, *states["after"], data, report["k"])
                total += 1
            stats = percentiles([r["latency_ms"] for r in item["rows"]])
            require(
                all(item[key] == value for key, value in stats.items()), "load percentile mismatch"
            )
        for phase, count in (("ingestion", len(data.base)), ("update", len(data.update_ids))):
            require(report[phase]["records"] == count, "record count mismatch")
            for timing_key in ("cpu_seconds", "wall_seconds"):
                require(
                    math.isfinite(report[phase][timing_key]) and report[phase][timing_key] >= 0,
                    "invalid phase timing",
                )
        require(
            report["memory"]["sampled_peak_rss_bytes"]
            >= report["memory"]["baseline_rss_bytes"]
            > 0,
            "invalid RSS",
        )
        require(
            report["disk_before_bytes"] > 0 and report["disk_after_bytes"] > 0, "invalid disk size"
        )
    return {"reports": len(files), "verified_searches": total}
