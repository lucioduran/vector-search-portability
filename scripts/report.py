"""Render median trial measurements from verified reference evidence."""

import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for engine in ("hnswlib", "sqlite-vec"):
    reports = [
        json.loads(path.read_text())
        for path in sorted((root / "results/reference").glob(f"{engine}-*.json"))
    ]

    def median(*keys):
        values = []
        for report in reports:
            value = report
            for key in keys:
                value = value[key]
            values.append(value)
        return statistics.median(values)

    print(
        f"| {engine} | {median('ingestion', 'wall_seconds'):.3f} | {median('update', 'wall_seconds'):.3f} | {median('memory', 'sampled_peak_rss_bytes') / 2**20:.1f} | {median('disk_after_bytes') / 2**20:.2f} |"
    )
    for index in (0, 1):
        print(
            f"| {engine} | {median('load', index, 'workers')} | {median('load', index, 'requests_per_second'):.1f} | {median('load', index, 'p50_ms'):.3f} | {median('load', index, 'p95_ms'):.3f} |"
        )
