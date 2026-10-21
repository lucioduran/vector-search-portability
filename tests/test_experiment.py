import json
import subprocess
import sys
from pathlib import Path

import pytest

from vector_portability.core import generate
from vector_portability.experiment import digest, run, write_json
from vector_portability.verify import verify


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiment")
    dataset = root / "fixed.npz"
    generate(20, 4, 2).save(dataset)
    output = root / "run"
    run(dataset, output, repeats=1, requests=8, k=3)
    return dataset, output


def test_real_isolated_trials_verify(evidence):
    dataset, output = evidence
    assert verify(dataset, output) == {"reports": 2, "verified_searches": 112}
    with pytest.raises(FileExistsError):
        run(dataset, output, 1, 8, 3)
    with pytest.raises(ValueError):
        run(dataset, output, 0, 8, 3)


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "truth",
        "score",
        "filter",
        "distance",
        "missing",
        "duplicate",
        "load",
        "summary",
        "manifest",
    ],
)
def test_tampering_rejected(evidence, tmp_path, mutation):
    dataset, original = evidence
    for path in original.glob("*.json"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["reports"][0]
    path = tmp_path / entry["file"]
    report = json.loads(path.read_text())
    row = report["rows"][0]
    if mutation == "truth":
        row["truth"] = []
    elif mutation == "score":
        row["recall"] = -1
    elif mutation == "filter":
        row["filter"] = "empty"
    elif mutation == "distance":
        row["squared_l2"][0] += 100
    elif mutation == "missing":
        report["rows"].pop()
    elif mutation == "duplicate":
        report["rows"].append(row)
    elif mutation == "load":
        report["load"][0]["rows"][0]["latency_ms"] = -1
    elif mutation == "summary":
        report["summary"] = {}
    elif mutation == "manifest":
        manifest["reports"].pop()
    else:
        report["k"] = 4
    write_json(path, report)
    if mutation != "hash":
        entry["sha256"] = digest(path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        verify(dataset, tmp_path)


def test_cli_errors(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vector_portability",
            "generate",
            "--output",
            str(tmp_path / "x.npz"),
            "--count",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "count" in result.stderr


def test_repository_rules():
    headings = [
        line[3:] for line in Path("README.md").read_text().splitlines() if line.startswith("## ")
    ]
    assert headings == [
        "Problem",
        "Scope",
        "Quick start",
        "Usage",
        "Architecture",
        "Reproduction",
        "Results",
        "Testing",
        "Limitations",
        "Contributing",
        "License",
    ]
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    for filename in tracked:
        if Path(filename).suffix in {".py", ".md", ".yml", ".toml", ".txt"}:
            assert chr(0x2014) not in Path(filename).read_text(), filename


def test_zero_cpu_delta_does_not_become_negative(monkeypatch):
    from types import SimpleNamespace

    from vector_portability.experiment import Resources

    resource = Resources()
    times = SimpleNamespace(user=0.41, system=0.04)
    monkeypatch.setattr(resource, "process", SimpleNamespace(cpu_times=lambda: times))
    assert resource.measure(lambda: None)["cpu_seconds"] == 0.0
