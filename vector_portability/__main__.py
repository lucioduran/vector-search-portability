"""Command line entry point."""

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

from .core import generate
from .experiment import run, trial
from .verify import verify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("generate")
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--count", type=int, default=5000)
    make.add_argument("--dimension", type=int, default=64)
    make.add_argument("--queries", type=int, default=40)
    for command in ("run", "trial", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", type=Path, default=Path("data/fixed.npz"))
        sub.add_argument("--output", type=Path, required=True)
        if command != "verify":
            sub.add_argument("--requests", type=int, default=400)
            sub.add_argument("--k", type=int, default=10)
        if command == "run":
            sub.add_argument("--repeats", type=int, default=3)
        if command == "trial":
            sub.add_argument("--engine", choices=["hnswlib", "sqlite-vec"], required=True)
            sub.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            generate(args.count, args.dimension, args.queries).save(args.output)
        elif args.command == "run":
            run(args.dataset, args.output, args.repeats, args.requests, args.k)
        elif args.command == "trial":
            trial(args.dataset, args.output, args.engine, args.repeat, args.k, args.requests)
        else:
            print(json.dumps(verify(args.dataset, args.output)))
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        sqlite3.Error,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
