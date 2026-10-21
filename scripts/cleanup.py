"""Remove only this checkout's disposable build directory."""

import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / "build"
if target.is_symlink():
    raise SystemExit("Refusing to follow a build directory symlink")
if target.exists():
    shutil.rmtree(target)
print("Disposable build directory cleaned; data and results preserved.")
