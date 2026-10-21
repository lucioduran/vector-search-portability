"""Create the project-local environment and install pinned dependencies."""

import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
environment = root / ".venv"
if not environment.exists():
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
subprocess.run(
    [str(python), "-m", "pip", "install", "-r", str(root / "requirements-dev.txt")], check=True
)
(root / "build").mkdir(exist_ok=True)
