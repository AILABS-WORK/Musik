"""Build the standalone `mgc-sidecar` binary with PyInstaller.

Run from anywhere with the engine venv's python:
    engine/.venv/Scripts/python engine/packaging/build_sidecar.py
Output: dist/mgc-sidecar(.exe)

Copy the result into app/src-tauri/binaries/mgc-sidecar-<target-triple>(.exe)
and add it to tauri.conf.json `bundle.externalBin` to ship it with the app.
This bundles the BASELINE engine only (numpy/soundfile/sklearn/mutagen/fastapi).
The heavy model backends (torch/essentia/...) are intentionally NOT bundled.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "engine" / "packaging" / "sidecar.py"


def main() -> int:
    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onefile",
        "--name", "mgc-sidecar",
        "--paths", str(ROOT / "engine"),
        "--collect-submodules", "mgc",
        "--collect-submodules", "uvicorn",
        "--collect-submodules", "fastapi",
        "--collect-submodules", "starlette",
        "--collect-all", "soundfile",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"),
        str(ENTRY),
    ]
    print("running:", " ".join(args))
    return subprocess.call(args)


if __name__ == "__main__":
    raise SystemExit(main())
