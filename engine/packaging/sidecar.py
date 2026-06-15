"""PyInstaller entry point for the standalone `mgc-sidecar` binary.

Bundled into the Tauri app so end users don't need a Python install. Runs the
FastAPI engine sidecar (uvicorn) on 127.0.0.1:8000 by default.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(prog="mgc-sidecar")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("MGC_PORT", "8000")))
    ap.add_argument("--config", default=None, help="path to mgc.config.json")
    args = ap.parse_args()

    if args.config:
        os.environ["MGC_CONFIG"] = args.config

    import uvicorn

    from mgc.server import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
