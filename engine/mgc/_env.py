"""Minimal .env loader (stdlib only).

Reads ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ`` (without
overwriting anything already set), so secrets like ``ACOUSTID_API_KEY`` live in a
gitignored file instead of the code or the shell. Searches the obvious spots:
the cwd, the app/ folder, and the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = [False]


def _candidates() -> list[Path]:
    here = Path(__file__).resolve()
    repo = here.parents[2]  # engine/mgc/_env.py -> engine -> <repo>
    return [
        Path.cwd() / ".env",
        Path.cwd() / "app" / ".env",
        repo / "app" / ".env",
        repo / ".env",
    ]


def load_env(force: bool = False) -> None:
    """Populate os.environ from the first .env files found (idempotent)."""
    if _LOADED[0] and not force:
        return
    _LOADED[0] = True
    seen: set[str] = set()
    for path in _candidates():
        try:
            if not path.is_file() or str(path) in seen:
                continue
            seen.add(str(path))
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.lower().startswith("export "):
                    line = line[7:]
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
        except Exception:
            pass
