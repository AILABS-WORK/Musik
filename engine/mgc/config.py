"""Engine configuration (JSON-backed, stdlib only).

Part of the FOUNDATION CONTRACT. Modules read Config; they do not redefine it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SUPPORTED_EXTS = (".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".aiff", ".aif")


@dataclass
class Config:
    # Storage
    db_path: str = "mgc.sqlite"
    # Source library to scan
    library_root: Optional[str] = None
    # Active embedding model — exactly one at a time; never mix embedding spaces.
    active_model: str = "baseline"  # baseline | discogs | mert | clap
    # Classification
    confidence_threshold: float = 0.35  # cosine; below => "needs review"
    top_k: int = 3
    # Audio windowing (whole-track, non-overlapping by default)
    window_seconds: float = 5.0
    window_hop_seconds: float = 5.0
    max_windows: int = 24  # cap windows per track for speed
    # Output: folder organization
    organize_root: Optional[str] = None
    organize_mode: str = "copy"  # copy | move
    # Output: tags — write the specific subgenre into the single genre field
    write_parent_to_grouping: bool = False

    extensions: tuple = field(default_factory=lambda: SUPPORTED_EXTS)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        # tuples come back as lists from JSON
        if "extensions" in data and data["extensions"] is not None:
            data["extensions"] = tuple(data["extensions"])
        known = {f for f in cls().__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
