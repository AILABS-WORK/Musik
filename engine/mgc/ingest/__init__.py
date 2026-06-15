"""Ingest module — filesystem scanning, content hashing and tag reading.

Walks a library root, fingerprints each audio file by content hash (so re-scans
are idempotent), reads embedded tags and basic audio info, and upserts Tracks
into the Store.
"""

from __future__ import annotations

from mgc.ingest.scanner import content_hash, read_tags, scan

__all__ = ["content_hash", "read_tags", "scan"]
