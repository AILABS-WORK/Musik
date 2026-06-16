"""SQLite store — the single source of truth.

Part of the FOUNDATION CONTRACT. Modules use this API (and may run bespoke SQL
via ``store.conn`` for narrow queries) but MUST NOT edit this file.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from mgc.store.schema import SCHEMA
from mgc.types import ActionRecord, ClusterResult, GenreNode, Track


def vec_to_blob(v) -> bytes:
    return np.asarray(v, dtype=np.float32).ravel().tobytes()


def blob_to_vec(b) -> Optional[np.ndarray]:
    if b is None:
        return None
    return np.frombuffer(b, dtype=np.float32).copy()


class Store:
    """Thin typed wrapper over a sqlite3 connection."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---- lifecycle ----------------------------------------------------------
    @classmethod
    def open(cls, path: str | Path, check_same_thread: bool = True) -> "Store":
        # check_same_thread=False lets the API server share one connection across
        # request threads (access is serialized by a lock in the server).
        conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        store = cls(conn)
        store.migrate()
        return store

    def migrate(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.conn.close()

    # ---- row mappers --------------------------------------------------------
    @staticmethod
    def _track(row: sqlite3.Row) -> Track:
        return Track(
            id=row["id"],
            path=row["path"],
            content_hash=row["content_hash"],
            fmt=row["fmt"],
            duration=row["duration"],
            sample_rate=row["sample_rate"],
            existing_tags=json.loads(row["existing_tags"]) if row["existing_tags"] else {},
            status=row["status"],
        )

    @staticmethod
    def _genre(row: sqlite3.Row) -> GenreNode:
        return GenreNode(
            id=row["id"],
            name=row["name"],
            parent_id=row["parent_id"],
            level=row["level"],
            source=row["source"],
            description=row["description"],
            threshold=row["threshold"],
        )

    # ---- tracks -------------------------------------------------------------
    def upsert_track(self, t: Track) -> int:
        cur = self.conn.execute(
            """INSERT INTO tracks(path, content_hash, fmt, duration, sample_rate, existing_tags, status)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(content_hash) DO UPDATE SET
                 path=excluded.path, fmt=excluded.fmt, duration=excluded.duration,
                 sample_rate=excluded.sample_rate, existing_tags=excluded.existing_tags
               RETURNING id""",
            (t.path, t.content_hash, t.fmt, t.duration, t.sample_rate,
             json.dumps(t.existing_tags or {}), t.status or "new"),
        )
        tid = cur.fetchone()["id"]
        self.conn.commit()
        return tid

    def get_track(self, track_id: int) -> Optional[Track]:
        row = self.conn.execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone()
        return self._track(row) if row else None

    def get_track_by_hash(self, content_hash: str) -> Optional[Track]:
        row = self.conn.execute("SELECT * FROM tracks WHERE content_hash=?", (content_hash,)).fetchone()
        return self._track(row) if row else None

    def iter_tracks(self, status: Optional[str] = None) -> list[Track]:
        if status:
            rows = self.conn.execute("SELECT * FROM tracks WHERE status=? ORDER BY id", (status,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM tracks ORDER BY id").fetchall()
        return [self._track(r) for r in rows]

    def count_tracks(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS c FROM tracks").fetchone()["c"]

    def set_track_status(self, track_id: int, status: str) -> None:
        self.conn.execute("UPDATE tracks SET status=? WHERE id=?", (status, track_id))
        self.conn.commit()

    # ---- embeddings ---------------------------------------------------------
    def save_embedding(self, track_id: int, model: str, vector) -> None:
        v = np.asarray(vector, dtype=np.float32).ravel()
        self.conn.execute(
            """INSERT INTO embeddings(track_id, model, vector, dims) VALUES(?,?,?,?)
               ON CONFLICT(track_id, model) DO UPDATE SET vector=excluded.vector, dims=excluded.dims""",
            (track_id, model, v.tobytes(), int(v.shape[0])),
        )
        self.conn.commit()

    def get_embedding(self, track_id: int, model: str) -> Optional[np.ndarray]:
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE track_id=? AND model=?", (track_id, model)
        ).fetchone()
        return blob_to_vec(row["vector"]) if row else None

    def has_embedding(self, track_id: int, model: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM embeddings WHERE track_id=? AND model=?", (track_id, model)
        ).fetchone() is not None

    def load_matrix(self, model: str) -> tuple[list[int], np.ndarray]:
        """Return (track_ids, matrix[n, dims]) for all embeddings of ``model``."""
        rows = self.conn.execute(
            "SELECT track_id, vector FROM embeddings WHERE model=? ORDER BY track_id", (model,)
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [r["track_id"] for r in rows]
        mat = np.stack([blob_to_vec(r["vector"]) for r in rows]).astype(np.float32)
        return ids, mat

    # ---- genres -------------------------------------------------------------
    def upsert_genre(self, g: GenreNode) -> int:
        cur = self.conn.execute(
            """INSERT INTO genres(name, parent_id, level, source, description, threshold)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(name, parent_id) DO UPDATE SET
                 level=excluded.level, source=excluded.source,
                 description=COALESCE(excluded.description, genres.description),
                 threshold=COALESCE(excluded.threshold, genres.threshold)
               RETURNING id""",
            (g.name, g.parent_id, g.level, g.source, g.description, g.threshold),
        )
        gid = cur.fetchone()["id"]
        self.conn.commit()
        return gid

    def get_genre(self, genre_id: int) -> Optional[GenreNode]:
        row = self.conn.execute("SELECT * FROM genres WHERE id=?", (genre_id,)).fetchone()
        return self._genre(row) if row else None

    def get_genre_by_name(self, name: str, parent_id: Optional[int] = None) -> Optional[GenreNode]:
        if parent_id is None:
            row = self.conn.execute(
                "SELECT * FROM genres WHERE name=? AND parent_id IS NULL", (name,)
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM genres WHERE name=? AND parent_id=?", (name, parent_id)
            ).fetchone()
        return self._genre(row) if row else None

    def iter_genres(self, level: Optional[str] = None) -> list[GenreNode]:
        if level:
            rows = self.conn.execute("SELECT * FROM genres WHERE level=? ORDER BY id", (level,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM genres ORDER BY id").fetchall()
        return [self._genre(r) for r in rows]

    def children(self, parent_id: Optional[int]) -> list[GenreNode]:
        if parent_id is None:
            rows = self.conn.execute("SELECT * FROM genres WHERE parent_id IS NULL ORDER BY id").fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM genres WHERE parent_id=? ORDER BY id", (parent_id,)).fetchall()
        return [self._genre(r) for r in rows]

    def set_centroid(self, genre_id: int, vector, is_text: bool = False) -> None:
        self.conn.execute(
            "UPDATE genres SET centroid=?, is_text_centroid=? WHERE id=?",
            (vec_to_blob(vector), 1 if is_text else 0, genre_id),
        )
        self.conn.commit()

    def get_centroid(self, genre_id: int) -> Optional[np.ndarray]:
        row = self.conn.execute("SELECT centroid FROM genres WHERE id=?", (genre_id,)).fetchone()
        return blob_to_vec(row["centroid"]) if row and row["centroid"] is not None else None

    def iter_centroids(self) -> list[tuple[int, str, np.ndarray, bool]]:
        """All genres that have a centroid: (genre_id, name, vector, is_text)."""
        rows = self.conn.execute(
            "SELECT id, name, centroid, is_text_centroid FROM genres WHERE centroid IS NOT NULL ORDER BY id"
        ).fetchall()
        return [(r["id"], r["name"], blob_to_vec(r["centroid"]), bool(r["is_text_centroid"])) for r in rows]

    # ---- exemplars ----------------------------------------------------------
    def add_exemplar(self, genre_id: int, track_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO exemplars(genre_id, track_id) VALUES(?,?)", (genre_id, track_id)
        )
        self.conn.commit()

    def get_exemplars(self, genre_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT track_id FROM exemplars WHERE genre_id=? ORDER BY track_id", (genre_id,)
        ).fetchall()
        return [r["track_id"] for r in rows]

    # ---- assignments --------------------------------------------------------
    def set_assignment(self, track_id: int, genre_id: Optional[int], confidence: float,
                       method: str, status: str = "suggested") -> None:
        self.conn.execute(
            """INSERT INTO assignments(track_id, genre_id, confidence, method, status)
               VALUES(?,?,?,?,?)
               ON CONFLICT(track_id) DO UPDATE SET
                 genre_id=excluded.genre_id, confidence=excluded.confidence,
                 method=excluded.method, status=excluded.status,
                 decided_at=CURRENT_TIMESTAMP""",
            (track_id, genre_id, confidence, method, status),
        )
        self.conn.commit()

    def get_assignment(self, track_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM assignments WHERE track_id=?", (track_id,)).fetchone()

    def iter_assignments(self, status: Optional[str] = None) -> list[sqlite3.Row]:
        if status:
            return self.conn.execute("SELECT * FROM assignments WHERE status=?", (status,)).fetchall()
        return self.conn.execute("SELECT * FROM assignments").fetchall()

    # ---- clusters -----------------------------------------------------------
    def clear_clusters(self) -> None:
        self.conn.execute("DELETE FROM cluster_members")
        self.conn.execute("DELETE FROM clusters")
        self.conn.commit()

    def add_cluster(self, run_id: str, suggested_genre_id: Optional[int] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO clusters(run_id, suggested_genre_id) VALUES(?,?) RETURNING id",
            (run_id, suggested_genre_id),
        )
        cid = cur.fetchone()["id"]
        self.conn.commit()
        return cid

    def add_cluster_member(self, cluster_id: int, track_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO cluster_members(cluster_id, track_id) VALUES(?,?)",
            (cluster_id, track_id),
        )
        self.conn.commit()

    def get_clusters(self, run_id: Optional[str] = None) -> list[ClusterResult]:
        if run_id:
            crows = self.conn.execute("SELECT * FROM clusters WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        else:
            crows = self.conn.execute("SELECT * FROM clusters ORDER BY id").fetchall()
        out = []
        for c in crows:
            members = [r["track_id"] for r in self.conn.execute(
                "SELECT track_id FROM cluster_members WHERE cluster_id=? ORDER BY track_id", (c["id"],)
            ).fetchall()]
            out.append(ClusterResult(cluster_id=c["id"], member_track_ids=members,
                                     suggested_genre_id=c["suggested_genre_id"]))
        return out

    # ---- actions log --------------------------------------------------------
    def log_action(self, type: str, track_id: Optional[int], from_value: Optional[str],
                   to_value: Optional[str], undo_token: Optional[str] = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO actions_log(type, track_id, from_value, to_value, undo_token, status)
               VALUES(?,?,?,?,?, 'done') RETURNING id""",
            (type, track_id, from_value, to_value, undo_token),
        )
        aid = cur.fetchone()["id"]
        self.conn.commit()
        return aid

    @staticmethod
    def _action(row: sqlite3.Row) -> ActionRecord:
        return ActionRecord(
            id=row["id"], type=row["type"], track_id=row["track_id"],
            from_value=row["from_value"], to_value=row["to_value"],
            undo_token=row["undo_token"], status=row["status"], ts=row["ts"],
        )

    def iter_actions(self, status: Optional[str] = None, type: Optional[str] = None) -> list[ActionRecord]:
        q = "SELECT * FROM actions_log"
        clauses, params = [], []
        if status:
            clauses.append("status=?"); params.append(status)
        if type:
            clauses.append("type=?"); params.append(type)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY id"
        return [self._action(r) for r in self.conn.execute(q, params).fetchall()]

    def set_action_status(self, action_id: int, status: str) -> None:
        self.conn.execute("UPDATE actions_log SET status=? WHERE id=?", (status, action_id))
        self.conn.commit()

    # ---- analysis (bpm / key / energy) -------------------------------------
    def save_analysis(self, track_id: int, bpm: Optional[float] = None,
                      music_key: Optional[str] = None, energy: Optional[float] = None,
                      danceability: Optional[float] = None, extra: Optional[dict] = None) -> None:
        self.conn.execute(
            """INSERT INTO analysis(track_id, bpm, music_key, energy, danceability, extra)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(track_id) DO UPDATE SET
                 bpm=excluded.bpm, music_key=excluded.music_key, energy=excluded.energy,
                 danceability=excluded.danceability, extra=excluded.extra""",
            (track_id, bpm, music_key, energy, danceability,
             json.dumps(extra) if extra else None),
        )
        self.conn.commit()

    def get_analysis(self, track_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM analysis WHERE track_id=?", (track_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["extra"] = json.loads(d["extra"]) if d["extra"] else {}
        return d

    def has_analysis(self, track_id: int) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM analysis WHERE track_id=?", (track_id,)
        ).fetchone() is not None

    def load_analysis(self) -> dict[int, dict]:
        """All analysis rows as {track_id: {bpm, music_key, energy, danceability, extra}}."""
        out: dict[int, dict] = {}
        for row in self.conn.execute("SELECT * FROM analysis").fetchall():
            d = dict(row)
            d["extra"] = json.loads(d["extra"]) if d["extra"] else {}
            out[d["track_id"]] = d
        return out
