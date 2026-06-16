"""SQLite schema (DDL). Part of the FOUNDATION CONTRACT."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id            INTEGER PRIMARY KEY,
    path          TEXT NOT NULL,
    content_hash  TEXT NOT NULL UNIQUE,
    fmt           TEXT,
    duration      REAL,
    sample_rate   INTEGER,
    existing_tags TEXT,                     -- JSON
    status        TEXT DEFAULT 'new',
    added_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
    track_id INTEGER NOT NULL,
    model    TEXT NOT NULL,
    vector   BLOB NOT NULL,                 -- float32 .tobytes()
    dims     INTEGER NOT NULL,
    PRIMARY KEY (track_id, model),
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS genres (
    id              INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    parent_id       INTEGER,
    level           TEXT,                   -- subset | genre | subgenre
    source          TEXT DEFAULT 'seed',    -- seed | custom
    description     TEXT,
    threshold       REAL,
    centroid        BLOB,                   -- float32 .tobytes() (example- or text-derived)
    is_text_centroid INTEGER DEFAULT 0,
    UNIQUE (name, parent_id),
    FOREIGN KEY (parent_id) REFERENCES genres(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS exemplars (
    genre_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    PRIMARY KEY (genre_id, track_id),
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS assignments (
    track_id   INTEGER PRIMARY KEY,
    genre_id   INTEGER,
    confidence REAL,
    method     TEXT,                        -- zeroshot | centroid | manual
    status     TEXT,                        -- suggested | confirmed | rejected
    decided_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    id                 INTEGER PRIMARY KEY,
    run_id             TEXT,
    suggested_genre_id INTEGER
);

CREATE TABLE IF NOT EXISTS cluster_members (
    cluster_id INTEGER NOT NULL,
    track_id   INTEGER NOT NULL,
    PRIMARY KEY (cluster_id, track_id)
);

CREATE TABLE IF NOT EXISTS actions_log (
    id          INTEGER PRIMARY KEY,
    type        TEXT,                        -- tag_write | copy | move
    track_id    INTEGER,
    from_value  TEXT,
    to_value    TEXT,
    undo_token  TEXT,
    status      TEXT DEFAULT 'done',         -- done | undone
    ts          TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analysis (
    track_id     INTEGER PRIMARY KEY,
    bpm          REAL,
    music_key    TEXT,
    energy       REAL,          -- 0..1 perceived intensity
    danceability REAL,          -- 0..1
    extra        TEXT,          -- JSON for anything else
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_genres_parent ON genres(parent_id);
CREATE INDEX IF NOT EXISTS idx_exemplars_genre ON exemplars(genre_id);
CREATE INDEX IF NOT EXISTS idx_actions_status ON actions_log(status);
"""
