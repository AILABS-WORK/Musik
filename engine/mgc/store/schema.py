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

CREATE TABLE IF NOT EXISTS understanding (
    track_id        INTEGER PRIMARY KEY,
    audioset        BLOB,          -- float32[527] AudioSet probability vector
    audioset_model  TEXT,          -- 'ast' | 'efficientat'
    instruments     TEXT,          -- JSON {name: prob}
    vocal           TEXT,          -- JSON {voice_instrumental, gender, language, ...}
    mood            TEXT,          -- JSON {arousal, valence, tags, scores}
    caption         TEXT,
    tags_canonical  TEXT,          -- JSON list
    deep_done       INTEGER DEFAULT 0,
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- AcoustID/MusicBrainz identity: the track recognised by its audio fingerprint,
-- giving authoritative artist/title/genre/region regardless of messy filenames.
CREATE TABLE IF NOT EXISTS identity (
    track_id        INTEGER PRIMARY KEY,
    recording_mbid  TEXT,
    artist          TEXT,
    title           TEXT,
    genres          TEXT,          -- JSON list of MusicBrainz genres (authoritative)
    area            TEXT,          -- region (release country)
    year            TEXT,
    score           REAL,          -- AcoustID match score 0..1
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- top-N genre suggestions per track (a track can be a blend): the primary is the
-- stored assignment; these are the alternatives with their scores, for relabeling.
CREATE TABLE IF NOT EXISTS suggestions (
    track_id   INTEGER NOT NULL,
    genre_id   INTEGER NOT NULL,
    confidence REAL NOT NULL,
    rank       INTEGER NOT NULL,
    method     TEXT,
    PRIMARY KEY (track_id, genre_id),
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_suggestions_track ON suggestions(track_id, rank);

-- per-window embeddings (the "segment index"): lets us find tracks that contain
-- a part that sounds like a selected region of another track.
CREATE TABLE IF NOT EXISTS segment_embeddings (
    id        INTEGER PRIMARY KEY,
    track_id  INTEGER NOT NULL,
    model     TEXT NOT NULL,
    start     REAL NOT NULL,
    end       REAL NOT NULL,
    vector    BLOB NOT NULL,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_segemb_model ON segment_embeddings(model);
CREATE INDEX IF NOT EXISTS idx_segemb_track ON segment_embeddings(track_id, model);

-- labeled segment exemplars: "this region of this track IS the electroclash
-- cowbell" -> a sound-level fingerprint that can define/seed a subgenre.
CREATE TABLE IF NOT EXISTS segments (
    id         INTEGER PRIMARY KEY,
    track_id   INTEGER NOT NULL,
    model      TEXT NOT NULL,
    start      REAL NOT NULL,
    end        REAL NOT NULL,
    label      TEXT,
    note       TEXT,
    genre_id   INTEGER,
    vector     BLOB NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_genres_parent ON genres(parent_id);
CREATE INDEX IF NOT EXISTS idx_exemplars_genre ON exemplars(genre_id);
CREATE INDEX IF NOT EXISTS idx_actions_status ON actions_log(status);
"""
