"""mgc — Music Genre Classifier engine (Phase 1).

Embedding-based music genre/subgenre classification with few-shot custom
genres (no retraining), bulk tagging (Rekordbox-readable) and folder
organization. SQLite is the source of truth; embeddings cache by content hash.
"""

__version__ = "0.1.0"
