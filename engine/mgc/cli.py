"""Typer CLI over the Engine facade.

Thin wrapper: every command maps to one Engine method. Phase 2's Tauri sidecar
calls the same Engine, so keep logic out of here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mgc.config import Config

app = typer.Typer(add_completion=False, help="mgc — music genre classifier engine (Phase 1)")

DEFAULT_CONFIG = "mgc.config.json"


def _load_config(config_path: str, library: Optional[str], db: Optional[str],
                 model: Optional[str]) -> Config:
    cfg = Config.load(config_path)
    if library:
        cfg.library_root = library
    if db:
        cfg.db_path = db
    if model:
        cfg.active_model = model
    return cfg


def _engine(config_path: str, library=None, db=None, model=None):
    from mgc.api.service import Engine
    return Engine(_load_config(config_path, library, db, model))


# Shared options
ConfigOpt = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to config JSON")
LibraryOpt = typer.Option(None, "--library", "-l", help="Library root to scan")
DbOpt = typer.Option(None, "--db", help="SQLite path override")
ModelOpt = typer.Option(None, "--model", "-m", help="Active embedder: baseline|discogs|mert|clap")


@app.command()
def init(config: str = ConfigOpt, library: str = LibraryOpt, db: str = DbOpt, model: str = ModelOpt):
    """Write a config file with sensible defaults."""
    cfg = _load_config(config, library, db, model)
    cfg.save(config)
    typer.echo(f"Wrote {config} (model={cfg.active_model}, db={cfg.db_path}, library={cfg.library_root})")


@app.command()
def scan(config: str = ConfigOpt, library: str = LibraryOpt, db: str = DbOpt):
    """Scan the library folder and register tracks."""
    with _engine(config, library, db) as e:
        ids = e.scan()
        typer.echo(f"Scanned {len(ids)} tracks (total in db: {e.store.count_tracks()}).")


@app.command()
def embed(config: str = ConfigOpt, db: str = DbOpt, model: str = ModelOpt, force: bool = False):
    """Embed all tracks (cached by content hash)."""
    with _engine(config, db=db, model=model) as e:
        n = e.embed_all(force=force, progress=lambda m: None)
        typer.echo(f"Embedded {n} tracks with model '{e.model}'.")


@app.command()
def suggest(config: str = ConfigOpt, db: str = DbOpt, model: str = ModelOpt):
    """Auto-suggest a genre for every embedded track (persists suggestions)."""
    with _engine(config, db=db, model=model) as e:
        out = e.suggest_all(persist=True)
        known = sum(1 for v in out.values() if v and v[0].genre_id is not None)
        typer.echo(f"Suggested genres for {len(out)} tracks ({known} above threshold).")


@app.command()
def cluster(config: str = ConfigOpt, db: str = DbOpt, model: str = ModelOpt, min_size: int = 2):
    """Cluster tracks into sound-alike groups."""
    with _engine(config, db=db, model=model) as e:
        clusters = e.cluster(min_cluster_size=min_size)
        typer.echo(f"Found {len(clusters)} clusters.")
        for c in clusters:
            typer.echo(f"  cluster {c.cluster_id}: {len(c.member_track_ids)} tracks")


@app.command()
def similar(track_id: int, config: str = ConfigOpt, db: str = DbOpt, model: str = ModelOpt, n: int = 10):
    """Show tracks most similar (sound-alike) to a track."""
    with _engine(config, db=db, model=model) as e:
        for tid, score in e.similar(track_id, n=n):
            t = e.store.get_track(tid)
            typer.echo(f"  {score:.3f}  {t.path if t else tid}")


@app.command()
def review(config: str = ConfigOpt, db: str = DbOpt, limit: int = 20):
    """List the lowest-confidence suggestions to edit."""
    with _engine(config, db=db) as e:
        for r in e.review(limit=limit):
            g = e.store.get_genre(r["genre_id"]) if r["genre_id"] else None
            typer.echo(f"  [{r['confidence']:.3f}] {Path(r['path']).name} -> {g.name if g else '??'}")


@app.command(name="write-tags")
def write_tags(config: str = ConfigOpt, db: str = DbOpt, apply: bool = typer.Option(False, help="Actually write (default dry-run)")):
    """Write accepted subgenres into file metadata (dry-run unless --apply)."""
    with _engine(config, db=db) as e:
        plans = e.write_tags(dry_run=not apply)
        typer.echo(f"{'WROTE' if apply else 'DRY-RUN'} {len(plans)} tag writes.")
        for p in plans[:20]:
            typer.echo(f"  {Path(p['path']).name}: '{p['from']}' -> '{p['to']}'")


@app.command()
def organize(config: str = ConfigOpt, db: str = DbOpt, apply: bool = typer.Option(False, help="Actually copy/move (default dry-run)")):
    """Build the Genre/Subgenre folder tree (dry-run unless --apply)."""
    with _engine(config, db=db) as e:
        plan = e.organize(dry_run=not apply)
        typer.echo(f"{'EXECUTED' if apply else 'DRY-RUN'} {len(plan)} file operations.")
        for p in plan[:20]:
            typer.echo(f"  {Path(p['src']).name} -> {p['dest']}")


@app.command()
def undo(config: str = ConfigOpt, db: str = DbOpt):
    """Undo the last tag writes and folder operations."""
    with _engine(config, db=db) as e:
        res = e.undo()
        typer.echo(f"Undid {res['tags']} tag writes and {res['organize']} file operations.")


@app.command(name="seed-taxonomy")
def seed_taxonomy(refs_dir: str, config: str = ConfigOpt, db: str = DbOpt, limit: int = typer.Option(None)):
    """Seed the genre taxonomy from a RateYourMusic JSON references dir."""
    with _engine(config, db=db) as e:
        n = e.seed_taxonomy(refs_dir, limit=limit)
        typer.echo(f"Seeded {n} genres.")


@app.command()
def tracks(config: str = ConfigOpt, db: str = DbOpt, limit: int = 50):
    """List tracks and their ids (use the ids as examples)."""
    with _engine(config, db=db) as e:
        for t in e.store.iter_tracks()[:limit]:
            typer.echo(f"  {t.id}\t{Path(t.path).name}")


@app.command()
def genres(config: str = ConfigOpt, db: str = DbOpt,
           level: str = typer.Option(None, help="subset|genre|subgenre"),
           like: str = typer.Option(None, help="substring filter")):
    """List genres in the taxonomy (find ids for --parent / confirm)."""
    with _engine(config, db=db) as e:
        for g in e.store.iter_genres(level=level):
            if like and like.lower() not in g.name.lower():
                continue
            parent = e.store.get_genre(g.parent_id).name if g.parent_id else "-"
            typer.echo(f"  {g.id}\t[{g.level}] {g.name}  (parent: {parent})")


@app.command(name="add-genre")
def add_genre(name: str,
              examples: str = typer.Option(..., "--examples", "-e", help="comma-separated track ids"),
              parent: int = typer.Option(None, "--parent", "-p", help="parent genre id"),
              level: str = typer.Option("subgenre"),
              config: str = ConfigOpt, db: str = DbOpt, model: str = ModelOpt):
    """Define a custom genre BY EXAMPLE from track ids (builds its centroid)."""
    ids = [int(x) for x in examples.split(",") if x.strip()]
    with _engine(config, db=db, model=model) as e:
        gid = e.add_genre_by_example(name, ids, parent_id=parent, level=level)
        typer.echo(f"Created '{name}' (id {gid}) from {len(ids)} examples; centroid built.")


@app.command()
def confirm(track_id: int, genre_id: int, config: str = ConfigOpt, db: str = DbOpt, model: str = ModelOpt):
    """Confirm a track's genre (adds it as an exemplar — the active-learning loop)."""
    with _engine(config, db=db, model=model) as e:
        e.confirm(track_id, genre_id)
        typer.echo(f"Confirmed track {track_id} -> genre {genre_id} (exemplar added, centroid updated).")


if __name__ == "__main__":
    app()
