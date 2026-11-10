"""SQLite persistence for reviewed-paper history and the reference cache."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from .models import Paper


def conn_open(path: Path) -> sqlite3.Connection:
    """Open SQLite and create the history and embedding tables if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    # Wait up to ten seconds for temporary locks held by another SQLite connection.
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("""CREATE TABLE IF NOT EXISTS reviewed (
       paper_id TEXT PRIMARY KEY, reviewed_at TEXT NOT NULL, model_score INTEGER)""")
    # Include the model in the key to avoid mixing vectors from different models.
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
       model TEXT NOT NULL, source TEXT NOT NULL, source_hash TEXT NOT NULL,
       chunk_no INTEGER NOT NULL, body TEXT NOT NULL, vector_json TEXT NOT NULL,
       PRIMARY KEY(model, source, chunk_no))""")
    conn.commit()
    return conn


def already_reviewed(conn: sqlite3.Connection) -> set[str]:
    """Retrieve previously reviewed identifiers to avoid repeated processing."""
    return {r[0] for r in conn.execute("SELECT paper_id FROM reviewed")}


def mark_reviewed(conn: sqlite3.Connection, papers: list[Paper], scores: dict[str, int]) -> None:
    """Record assessments in a transaction; repeated calls do not duplicate papers."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO reviewed(paper_id, reviewed_at, model_score) VALUES(?, ?, ?)",
            [(p.paper_id, timestamp, scores.get(p.paper_id)) for p in papers],
        )
