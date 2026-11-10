"""Extract local references and maintain their embeddings incrementally."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from pypdf import PdfReader

from .ollama import embed_many
from .text import clean_text, cosine, split_chunks

# Prevent a single document from dominating indexing cost.
MAX_REFERENCE_CHARS_PER_FILE = 120_000
ALLOWED_DOC_EXT = {".pdf", ".md", ".txt"}


def iter_reference_docs(folder: Path) -> list[Path]:
    """Recursively list PDF, Markdown and text documents in a stable order."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_DOC_EXT)


def read_doc_chunks(path: Path, relative_name: str) -> list[tuple[str, str]]:
    """Extract (source, chunk) pairs, preserving page numbers for PDF documents.

    Text is limited per file and OCR is not performed; scanned PDFs need
    a text layer before indexing.
    """
    pieces: list[tuple[str, str]] = []
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        used = 0
        for page_num, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if not text:
                continue
            remaining = MAX_REFERENCE_CHARS_PER_FILE - used
            if remaining <= 0:
                break
            text = text[:remaining]
            used += len(text)
            for chunk in split_chunks(text):
                pieces.append((f"{relative_name} p.{page_num}", chunk))
    else:
        text = path.read_text(encoding="utf-8", errors="replace")[:MAX_REFERENCE_CHARS_PER_FILE]
        for chunk in split_chunks(text):
            pieces.append((relative_name, chunk))
    return pieces


def build_or_reuse_index(conn: sqlite3.Connection, folder: Path, model: str) -> list[tuple[str, str, list[float]]]:
    """Reuse embeddings by model and document hash, updating SQLite as needed.

    Changed files are reindexed; deleted sources are removed from the current
    model index. Return (source, chunk, vector) tuples.
    """
    docs = iter_reference_docs(folder)
    keep = set()
    for path in docs:
        name = path.relative_to(folder).as_posix()
        keep.add(name)
        # Content, rather than modification time, determines cache validity.
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        old = conn.execute("SELECT source_hash FROM chunks WHERE model=? AND source=? LIMIT 1", (model, name)).fetchone()
        if old and old[0] == fingerprint:
            continue
        content = read_doc_chunks(path, name)
        if not content:
            print(f"WARNING: no readable text in {name}; scanned PDFs need OCR beforehand.", file=sys.stderr)
            with conn:
                conn.execute("DELETE FROM chunks WHERE model=? AND source=?", (model, name))
            continue
        all_vectors: list[list[float]] = []
        # Small batches limit memory use and the size of each Ollama request.
        for i in range(0, len(content), 12):
            all_vectors.extend(embed_many([body for _, body in content[i:i+12]], model))
        # Replace all chunks in one transaction after obtaining the new vectors.
        with conn:
            conn.execute("DELETE FROM chunks WHERE model=? AND source=?", (model, name))
            conn.executemany(
                "INSERT INTO chunks(model,source,source_hash,chunk_no,body,vector_json) "
                "VALUES(?,?,?,?,?,?)",
                [(model, name, fingerprint, idx, f"[{source}] {body}", json.dumps(vec))
                 for idx, ((source, body), vec) in enumerate(zip(content, all_vectors))],
            )
        print(f"Indexed {name}: {len(content)} reference chunks", flush=True)
    # Remove deleted sources only for the current model, preserving other indexes.
    for (name,) in conn.execute("SELECT DISTINCT source FROM chunks WHERE model=?", (model,)).fetchall():
        if name not in keep:
            with conn:
                conn.execute("DELETE FROM chunks WHERE model=? AND source=?", (model, name))
    result = []
    for name, body, vec_json in conn.execute(
        "SELECT source, body, vector_json FROM chunks WHERE model=? ORDER BY source, chunk_no", (model,)
    ):
        result.append((name, body, json.loads(vec_json)))
    return result


def choose_references(embedding: list[float], chunks: list[tuple[str, str, list[float]]],
                      limit: int = 3) -> list[str]:
    """Choose the closest excerpts and limit the text sent to the model."""
    ranked = sorted(((cosine(embedding, vec), src, body) for src, body, vec in chunks), reverse=True)
    return [body[:850] for _, _, body in ranked[:limit]]
