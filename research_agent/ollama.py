"""HTTP client for inference and embeddings on the configured Ollama server."""
from __future__ import annotations

import os
import sys
from typing import Any
from urllib.parse import urlparse

import requests


def ollama_post(path: str, payload: dict[str, Any], timeout: int = 240) -> dict[str, Any]:
    """Send JSON to Ollama and propagate HTTP errors or invalid JSON responses."""
    base = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    host = urlparse(base).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: OLLAMA_URL is not localhost; research excerpts may leave this host.", file=sys.stderr)
    resp = requests.post(base + path, json=payload, timeout=(15, timeout))
    resp.raise_for_status()
    return resp.json()


def embed_many(texts: list[str], model: str) -> list[list[float]]:
    """Generate one vector per text and check that Ollama returned the full batch."""
    if not texts:
        return []
    output = ollama_post("/api/embed", {"model": model, "input": texts}, timeout=420)
    result = output.get("embeddings")
    if not isinstance(result, list) or len(result) != len(texts):
        raise ValueError("Ollama embedding response missing expected number of embeddings")
    return [[float(v) for v in emb] for emb in result]
