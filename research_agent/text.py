"""Text normalization and chunking for document extraction and comparison."""
from __future__ import annotations

import math

# Overlap preserves context between neighboring reference chunks.
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 220


def clean_text(value: str) -> str:
    """Collapse line breaks and repeated whitespace into single spaces."""
    return " ".join((value or "").split())


def split_chunks(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split normalized text into overlapping chunks without a repeated short tail."""
    text = clean_text(text)
    if not text:
        return []
    if size <= overlap:
        raise ValueError("chunk size must be larger than overlap")
    result = []
    for start in range(0, len(text), size - overlap):
        chunk = text[start:start + size]
        if not chunk:
            break
        # Avoid appending a short tail already contained in the preceding chunk.
        if result and start + len(chunk) == len(text) and len(chunk) <= overlap:
            break
        result.append(chunk)
        if start + size >= len(text):
            break
    return result


def cosine(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity; return -1 for empty or incompatible vectors."""
    if len(a) != len(b) or not a:
        return -1.0
    denom = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return sum(x*y for x, y in zip(a, b)) / denom if denom else -1.0
