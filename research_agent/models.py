"""Data structures shared by the agent stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Paper:
    """Paper metadata with a version-independent identifier and canonical URL."""
    paper_id: str
    title: str
    abstract: str
    authors: str
    published: str
    updated: str
    categories: str
    url: str


@dataclass
class Evaluated:
    """A paper together with its semantic similarity and model assessment."""
    paper: Paper
    semantic_similarity: float
    assessment: dict[str, Any]
