"""Read the official daily arXiv Atom feed.

The feed contains recent announcements for the selected categories; it does
not provide historical searches by date or keyword. The Paper class is passed
as an argument so this module can be reused without importing the pipeline.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any
from urllib.parse import urlsplit

from defusedxml import ElementTree as ET
import requests

_ATOM = "{http://www.w3.org/2005/Atom}"
_DC = "{http://purl.org/dc/elements/1.1/}"
_ID = re.compile(r"^(?:\d{4}\.\d{4,5}|[A-Za-z-]+(?:\.[A-Za-z-]+)?/\d{7})(?:v\d+)?$")
_CATEGORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,39}$")
_FEED_BASE = "https://rss.arxiv.org/atom/"


def _clean(value: str) -> str:
    """Normalize whitespace and line breaks in incoming metadata."""
    return " ".join((value or "").split())


def _normalise_id(raw: str) -> str:
    """Accept URLs, plain IDs and OAI identifiers; remove the paper version."""
    value = raw.strip()
    if "/abs/" in value:
        value = urlsplit(value).path.split("/abs/", 1)[1]
    elif value.startswith("oai:arXiv.org:"):
        value = value[len("oai:arXiv.org:"):]
    if not _ID.fullmatch(value):
        raise ValueError(f"Invalid arXiv ID in daily feed: {raw!r}")
    return re.sub(r"v\d+$", "", value)


def parse_daily_atom(content: bytes, Paper: type, cap: int) -> list[Any]:
    """Convert daily Atom entries into papers, stripping announcement metadata."""
    root = ET.fromstring(content)
    if root.tag != f"{_ATOM}feed":
        raise ValueError("arXiv daily feed returned unexpected XML (not Atom feed)")
    papers: dict[str, Any] = {}
    for entry in root.findall(f"{_ATOM}entry"):
        links = [el.attrib.get("href", "") for el in entry.findall(f"{_ATOM}link")
                 if "/abs/" in el.attrib.get("href", "")]
        raw_id = links[0] if links else entry.findtext(f"{_ATOM}id", default="")
        try:
            paper_id = _normalise_id(raw_id)
        except ValueError:
            print(f"WARNING: ignoring invalid arXiv RSS identifier: {raw_id!r}", file=sys.stderr)
            continue
        abstract = entry.findtext(f"{_ATOM}summary", default="")
        # Daily summaries may start with metadata; the abstract follows Abstract:.
        match = re.search(r"\bAbstract:\s*", abstract, flags=re.IGNORECASE)
        if match:
            abstract = abstract[match.end():]
        authors = entry.findtext(f"{_DC}creator", default="")
        if not authors:
            authors = ", ".join(_clean(a.findtext(f"{_ATOM}name", default=""))
                                for a in entry.findall(f"{_ATOM}author"))
        date = entry.findtext(f"{_ATOM}published", default="")[:10]
        # In RSS, updated is the feed generation time, not the paper revision date.
        # Use the published date without inventing a separate revision date.
        categories = ", ".join(dict.fromkeys(el.attrib.get("term", "")
                             for el in entry.findall(f"{_ATOM}category") if el.attrib.get("term")))
        papers[paper_id] = Paper(
            paper_id=paper_id,
            title=_clean(entry.findtext(f"{_ATOM}title", default="")),
            abstract=_clean(abstract),
            authors=_clean(authors),
            published=date,
            updated=date,
            categories=categories,
            url=f"https://arxiv.org/abs/{paper_id}",
        )
    if len(papers) > cap:
        print(f"WARNING: daily RSS returned {len(papers)} entries; cap={cap} truncates the feed.", file=sys.stderr)
    return list(papers.values())[:cap]


def fetch_daily_rss(Paper: type, categories: str = "astro-ph.EP", cap: int = 500,
                    session: requests.Session | None = None) -> list[Any]:
    """Fetch the daily feed for the requested categories, limited to cap papers.

    The feed API accepts categories joined with '+'. An optional session allows
    HTTP connection reuse or testing without network access.
    """
    if cap <= 0:
        raise ValueError("cap must be a positive integer")
    # Remove duplicate categories while preserving their requested order.
    cats = list(dict.fromkeys(x.strip() for x in categories.split(",") if x.strip()))
    if not cats or len(cats) > 20 or any(not _CATEGORY.fullmatch(c) for c in cats):
        raise ValueError("ARXIV_RSS_CATEGORIES must be 1-20 comma-separated arXiv categories")
    url = _FEED_BASE + "+".join(cats)
    sess = session or requests.Session()
    contact = os.getenv("CONTACT_EMAIL", "").strip()
    sess.headers["User-Agent"] = f"ResearchRadar/1.0 (mailto:{contact})" if contact else "ResearchRadar/1.0"
    print(f"arXiv daily Atom fallback: {url}", flush=True)
    response = sess.get(url, timeout=(15, 90))
    response.raise_for_status()
    found = parse_daily_atom(response.content, Paper, cap)
    print(f"arXiv daily Atom feed: {len(found)} distinct entries; source covers recent announcements only", flush=True)
    return found
