"""Search the arXiv Query API and select the metadata source."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import os
import re
import sys
import time
from urllib.parse import urlsplit

from defusedxml import ElementTree as ET
import requests

from arxiv_rss_fallback import fetch_daily_rss
from .models import Paper
from .text import clean_text

# Namespaces for Atom elements and OpenSearch pagination totals.
ATOM = "{http://www.w3.org/2005/Atom}"
OPEN_SEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_ID_PATTERN = re.compile(r"(?:\d{4}\.\d{4,5}|[A-Za-z-]+(?:\.[A-Za-z-]+)?/\d{7})(?:v\d+)?$")


def norm_arxiv_id(raw: str) -> str:
    """Validate an arXiv identifier or URL and remove its version suffix."""
    # Identifiers from before 2007 contain a slash (for example, hep-th/9901001).
    aid = raw.strip().rstrip("/")
    if "/abs/" in aid:
        aid = urlsplit(aid).path.split("/abs/", 1)[1]
    if not ARXIV_ID_PATTERN.fullmatch(aid):
        raise ValueError(f"Unexpected arXiv ID: {aid!r}")
    return re.sub(r"v\d+$", "", aid)


def parse_feed(content: bytes) -> tuple[list[Paper], int]:
    """Convert Query API Atom into papers and the total reported by arXiv."""
    root = ET.fromstring(content)
    total_tag = root.find(f"{OPEN_SEARCH}totalResults")
    total = int(total_tag.text or "0") if total_tag is not None else 0
    papers: list[Paper] = []
    for e in root.findall(f"{ATOM}entry"):
        id_tag = e.findtext(f"{ATOM}id", default="")
        if not id_tag:
            continue
        try:
            paper_id = norm_arxiv_id(id_tag)
        except ValueError:
            print(f"WARNING: skipping invalid arXiv ID: {id_tag}", file=sys.stderr)
            continue
        title = clean_text(e.findtext(f"{ATOM}title", default=""))
        abstract = clean_text(e.findtext(f"{ATOM}summary", default=""))
        authors = ", ".join(clean_text(a.findtext(f"{ATOM}name", default="")) for a in e.findall(f"{ATOM}author"))
        categories = ", ".join(c.attrib.get("term", "") for c in e.findall(f"{ATOM}category"))
        papers.append(Paper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            authors=authors,
            published=e.findtext(f"{ATOM}published", default="")[:10],
            updated=e.findtext(f"{ATOM}updated", default="")[:10],
            categories=categories,
            url=f"https://arxiv.org/abs/{paper_id}",
        ))
    return papers, total


def make_session() -> requests.Session:
    """Create an HTTP session identified by the configured contact address."""
    sess = requests.Session()

    sess.headers["User-Agent"] = (
        f"LocalResearchDigest/1.0 "
        f"({os.getenv('CONTACT_EMAIL', 'research monitoring')})"
    )

    return sess


def fetch_arxiv(
    query: str,
    lookback_days: int,
    cap: int,
    page_size: int,
    now: datetime | None = None,
    session: requests.Session | None = None,
) -> list[Paper]:
    """Search by UTC date, paginate results and retry HTTP 429/503 responses.

    The window includes both its starting date and the current date. Results
    are deduplicated by identifier, limited to cap and returned in arXiv order.
    A session and date can be supplied for tests without network access.
    """
    # Treat naive dates as UTC to keep the search window deterministic.
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    today = now.astimezone(timezone.utc).date()
    since_day = today - timedelta(days=lookback_days)

    date_range = (
        f"submittedDate:[{since_day:%Y%m%d}0000 "
        f"TO {today:%Y%m%d}2359]"
    )

    q = f"({query}) AND {date_range}"
    sess = session or make_session()

    papers: dict[str, Paper] = {}
    total = None

    if cap <= 0 or page_size <= 0:
        raise ValueError("cap and page_size must be positive")

    for start in range(0, cap, page_size):

        # Wait between pages to respect the arXiv request rate limit.
        if start:
            time.sleep(3.1)

        params = {
            "search_query": q,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": start,
            "max_results": min(page_size, cap - start),
        }

        # Allow up to three attempts per page; propagate other HTTP errors.
        for attempt in range(3):

            resp = sess.get(
                ARXIV_API,
                params=params,
                timeout=(15, 90)
            )

            # Responses that do not indicate overload bypass the backoff loop.
            if resp.status_code not in (429, 503):
                resp.raise_for_status()
                break

            # The final failure may trigger fallback in auto mode.
            if attempt == 2:
                resp.raise_for_status()

            # Use increasing backoff delays of 60 and 180 seconds between attempts.
            delay = [60, 180][attempt]

            # Retry-After can contain a number of seconds or an HTTP date.
            retry_after = resp.headers.get("Retry-After")

            if retry_after:
                try:
                    if retry_after.strip().isdigit():
                        server_delay = int(retry_after)
                    else:
                        retry_date = parsedate_to_datetime(
                            retry_after
                        )

                        if retry_date.tzinfo is None:
                            retry_date = retry_date.replace(
                                tzinfo=timezone.utc
                            )

                        server_delay = max(
                            0,
                            int(
                                (
                                    retry_date -
                                    datetime.now(timezone.utc)
                                ).total_seconds()
                            ) + 1
                        )

                    delay = max(delay, server_delay)

                except (ValueError, TypeError, OverflowError):
                    pass

            print(
                f"arXiv HTTP {resp.status_code}. "
                f"Attempt {attempt + 1}/3. "
                f"Waiting {delay} seconds...",
                flush=True
            )

            resp.close()
            time.sleep(delay)

        page, total = parse_feed(resp.content)

        for p in page:
            papers[p.paper_id] = p

        print(
            f"arXiv: fetched page starting {start}, "
            f"{len(page)} entries; "
            f"total matches {total}",
            flush=True
        )

        if not page or start + len(page) >= total:
            break

    if total is not None and total > cap:
        print(
            f"WARNING: arXiv query matched {total} papers, "
            f"but cap is {cap}. "
            "Narrow ARXIV_QUERY or increase "
            "MAX_ARXIV_RESULTS to avoid missing papers.",
            file=sys.stderr
        )

    return list(papers.values())


def fetch_papers(query: str, lookback_days: int, cap: int, page_size: int,
                 now: datetime | None = None) -> list[Paper]:
    """Select RSS, API or API with fallback, keeping RSS as the default.

    In auto mode, only HTTP 429/503 after API retries trigger the RSS fallback.
    RSS uses categories without applying the query expression or date window.
    """
    source = os.getenv("ARXIV_SOURCE", "rss").strip().lower()
    if source not in {"rss", "api", "auto"}:
        raise ValueError("ARXIV_SOURCE must be rss, api or auto")
    categories = os.getenv("ARXIV_RSS_CATEGORIES", "astro-ph.EP,astro-ph.IM")
    if source == "rss":
        return fetch_daily_rss(Paper, categories=categories, cap=cap)
    if not query.strip():
        raise ValueError("Set ARXIV_QUERY when ARXIV_SOURCE is api or auto")
    try:
        return fetch_arxiv(query, lookback_days, cap, page_size, now=now)
    except requests.HTTPError as exc:
        if source != "auto" or exc.response is None or exc.response.status_code not in (429, 503):
            raise
        print("WARNING: arXiv Query API unavailable; using daily Atom feed instead.", flush=True)
        return fetch_daily_rss(Paper, categories=categories, cap=cap)
