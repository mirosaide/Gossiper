"""Daily processing orchestration that delegates individual tasks to their modules."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import os
import sys

from .arxiv import fetch_papers
from .assessment import assess
from .config import BASE, env_bool, env_int
from .digest import render_digest
from .mailer import create_email, send_email, valid_email
from .models import Evaluated
from .ranking import rank_candidates
from .references import build_or_reuse_index
from .storage import already_reviewed, conn_open, mark_reviewed


def execute(dry_run: bool) -> int:
    """Run retrieval, selection, assessment, reporting and reviewed-history updates.

    In dry_run mode, save previews without sending email or marking papers as
    reviewed. The reference embedding cache may still be updated.
    """
    # Validate early to avoid calling services with an incomplete research profile.
    profile_path = BASE / os.getenv("RESEARCH_PROFILE", "research_profile.md")
    if not profile_path.exists():
        raise ValueError(f"Missing research profile: {profile_path}")
    profile = profile_path.read_text(encoding="utf-8").strip()
    if len(profile) < 120 or "[Describe" in profile or "[Replace" in profile:
        raise ValueError("Fill in research_profile.md with your actual research before running.")
    query = os.getenv("ARXIV_QUERY", "").strip()
    if os.getenv("ARXIV_SOURCE", "rss").strip().lower() in {"api", "auto"} and not query:
        raise ValueError("Set ARXIV_QUERY in .env; choose a few relevant arXiv categories/keywords.")
    lookup_days = env_int("LOOKBACK_DAYS", 7, 1, 60)
    fetched_cap = env_int("MAX_ARXIV_RESULTS", 500, 20, 5000)
    page_size = env_int("PAGE_SIZE", 100, 10, 1000)
    llm_cap = env_int("MAX_LLM_EVALUATIONS", 40, 1, 500)
    email_cap = env_int("MAX_EMAIL_PAPERS", 12, 1, 100)
    threshold = env_int("RELEVANCE_THRESHOLD", 70, 0, 100)
    model = os.getenv("LLM_MODEL", "qwen3.6:35b-a3b")
    embed_model = os.getenv("EMBED_MODEL", "embeddinggemma")
    now = datetime.now(timezone.utc)
    date = now.strftime("%Y-%m-%d")

    with contextlib.closing(conn_open(BASE / "state" / "papers.sqlite3")) as conn:
        reference_chunks = build_or_reuse_index(conn, BASE / os.getenv("REFERENCE_DIR", "references"), embed_model)
        print(f"Reference index: {len(reference_chunks)} chunks", flush=True)
        fetched = fetch_papers(query, lookup_days, fetched_cap, page_size, now=now)
        # History uses version-independent IDs, so revisions are not treated as new papers.
        seen = already_reviewed(conn)
        fresh = [p for p in fetched if p.paper_id not in seen]
        print(f"New candidates: {len(fresh)} of {len(fetched)} fetched", flush=True)
        examined: list[Evaluated] = []
        failures: list[str] = []
        if fresh:
            for similarity, paper, snippets in rank_candidates(fresh, profile, reference_chunks, embed_model, llm_cap):
                try:
                    analysis = assess(paper, profile, snippets, model)
                    examined.append(Evaluated(paper, similarity, analysis))
                    print(f"  {paper.paper_id} score={analysis['score']} {paper.title[:75]}", flush=True)
                except Exception as exc:
                    # An invalid response must not prevent assessment of the remaining papers.
                    failures.append(paper.paper_id)
                    print(f"ERROR: model assessment failed on {paper.paper_id}: {exc}", file=sys.stderr)
            if failures:
                print(f"WARNING: {len(failures)} paper(s) could not be assessed; they will remain eligible next run.", file=sys.stderr)
        if fresh and not examined and failures:
            raise RuntimeError("All paper assessments failed; refusing to mark any paper as reviewed.")
        # Sort by model score, using semantic similarity to break ties.
        high = sorted((e for e in examined if e.assessment["score"] >= threshold),
                      key=lambda e: (-e.assessment["score"], -e.semantic_similarity))
        if len(high) > email_cap:
            print(f"WARNING: {len(high)} papers passed threshold; email limited to {email_cap}. "
                  "Increase MAX_EMAIL_PAPERS to receive them all.", file=sys.stderr)
        high = high[:email_cap]
        sender = os.getenv("EMAIL_FROM", os.getenv("SMTP_USER", "")).strip()
        recipients = [x.strip() for x in os.getenv("EMAIL_TO", "").split(",") if x.strip()]
        if not dry_run and (not valid_email(sender) or not recipients or any(not valid_email(x) for x in recipients)):
            raise ValueError("Set valid EMAIL_FROM and EMAIL_TO in .env")
        # Previews do not require SMTP credentials or real email addresses.
        msg = create_email(high, date, len(fresh), threshold,
                           sender or "preview@example.invalid", recipients or ["preview@example.invalid"])
        plain, rich = render_digest(high, date, len(fresh), threshold)
        basename = f"{now:%Y%m%d-%H%M%S}-arxiv-digest"
        (BASE / "output" / f"{basename}.txt").write_text(plain, encoding="utf-8")
        (BASE / "output" / f"{basename}.html").write_text(rich, encoding="utf-8")
        print(f"Preview saved as output/{basename}.html", flush=True)

        if dry_run:
            print("DRY RUN: no email sent; reviewed state was not updated.", flush=True)
            return 0
        if high or env_bool("EMAIL_ALWAYS_SEND", False):
            send_email(msg)
            print(f"Sent email with {len(high)} paper(s) to {len(recipients)} recipient(s).", flush=True)
        else:
            print("No papers passed threshold; skipping email (EMAIL_ALWAYS_SEND=false).", flush=True)
        # Advance only after SMTP accepts delivery or no email is needed that day.
        # Candidates outside the shortlist and failed assessments remain unreviewed.
        safe_to_mark = [e.paper for e in examined if e.paper.paper_id not in failures]
        score_lookup = {e.paper.paper_id: e.assessment["score"] for e in examined}
        mark_reviewed(conn, safe_to_mark, score_lookup)
        return 0
