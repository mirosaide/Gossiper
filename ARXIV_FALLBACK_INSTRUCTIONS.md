# arXiv sources and daily feed fallback

Source selection is integrated into `research_agent/arxiv.py`. There is no need to copy functions or edit `agent.py`.

## Select a source in `.env`

```ini
# Preserve the original source: recent announcements by category.
ARXIV_SOURCE=rss
ARXIV_RSS_CATEGORIES=astro-ph.EP,astro-ph.IM
```

Use `ARXIV_SOURCE=api` to search the `ARXIV_QUERY` expression within the `LOOKBACK_DAYS` window. Use `ARXIV_SOURCE=auto` to try that search and fall back to the daily feed if the API remains unavailable with HTTP 429 or 503.

The API makes up to three attempts per page. Retry delays are 60 and 180 seconds and may increase according to `Retry-After`. The `rss` mode skips these initial API attempts. Errors other than HTTP 429/503 are propagated without automatic fallback.

## Module responsibilities

- `research_agent/arxiv.py`: source selection, Query API, pagination, dates and retries.
- `arxiv_rss_fallback.py`: requests to the official feed at `https://rss.arxiv.org/atom/` and parsing of daily entries.
- `research_agent/pipeline.py`: receives papers and applies deduplication, selection and analysis.

## Verify

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python agent.py --dry-run
```

Tests simulate external services. The second command uses real arXiv and Ollama services, saves previews and may update the reference cache; it does not send email or advance reviewed-paper history.

## Coverage and limits

The daily feed contains only the announcements available at that time. It does not apply `ARXIV_QUERY`, `LOOKBACK_DAYS` or `PAGE_SIZE` and does not recover the complete history of missed days. `MAX_ARXIV_RESULTS` limits results from both sources.

If both the API and feed fail, the run fails without completing processing. The project does not implement a persistent candidate queue or historical recovery through OAI-PMH. In `auto` mode, check the fallback warning to see when coverage changes from date-based search to daily announcements.
