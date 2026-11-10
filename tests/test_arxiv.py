"""Regression tests for source selection and parsing using simulated HTTP requests."""
from datetime import datetime, timezone
import os
import unittest
from unittest.mock import Mock, patch

import requests

from arxiv_rss_fallback import parse_daily_atom
from research_agent.arxiv import fetch_arxiv, fetch_papers, norm_arxiv_id
from research_agent.models import Paper
from test_agent import SAMPLE_FEED, paper


class ArxivTests(unittest.TestCase):
    """Isolate the network and clock to check the refactored search contracts."""

    def test_legacy_identifier_keeps_archive_prefix(self):
        """Identifiers from before 2007 retain the archive prefix before the slash."""
        self.assertEqual(norm_arxiv_id("https://arxiv.org/abs/hep-th/9901001v3"), "hep-th/9901001")
        self.assertEqual(norm_arxiv_id("astro-ph/0601001v2"), "astro-ph/0601001")

    def test_api_accepts_default_date(self):
        """The optional date defaults to UTC instead of causing an attribute error."""
        session = Mock()
        session.get.return_value = Mock(status_code=200, content=SAMPLE_FEED)
        self.assertEqual(fetch_arxiv("all:radio", 7, 20, 10, session=session)[0].paper_id, paper().paper_id)
        self.assertIn("submittedDate:[", session.get.call_args.kwargs["params"]["search_query"])

    def test_api_query_uses_requested_window(self):
        """Send the requested date window and pagination limits to the API."""
        session = Mock()
        session.get.return_value = Mock(status_code=200, content=SAMPLE_FEED)
        fetch_arxiv("all:radio", 7, 20, 10, datetime(2026, 11, 10, tzinfo=timezone.utc), session)
        params = session.get.call_args.kwargs["params"]
        self.assertEqual(params["search_query"], "(all:radio) AND submittedDate:[202611030000 TO 202611102359]")
        self.assertEqual(params["max_results"], 10)

    def test_default_source_remains_rss(self):
        """Without new configuration, the agent retains its original RSS source."""
        with patch.dict(os.environ, {}, clear=True), \
             patch("research_agent.arxiv.fetch_daily_rss", return_value=[paper()]) as rss, \
             patch("research_agent.arxiv.fetch_arxiv") as api:
            self.assertEqual(fetch_papers("", 7, 20, 10), [paper()])
            rss.assert_called_once_with(Paper, categories="astro-ph.EP,astro-ph.IM", cap=20)
            api.assert_not_called()

    def test_api_source_does_not_fetch_rss(self):
        """API mode searches by expression and date without consulting RSS."""
        with patch.dict(os.environ, {"ARXIV_SOURCE": "api"}), \
             patch("research_agent.arxiv.fetch_arxiv", return_value=[paper()]) as api, \
             patch("research_agent.arxiv.fetch_daily_rss") as rss:
            self.assertEqual(fetch_papers("all:radio", 7, 20, 10), [paper()])
            api.assert_called_once()
            rss.assert_not_called()

    def test_auto_falls_back_only_on_overload(self):
        """HTTP 429 and 503 allow feed fallback after Query API failure."""
        for status in (429, 503):
            with self.subTest(status=status):
                response = requests.Response()
                response.status_code = status
                with patch.dict(os.environ, {"ARXIV_SOURCE": "auto"}), \
                     patch("research_agent.arxiv.fetch_arxiv", side_effect=requests.HTTPError(response=response)), \
                     patch("research_agent.arxiv.fetch_daily_rss", return_value=[paper()]) as rss:
                    self.assertEqual(fetch_papers("all:radio", 7, 20, 10), [paper()])
                    rss.assert_called_once()

    def test_unrelated_errors_and_api_mode_do_not_fall_back(self):
        """Fallback must not hide query errors or override explicit API mode."""
        for source, status in (("auto", 400), ("api", 503)):
            with self.subTest(source=source, status=status):
                response = requests.Response()
                response.status_code = status
                with patch.dict(os.environ, {"ARXIV_SOURCE": source}), \
                     patch("research_agent.arxiv.fetch_arxiv", side_effect=requests.HTTPError(response=response)), \
                     patch("research_agent.arxiv.fetch_daily_rss") as rss:
                    with self.assertRaises(requests.HTTPError):
                        fetch_papers("all:radio", 7, 20, 10)
                    rss.assert_not_called()

    def test_invalid_source_is_rejected(self):
        """Reject an unknown source option before making HTTP requests."""
        with patch.dict(os.environ, {"ARXIV_SOURCE": "typo"}):
            with self.assertRaisesRegex(ValueError, "ARXIV_SOURCE"):
                fetch_papers("all:radio", 7, 20, 10)

    def test_daily_feed_metadata_is_not_part_of_abstract(self):
        """Strip the announcement header and do not use updated as the revision date."""
        feed = b'''<feed xmlns="http://www.w3.org/2005/Atom"
                        xmlns:dc="http://purl.org/dc/elements/1.1/">
          <entry><id>oai:arXiv.org:2610.12345v2</id><title>Radio methods</title>
            <summary>arXiv:2610.12345 Announce Type: new Abstract: Full abstract.</summary>
            <dc:creator>Alice, Bob</dc:creator>
            <published>2026-11-09T00:00:00Z</published>
            <updated>2026-11-10T08:00:00Z</updated><category term="astro-ph.IM"/>
          </entry></feed>'''
        result = parse_daily_atom(feed, Paper, 20)[0]
        self.assertEqual(result.abstract, "Full abstract.")
        self.assertEqual(result.authors, "Alice, Bob")
        self.assertEqual(result.updated, result.published)
        self.assertEqual(result.paper_id, "2610.12345")


if __name__ == "__main__":
    unittest.main()
