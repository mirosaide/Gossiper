"""Regression tests for the original logic and delivery/history update ordering."""
import json
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research_agent.arxiv import norm_arxiv_id, parse_feed
from research_agent.assessment import assess
from research_agent.digest import render_digest
from research_agent.models import Evaluated, Paper
from research_agent.ranking import rank_candidates
from research_agent.storage import already_reviewed, conn_open, mark_reviewed
from research_agent.text import cosine, split_chunks


SAMPLE_FEED = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
<opensearch:totalResults>1</opensearch:totalResults>
<entry>
<id>https://arxiv.org/abs/2610.12345v2</id>
<title> A New  Test for &amp; Signals </title>
<summary>First line\n  second line.</summary>
<published>2026-10-29T00:00:00Z</published>
<updated>2026-10-30T00:00:00Z</updated>
<author><name>Alice Example</name></author>
<author><name>Bob Example</name></author>
<category term="astro-ph.IM"/>
</entry></feed>'''


def paper():
    """Create test metadata containing characters that require HTML escaping."""
    return Paper("2610.12345", "A & B <Research>", "Some abstract <text>.", "A. Researcher",
                 "2026-10-29", "2026-10-30", "astro-ph.IM", "https://arxiv.org/abs/2610.12345")


class AgentTests(unittest.TestCase):
    """Check parsing, ranking, assessment and persistence without external services."""
    def test_feed(self):
        """Check metadata extraction and Atom abstract normalization."""
        parsed, total = parse_feed(SAMPLE_FEED)
        self.assertEqual(total, 1)
        self.assertEqual(parsed[0].paper_id, "2610.12345")
        self.assertEqual(parsed[0].authors, "Alice Example, Bob Example")
        self.assertEqual(parsed[0].abstract, "First line second line.")

    def test_id_validation(self):
        """Reject invalid identifiers and remove valid version suffixes."""
        self.assertEqual(norm_arxiv_id("https://arxiv.org/abs/2610.12345v2"), "2610.12345")
        with self.assertRaises(ValueError):
            norm_arxiv_id("https://example.com/not-an-id?evil=1")

    def test_chunks_and_similarity(self):
        """Check chunk sizes and known vector similarities."""
        chunks = split_chunks("abcd" * 100, size=110, overlap=20)
        self.assertGreater(len(chunks), 3)
        self.assertLessEqual(max(map(len, chunks)), 110)
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0)

    def test_output_escaping_and_full_abstract(self):
        """Preserve the full abstract and escape content inserted into HTML."""
        e = Evaluated(paper(), 0.92, {
            "score": 86, "confidence": "high", "why_relevant": "Direct fit",
            "direct_comparison": "Your paper <A> compared", "research_implication": "Test <x>",
            "limitations": "Abstract only",
        })
        txt, rich = render_digest([e], "2026-10-30", 1, 70)
        self.assertIn("Some abstract <text>.", txt)
        self.assertIn("Some abstract &lt;text&gt;.", rich)
        self.assertNotIn("Your paper <A>", rich)
        self.assertIn("https://arxiv.org/abs/2610.12345", rich)

    def test_db_deduplication(self):
        """Recording the same paper twice must leave a single row."""
        with TemporaryDirectory() as d:
            with closing(conn_open(Path(d) / "db.sqlite3")) as db:
                self.assertEqual(already_reviewed(db), set())
                mark_reviewed(db, [paper()], {paper().paper_id: 89})
                mark_reviewed(db, [paper()], {paper().paper_id: 89})
                self.assertEqual(already_reviewed(db), {"2610.12345"})
                self.assertEqual(db.execute("SELECT count(*) FROM reviewed").fetchone()[0], 1)

    def test_rank_prefers_profile_match(self):
        """The shortlist limit retains the paper closest to the profile."""
        one = paper()
        two = Paper("2610.22222", "Other", "Unrelated", "Author", "2026-10-29", "2026-10-29", "cs.LG", "https://arxiv.org/abs/2610.22222")
        def fake_embed(texts, model):
            """Simulate predictable vectors to isolate semantic ranking."""
            return [[1., 0.] if text.startswith("My profile") or text.startswith("A & B") else [0., 1.] for text in texts]
        with patch("research_agent.ranking.embed_many", side_effect=fake_embed):
            ranked = rank_candidates([two, one], "My profile on astronomy", [], "fake-embed", 1)
        self.assertEqual(ranked[0][1].paper_id, one.paper_id)

    def test_assess_validation(self):
        """Normalize the model score without losing the comparison."""
        obj = {
            "score": 101, "why_relevant": "Matches", "direct_comparison": "Different pipeline",
            "research_implication": "Benchmark it", "limitations": "Uncertain", "confidence": "medium"
        }
        with patch("research_agent.assessment.ollama_post", return_value={"message": {"content": json.dumps(obj)}}):
            result = assess(paper(), "My profile", [], "qwen")
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["direct_comparison"], "Different pipeline")

class PipelineTests(unittest.TestCase):
    """Ensure previews and SMTP failures do not advance reviewed history."""
    def test_state_is_not_advanced_when_smtp_fails(self):
        """Record a paper only after a successful delivery attempt."""
        import os
        from research_agent import pipeline
        with TemporaryDirectory() as d:
            base = Path(d)
            (base / 'research_profile.md').write_text(
                'My research work concerns transient radio signal analysis, noise rejection, '
                'novel candidate ranking, observing constraints, repeatability, and instrument '
                'calibration across telescope backends. The baseline approach combines supervised '
                'classification with careful calibration experiments and validation on held-out data.',
                encoding='utf-8')
            (base / 'references').mkdir()
            (base / 'state').mkdir()
            (base / 'output').mkdir()
            settings = {
                'RESEARCH_PROFILE': 'research_profile.md',
                'REFERENCE_DIR': 'references',
                'ARXIV_QUERY': 'cat:astro-ph.IM',
                'LLM_MODEL': 'fake-local',
                'EMBED_MODEL': 'fake-embed',
                'EMAIL_FROM': 'sender@example.org',
                'EMAIL_TO': 'recipient@example.org',
                'EMAIL_ALWAYS_SEND': 'true',
            }
            result = {
                'score': 92, 'confidence': 'high', 'why_relevant': 'Test fit',
                'direct_comparison': 'A comparable method', 'research_implication': 'Run benchmark',
                'limitations': 'Abstract based',
            }
            with patch.object(pipeline, 'BASE', base), patch.dict(os.environ, settings), \
                 patch('research_agent.pipeline.build_or_reuse_index', return_value=[]), \
                 patch('research_agent.pipeline.fetch_papers', return_value=[paper()]), \
                 patch('research_agent.pipeline.rank_candidates', return_value=[(0.9, paper(), [])]), \
                 patch('research_agent.pipeline.assess', return_value=result), \
                 patch('research_agent.pipeline.send_email', side_effect=RuntimeError('SMTP unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'SMTP unavailable'):
                    pipeline.execute(dry_run=False)
            with closing(conn_open(base / 'state' / 'papers.sqlite3')) as db:
                self.assertEqual(already_reviewed(db), set())
            with patch.object(pipeline, 'BASE', base), patch.dict(os.environ, settings), \
                 patch('research_agent.pipeline.build_or_reuse_index', return_value=[]), \
                 patch('research_agent.pipeline.fetch_papers', return_value=[paper()]), \
                 patch('research_agent.pipeline.rank_candidates', return_value=[(0.9, paper(), [])]), \
                 patch('research_agent.pipeline.assess', return_value=result), \
                 patch('research_agent.pipeline.send_email', return_value=None):
                pipeline.execute(dry_run=False)
            with closing(conn_open(base / 'state' / 'papers.sqlite3')) as db:
                self.assertEqual(already_reviewed(db), {paper().paper_id})

    def test_dry_run_does_not_send_or_advance_state(self):
        """Generate a preview without sending email or changing reviewed history."""
        import os
        from research_agent import pipeline
        with TemporaryDirectory() as d:
            base = Path(d)
            (base / 'research_profile.md').write_text('' + 'Research specifics on methodology and reference methods. ' * 4)
            (base / 'references').mkdir()
            (base / 'state').mkdir()
            (base / 'output').mkdir()
            with patch.object(pipeline, 'BASE', base), patch.dict(os.environ, {'ARXIV_QUERY': 'all:radio'}), \
                 patch('research_agent.pipeline.build_or_reuse_index', return_value=[]), \
                 patch('research_agent.pipeline.fetch_papers', return_value=[]), patch('research_agent.pipeline.send_email') as send:
                pipeline.execute(dry_run=True)
                send.assert_not_called()
            with closing(conn_open(base / 'state' / 'papers.sqlite3')) as db:
                self.assertEqual(already_reviewed(db), set())
            self.assertEqual(len(list((base / 'output').glob('*.html'))), 1)


if __name__ == "__main__":
    unittest.main()
