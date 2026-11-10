"""Regression test for the SQLite reference cache after modularization."""
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from research_agent.references import build_or_reuse_index
from research_agent.storage import conn_open


class ReferenceTests(unittest.TestCase):
    """Check index reuse, updates and removal without Ollama."""

    def test_cache_tracks_content_and_model(self):
        """Reuse valid vectors and keep caches for different models separate."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "references"
            folder.mkdir()
            document = folder / "methods.txt"
            document.write_text("Original radio methods", encoding="utf-8")
            with closing(conn_open(root / "state.sqlite3")) as conn, \
                 patch("research_agent.references.embed_many", return_value=[[1.0, 0.0]]) as embed:
                first = build_or_reuse_index(conn, folder, "model-a")
                self.assertEqual(build_or_reuse_index(conn, folder, "model-a"), first)
                self.assertEqual(embed.call_count, 1)
                build_or_reuse_index(conn, folder, "model-b")
                self.assertEqual(embed.call_count, 2)
                document.write_text("Revised radio methods", encoding="utf-8")
                changed = build_or_reuse_index(conn, folder, "model-a")
                self.assertIn("Revised radio methods", changed[0][1])
                self.assertEqual(embed.call_count, 3)
                document.unlink()
                self.assertEqual(build_or_reuse_index(conn, folder, "model-a"), [])
                # Cleanup must not delete the index belonging to the other model.
                self.assertEqual(conn.execute("SELECT count(*) FROM chunks WHERE model='model-b'").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
