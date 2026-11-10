# Reference documents

Place your papers, proposals or notes in this directory as `.pdf`, `.md` or `.txt`
files. Subdirectories are supported. PDFs must contain extractable text; scans
need OCR before they can be used by the agent.

`research_agent/references.py` extracts and chunks text, generates embeddings
through Ollama and stores vectors in `state/papers.sqlite3`. The content hash
allows the index to be reused until the document changes. The embedding model
is also part of the cache key.

Processing is local when `OLLAMA_URL` points to localhost. The email digest may
contain comparisons derived from these documents. Documents are excluded from
Git; this README and `.gitkeep` preserve only the directory structure.

This README is also a Markdown file and, like every `.md` file in this directory,
is indexed. To index only your own documents, set `REFERENCE_DIR` to a separate
directory.
