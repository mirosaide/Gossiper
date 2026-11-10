# arXiv Research Radar — local research agent

This project retrieves papers from arXiv, compares their abstracts with your research profile and reference documents, and prepares a digest ranked by relevance. Analysis and embeddings run locally through Ollama; history and the reference index are stored in SQLite. The digest can be sent over SMTP in plain text and HTML.

The entry point remains `agent.py`. The implementation is split into the `research_agent/` package, with English descriptions of modules, classes and functions, as well as inline comments. Prompts, runtime messages and reports are also in English.

## 1. Project and model requirements

### Software

| Component | Requirement and purpose |
|---|---|
| Python | **3.10 or later**, with `venv` and `pip` |
| Operating system | Linux/DGX OS or macOS; the execution lock uses `fcntl`. On Windows, use a Linux environment such as WSL |
| Ollama | A version supporting `qwen3.6:35b-a3b`, `/api/chat`, structured JSON output and `/api/embed`; prefer a current stable release |
| Python libraries | `requests`, `python-dotenv`, `pypdf` and `defusedxml`, within the ranges in `requirements.txt` |
| SQLite | Included with Python; no separate database server is needed |
| Scheduling | `systemd` is required only for the Linux installer; manual execution also works on macOS |
| Network | Required to download models, query arXiv and deliver email to the SMTP server |

### Analysis model: `qwen3.6:35b-a3b`

This is the model configured in `LLM_MODEL`. It receives the title, abstract, research profile and reference excerpts, then returns a relevance score, comparison, implications and limitations. The official catalog checked on **November 10, 2026** lists approximately **23 GB** for this tag and a **Q4_K_M** GGUF variant. The size may change when the tag is updated. [Model in the Ollama catalog](https://ollama.com/library/qwen3.6:35b-a3b).

In this project, `research_agent/assessment.py` requests an **8,192-token** context window, up to **1,000 output tokens**, `temperature=0` and `think=false`. These are program settings, not the model's advertised maximum context. The code uses text only, even if the model supports other modalities. [Chat API parameters](https://docs.ollama.com/api/chat).

For hardware planning, use the following **practical project estimates, not official minimum requirements or measured performance results**:

| Resource | Recommended planning allowance |
|---|---|
| Free disk space | Reserve **40 GB or more** for both models, downloads and local files; documents and reports may need additional space |
| RAM/unified memory | **32 GB is a tight configuration** for approximately 23 GB of weights plus the operating system and context; prefer **48–64 GB or more** of total capacity |
| Dedicated GPU | To try to keep analysis entirely on the GPU, plan for **32 GB of VRAM or more**; actual usage depends on the backend, context, cache and concurrent processes |
| CPU | CPU or partial CPU execution may be slower; it needs sufficient memory and may exceed the 600-second assessment timeout |
| DGX Spark | A unit with **128 GB of unified memory** provides headroom for these models; check available memory and Ollama/driver support on the machine |

Disk size is not equivalent to inference memory usage. System RAM and dedicated VRAM also do not automatically form a single unified memory pool. Check the actual CPU/GPU allocation with `ollama ps`. [Ollama memory diagnostics](https://docs.ollama.com/faq#how-can-i-tell-if-my-model-was-loaded-onto-the-gpu), [DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).

### Embedding model: `embeddinggemma`

This is the model configured in `EMBED_MODEL`. It converts the profile, abstracts and references into vectors for semantic comparison. The catalog lists **300 million parameters**, approximately **622 MB**, a **2K-token** context window and a requirement for **Ollama 0.11.10 or later**. That minimum version applies to EmbeddingGemma; it does not guarantee Qwen3.6 compatibility. [Official model and requirements](https://ollama.com/library/embeddinggemma).

The index stores the model name alongside each vector. Changing `EMBED_MODEL` creates or reuses a separate index. If you replace the weights while keeping exactly the same model name, the existing cache does not automatically detect that change.

## 2. Prepare the environment

Install Ollama on the machine that will run the models, following the [official documentation](https://docs.ollama.com/quickstart). For DGX Spark, also consult the [NVIDIA guide](https://build.nvidia.com/spark/open-webui). Then run:

```bash
ollama --version
ollama pull qwen3.6:35b-a3b
ollama pull embeddinggemma
ollama list
```

In the project directory, create an environment **on the machine where it will run**:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# Copy the example only if a personal configuration does not already exist.
cp -n .env.example .env
chmod 600 .env
```

A `.venv` copied from Linux will not work on macOS and should not be shared through Git. If an environment from another machine already exists, rename it before creating the local environment.

The program uses the server at `http://127.0.0.1:11434`. If Ollama is not already running as an application or service, start `ollama serve` in another terminal. Check the server and both models:

```bash
curl -fsS http://127.0.0.1:11434/api/tags

# Check that the analysis model responds with the settings used by the agent.
curl -fsS http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6:35b-a3b","messages":[{"role":"user","content":"Return a JSON object with status set to ok."}],"stream":false,"format":"json","think":false,"options":{"num_ctx":8192,"num_predict":100}}'

# Check that the embedding model is available.
curl -fsS http://127.0.0.1:11434/api/embed \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma","input":["Radio astronomy research"]}'

ollama ps
```

These installation and verification commands are instructions for your machine; the agent does not run them automatically. The Python libraries communicate with Ollama over HTTP, so this project does not require PyTorch or the Ollama Python SDK.

## 3. Configure your research and the arXiv source

Edit `research_profile.md` with your scientific questions, methods, data, instruments, metrics and topics to exclude. The profile must contain at least 120 characters and must not contain the example markers `[Describe` or `[Replace`.

Place `.pdf`, `.md` or `.txt` documents in `references/`. PDFs need extractable text: the program does not perform OCR. The index is updated when content changes and removes deleted sources. Extraction is limited to 120,000 characters per file, split into 1,800-character chunks with 220-character overlap.

The `.env` file defines the configuration. Variables already exported in the terminal take precedence. Three search modes are available:

| `ARXIV_SOURCE` | Behavior |
|---|---|
| `rss` — default | Uses the official daily feed for `ARXIV_RSS_CATEGORIES`, preserving the project's previous behavior |
| `api` | Searches `ARXIV_QUERY` within the `LOOKBACK_DAYS` window, with pagination |
| `auto` | Tries the API and falls back to the daily feed only if the API still returns HTTP 429/503 after retries |

RSS example:

```ini
ARXIV_SOURCE=rss
ARXIV_RSS_CATEGORIES=astro-ph.EP,astro-ph.IM
```

API example:

```ini
ARXIV_SOURCE=api
ARXIV_QUERY=(cat:astro-ph.IM OR cat:astro-ph.EP) AND (all:radio OR all:technosignature)
LOOKBACK_DAYS=7
PAGE_SIZE=100
```

The API uses UTC dates, including both the starting date and the current date. RSS contains recent announcements and **does not apply** `ARXIV_QUERY`, `LOOKBACK_DAYS` or `PAGE_SIZE`; it does not retrieve a historical window. In `auto` mode, coverage changes when fallback occurs, with a terminal warning. API pages are separated by a 3.1-second wait; HTTP 429/503 retries wait 60 and 180 seconds, or longer if required by `Retry-After`.

See [ARXIV_FALLBACK_INSTRUCTIONS.md](ARXIV_FALLBACK_INSTRUCTIONS.md) for details. Selecting a source no longer requires code changes.

## 4. Configure email

Set the details provided by your SMTP service in `.env`:

```ini
SMTP_HOST=smtp.example.org
SMTP_PORT=587
SMTP_USER=your-email@example.org
SMTP_PASSWORD=replace-with-app-password
EMAIL_FROM=your-email@example.org
EMAIL_TO=recipient@example.org
EMAIL_ALWAYS_SEND=false
```

Port 465 uses implicit TLS; other configured ports use STARTTLS before authentication. `EMAIL_TO` accepts multiple comma-separated recipients. Use the SMTP credential or application password appropriate for your provider; OAuth2 is not implemented.

Credentials stay in `.env`, which is excluded from Git. The model does not receive SMTP credentials. If `OLLAMA_URL` points to another server, research excerpts will be sent to that address. Email may contain comparisons derived from your private documents.

## 5. Run and verify

First, generate a preview:

```bash
.venv/bin/python agent.py --dry-run
```

This mode queries arXiv, uses Ollama and saves HTML/plain-text files in `output/`, but does not send email or mark papers as reviewed. It may create or update the embedding cache in `state/papers.sqlite3`.

For normal execution:

```bash
.venv/bin/python agent.py
```

Email is sent if any papers meet the threshold, or always when `EMAIL_ALWAYS_SEND=true`. Reviewed history is updated only after SMTP accepts the message, or after a day when no email is needed. SMTP acceptance does not guarantee inbox delivery. Failed assessments and candidates outside the semantic shortlist remain eligible for another run, provided they appear again in the queried source.

Each digest includes the title, authors, date, category, arXiv link, full abstract, score, confidence, research comparison, next step and limitations.

To check the logic without contacting arXiv, Ollama or SMTP:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python agent.py --help
bash -n install_timer.sh
```

Tests use simulated responses and temporary SQLite databases. They cover parsing, source selection, fallback, caching, ranking, assessment validation, HTML, deduplication and preservation of history when SMTP fails or `--dry-run` is used.

## 6. Module structure and processing flow

| File | Responsibility |
|---|---|
| `agent.py` | Entry point and top-level error handling |
| `research_agent/__init__.py` | Package description; importing does not run the agent |
| `research_agent/cli.py` | Arguments, `.env` loading, directories and concurrent execution lock |
| `research_agent/config.py` | Project root and boolean/integer option validation |
| `research_agent/models.py` | `Paper` and `Evaluated` data structures |
| `research_agent/arxiv.py` | Query API, identifiers, pagination, retries and RSS/API/auto selection |
| `arxiv_rss_fallback.py` | Retrieval and parsing of the official daily feed |
| `research_agent/text.py` | Normalization, text chunking and cosine similarity |
| `research_agent/storage.py` | SQLite schema and reviewed-paper history |
| `research_agent/references.py` | Document extraction, caching by hash/model and excerpt selection |
| `research_agent/ollama.py` | HTTP client and batched embedding generation |
| `research_agent/ranking.py` | Similarity shortlisting before detailed analysis |
| `research_agent/assessment.py` | Prompt, JSON schema and assessment normalization |
| `research_agent/digest.py` | Plain-text and HTML reports with escaped content |
| `research_agent/mailer.py` | MIME messages, addresses and SMTP delivery with TLS |
| `research_agent/pipeline.py` | Stage coordination and reviewed-history update timing |
| `tests/` | Regression tests without access to external services |
| `install_timer.sh` | Daily scheduling setup for Linux/systemd |

```text
CLI → reference index → arXiv (RSS/API) → remove previously reviewed IDs
    → embeddings → shortlist → Qwen assessment → filter and rank
    → preview → SMTP delivery when needed → SQLite history
```

Similarity combines **65% of the profile similarity** and **35% of the closest reference similarity**. Without references, only the profile is used. The final score comes from Qwen; similarity serves as a tiebreaker. The database schema and the `state/` and `output/` paths are preserved, so no migration is required.

To reuse functions, import their responsible module, for example:

```python
from research_agent.models import Paper
from research_agent.arxiv import fetch_arxiv
from research_agent.digest import render_digest
```

Functions previously imported from `agent` are now imported from these modules. The `python agent.py` command and `--dry-run` interface remain available.

## 7. Main settings

| Variable | Default | Effect |
|---|---|---|
| `RESEARCH_PROFILE` | `research_profile.md` | Personal research profile |
| `REFERENCE_DIR` | `references` | Local document directory |
| `ARXIV_SOURCE` | `rss` | Metadata source |
| `ARXIV_RSS_CATEGORIES` | `astro-ph.EP,astro-ph.IM` | Daily feed categories |
| `ARXIV_QUERY` | No default in code | Required expression in `api` and `auto` modes |
| `CONTACT_EMAIL` | Empty | Contact identification in arXiv requests |
| `LOOKBACK_DAYS` | 7 | Query API date window |
| `MAX_ARXIV_RESULTS` | 500 | Maximum papers retrieved from either source |
| `PAGE_SIZE` | 100 | Query API page size |
| `MAX_LLM_EVALUATIONS` | 40 | Maximum detailed assessments per run |
| `MAX_EMAIL_PAPERS` | 12 | Maximum papers included in the email |
| `RELEVANCE_THRESHOLD` | 70 | Minimum inclusion score, between 0 and 100 |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Inference server address |
| `LLM_MODEL` | `qwen3.6:35b-a3b` | Analysis model |
| `EMBED_MODEL` | `embeddinggemma` | Embedding model |
| `EMAIL_ALWAYS_SEND` | `false` | Send even when no papers meet the threshold |

## 8. Schedule daily at 07:00 London time

On Linux/DGX OS with systemd, after configuring and testing the project:

```bash
bash install_timer.sh
systemctl --user list-timers arxiv-research-digest.timer
journalctl --user -u arxiv-research-digest.service -n 150 --no-pager
```

The timer uses `Europe/London`, follows daylight-saving changes and catches up on missed runs with `Persistent=true`. The installer requires a path without spaces. To keep the service available after logout, when needed:

```bash
sudo loginctl enable-linger "$USER"
```

To run through the service or stop scheduling:

```bash
systemctl --user start arxiv-research-digest.service
systemctl --user disable --now arxiv-research-digest.timer
```

The service does not install or start Ollama; the machine and server must be available. On macOS, run manually or configure an appropriate scheduler; `install_timer.sh` is specific to systemd.

## 9. Limitations

- Comparisons use **abstracts**, not the full text of new papers. Scores and confidence labels are model judgments.
- Semantic shortlisting and retrieval limits may omit relevant papers. Increasing limits requires more processing.
- Assessed papers excluded by `MAX_EMAIL_PAPERS` are still marked as reviewed after a completed run. Increase this limit if you want every paper that meets the threshold.
- Identifiers are stored without a version suffix, so a paper revision is not treated as a new paper.
- RSS cannot automatically recover announcements from missed days. There is no persistent pending-candidate queue.
- Formulas, tables and images may not be extracted correctly from PDFs. Profile and excerpt limits reduce the context sent to the model.

To reset history, first stop scheduling and back up `state/papers.sqlite3` before moving it out of the configured path. The next run will rebuild the index and may reassess previous papers.

