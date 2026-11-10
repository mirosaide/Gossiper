"""Command-line interface and execution lock for Linux/macOS."""
from __future__ import annotations

import argparse
import fcntl
import os

from dotenv import load_dotenv

from .config import BASE
from .pipeline import execute


def main() -> int:
    """Load .env, parse CLI arguments and prevent concurrent runs with flock."""
    # Environment variables already exported in the shell take precedence over .env.
    load_dotenv(BASE / ".env")
    parser = argparse.ArgumentParser(description="Daily local arXiv research digest")
    parser.add_argument("--dry-run", action="store_true", help="Save previews only. No email or seen-state updates.")
    args = parser.parse_args()
    os.chdir(BASE)
    (BASE / "state").mkdir(exist_ok=True)
    (BASE / "output").mkdir(exist_ok=True)
    # Keep the file descriptor open to hold the lock throughout the pipeline.
    with open(BASE / "state" / "run.lock", "w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another instance is already running; exiting.")
            return 0
        return execute(args.dry_run)
