#!/usr/bin/env python3
"""Agent entry point: run ``python agent.py [--dry-run]``.

The implementation lives in the research_agent package. This file preserves
the command used in the terminal and the systemd service.
"""
import sys

from research_agent.cli import main


if __name__ == "__main__":
    # The CLI returns an exit code; failures remain visible in the service logs.
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
