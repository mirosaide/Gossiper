#!/usr/bin/env bash
# Install the daily service and timer in user systemd (Linux only).
# Run with: bash install_timer.sh, after preparing .venv and .env.
set -euo pipefail
# Resolve the project root even when the installer is called from another directory.
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$PROJECT_DIR" == *' '* ]]; then
  echo 'Please move this project to a path without spaces (systemd unit generation).' >&2
  exit 1
fi
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  echo 'Create the Python virtual environment first (see README.md).' >&2
  exit 1
fi
# Units belong to the current user and do not require a system-wide installation.
mkdir -p "$HOME/.config/systemd/user"
# The service uses the project Python interpreter without requiring shell activation.
cat > "$HOME/.config/systemd/user/arxiv-research-digest.service" <<EOF
[Unit]
Description=AI-assisted arXiv Research Digest
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/agent.py
TimeoutStartSec=6h
EOF
# Europe/London follows daylight-saving changes; Persistent catches up on missed runs.
cat > "$HOME/.config/systemd/user/arxiv-research-digest.timer" <<'EOF'
[Unit]
Description=Daily arXiv research digest at 07:00 London time

[Timer]
OnCalendar=*-*-* 07:00:00 Europe/London
Persistent=true
Unit=arxiv-research-digest.service

[Install]
WantedBy=timers.target
EOF
# Reload unit definitions and enable the schedule without manually starting the pipeline.
systemctl --user daemon-reload
systemctl --user enable --now arxiv-research-digest.timer
echo "Timer installed. Inspect: systemctl --user list-timers arxiv-research-digest.timer"
echo "For headless operation after logout, you may need: sudo loginctl enable-linger '$USER'"
