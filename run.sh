#!/bin/bash
# SWE-bench White Agent startup script for AgentBeats controller

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists (local development)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Use AGENT_URL if set by controller, otherwise no public URL
if [ -n "$AGENT_URL" ]; then
    python main.py serve --host "${HOST:-0.0.0.0}" --port "${AGENT_PORT:-9002}" --url "$AGENT_URL"
else
    python main.py serve --host "${HOST:-0.0.0.0}" --port "${AGENT_PORT:-9002}"
fi
