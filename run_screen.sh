#!/bin/bash
# Main screening runner that automatically activates a virtual environment when present.

set -euo pipefail

cd "$(dirname "$0")"

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}." python scripts/run_quant_engine.py "$@"
