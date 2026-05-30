#!/usr/bin/env bash
# Start the FastAPI backend on port 8000 using the project's existing .venv.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "[err] Could not find $VENV_PY — create a venv first:"
  echo "      python3 -m venv .venv && .venv/bin/pip install -r webapp/backend/requirements.txt"
  exit 1
fi

cd "$SCRIPT_DIR/backend"
conda 
if [ ! -f .env ]; then
  echo "[info] backend/.env not found, copying from .env.example."
  cp .env.example .env
fi

# Install missing webapp deps on top of the existing venv (fast no-op if already there).
"$VENV_PY" -m pip install -q -r requirements.txt

echo "[ok] http://localhost:8000  ·  http://localhost:8000/docs"
exec "$VENV_PY" -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
