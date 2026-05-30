#!/usr/bin/env bash
# Install + run the React/Vite dev server on port 5173.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/frontend"

if ! command -v npm >/dev/null 2>&1; then
  echo "[err] npm not found. Install Node.js (>=18): https://nodejs.org/"
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "[info] node_modules missing, running npm install..."
  npm install
fi

echo "[ok] http://localhost:5173"
exec npm run dev
