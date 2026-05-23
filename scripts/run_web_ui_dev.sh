#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found. Install Node.js first."
  exit 1
fi

echo "Starting API (FastAPI) on :8000 ..."
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi
python3 "$ROOT_DIR/scripts/run_webapp.py" --host 127.0.0.1 --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting React dev server on :5173 ..."
cd "$ROOT_DIR/web-ui"
npm install --silent
npm run dev
