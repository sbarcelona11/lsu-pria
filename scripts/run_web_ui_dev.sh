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

PY_BIN="python3"
if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PY_BIN="$ROOT_DIR/.venv/bin/python"
fi

if ! "$PY_BIN" -c "import cv2" >/dev/null 2>&1; then
  echo "OpenCV (cv2) is missing in this Python environment."
  echo "Fix:"
  echo "  source $ROOT_DIR/.venv/bin/activate"
  echo "  pip install -r $ROOT_DIR/requirements.txt"
  echo "Note: If you're using Python 3.13+/3.14 and OpenCV wheels aren't available, use Python 3.11/3.12."
  exit 1
fi

"$PY_BIN" "$ROOT_DIR/scripts/run_webapp.py" --host 127.0.0.1 --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting React dev server on :5173 ..."
cd "$ROOT_DIR/web-ui"
if [ -f package-lock.json ]; then
  npm ci --silent
else
  npm install --silent
fi
npm run dev
