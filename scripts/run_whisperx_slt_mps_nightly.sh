#!/usr/bin/env bash
set -euo pipefail

# Nightly-ish end-to-end SLT run on macOS (MPS) targeting "finishes overnight":
# - caps dataset size for faster iteration
# - uses fewer frames per clip
# - uses small batch size for MPS stability
#
# Usage:
#   ROOT=/abs/path/to/ilsut_extracted \
#   BACKEND_REPO=/abs/path/to/slt-mps \
#   WORK_ROOT=/abs/path/to/runs/whisperx_slt_mps_nightly \
#   ./scripts/run_whisperx_slt_mps_nightly.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${REPO_DIR}/.venv/bin/python3"

if [[ ! -x "$PY_BIN" ]]; then
  echo "Missing venv python: $PY_BIN"
  echo "Fix:"
  echo "  cd \"$REPO_DIR\" && /opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

ROOT="${ROOT:-${REPO_DIR}/data/ilsut_extracted}"
BACKEND_REPO="${BACKEND_REPO:-}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/runs/whisperx_slt_mps_nightly}"

if [[ -z "$BACKEND_REPO" ]]; then
  echo "Set BACKEND_REPO to your SLT backend repo (fork with MPS/native loader)."
  echo "Example:"
  echo "  BACKEND_REPO=/Users/sebastian/Documents/slt-mps"
  exit 1
fi

"$PY_BIN" "${REPO_DIR}/lsupria.py" run-whisperx-slt-pipeline \
  --root "$ROOT" \
  --sources source2 source3 \
  --work-root "$WORK_ROOT" \
  --limit 5000 \
  --sample-fps 6 \
  --max-frames 24 \
  --epochs 10 \
  --batch-size 1 \
  --device mps \
  --backend-repo "$BACKEND_REPO" \
  --backend-loader native \
  --dedup-eval-text train_exact \
  --run-backend

