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
#
# Resume an existing run (reuses the same WORK_ROOT and keeps backend checkpoints):
#   ROOT=/abs/path/to/ilsut_extracted \
#   BACKEND_REPO=/abs/path/to/slt-mps \
#   RESUME_FROM=/abs/path/to/runs/whisperx_slt_mps_nightly_YYYYMMDD_HHMM \
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
RESUME_FROM="${RESUME_FROM:-}"
WORK_ROOT="${WORK_ROOT:-${REPO_DIR}/runs/whisperx_slt_mps_nightly}"

if [[ -n "$RESUME_FROM" ]]; then
  WORK_ROOT="$RESUME_FROM"
fi

if [[ -z "$BACKEND_REPO" ]]; then
  echo "Set BACKEND_REPO to your SLT backend repo (fork with MPS/native loader)."
  echo "Example:"
  echo "  BACKEND_REPO=/Users/sebastian/Documents/slt-mps"
  exit 1
fi

if [[ -d "$WORK_ROOT/train/external_backend_model" ]]; then
  if ls "$WORK_ROOT/train/external_backend_model"/*.ckpt >/dev/null 2>&1; then
    echo "Resuming backend from existing checkpoints under:"
    echo "  $WORK_ROOT/train/external_backend_model"
  fi
fi

# Log everything to a timestamped file inside the run folder too.
mkdir -p "$WORK_ROOT"
LOG_PATH="$WORK_ROOT/run.log"
echo "Logging to: $LOG_PATH"
echo "Started at: $(date +\"%Y-%m-%dT%H:%M:%S%z\")" | tee -a "$LOG_PATH"

set -o pipefail
"$PY_BIN" "${REPO_DIR}/lsupria.py" run-whisperx-slt-pipeline \
  --root "$ROOT" \
  --sources source2 source3 \
  --work-root "$WORK_ROOT" \
  --reuse-existing \
  --limit 5000 \
  --max-clips 5000 \
  --sample-fps 6 \
  --max-frames 24 \
  --epochs 10 \
  --batch-size 1 \
  --device mps \
  --backend-repo "$BACKEND_REPO" \
  --backend-loader native \
  --dedup-eval-text train_exact \
  --run-backend | tee -a "$LOG_PATH"
