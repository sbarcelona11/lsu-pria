#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Cleaning Python caches..."
find "$ROOT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$ROOT_DIR" -type d -name ".pytest_cache" -prune -exec rm -rf {} + 2>/dev/null || true

echo "Cleaning local run/tmp artifacts..."
rm -rf "$ROOT_DIR/runs"/* 2>/dev/null || true
rm -rf "$ROOT_DIR/tmp"/* 2>/dev/null || true

echo "Cleaning web-ui build artifacts..."
rm -rf "$ROOT_DIR/web-ui/dist" 2>/dev/null || true
rm -f "$ROOT_DIR/web-ui/tsconfig.tsbuildinfo" 2>/dev/null || true

echo "Done."

