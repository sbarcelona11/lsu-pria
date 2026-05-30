#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/web-ui"

if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi
npm run build

echo "Built UI to $ROOT_DIR/web-ui/dist"
