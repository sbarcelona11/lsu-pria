#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/web-ui"

npm install
npm run build

echo "Built UI to $ROOT_DIR/web-ui/dist"

