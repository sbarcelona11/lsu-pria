from __future__ import annotations

import argparse
from pathlib import Path

import requests

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--session-id", type=str, required=True)
    p.add_argument("--out", type=str, required=True, help="Output CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    url = args.base_url.rstrip("/") + f"/api/session/{args.session_id}/export.csv"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    csv_text = r.text
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(csv_text, encoding="utf-8")
    rows = max(0, len(csv_text.strip().splitlines()) - 1) if csv_text.strip() else 0
    print(f"Saved: {out_path} (rows={rows})")


if __name__ == "__main__":
    main()
