from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--session-id", type=str, required=True)
    p.add_argument("--out", type=str, required=True, help="Output JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    url = args.base_url.rstrip("/") + f"/api/session/{args.session_id}/export"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    payload = r.json()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
