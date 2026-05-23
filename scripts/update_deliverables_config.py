from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="deliverables/config.json")
    p.add_argument("--title", default="")
    p.add_argument("--course", default="")
    p.add_argument("--group", default="")
    p.add_argument("--date", default="")
    p.add_argument("--pitch-minutes", type=int, default=0)
    p.add_argument("--members", nargs="*", default=None, help="List of member names")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    cfg_path = repo / args.config
    cfg: dict = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    def set_if(name: str, value) -> None:
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        if isinstance(value, int) and value <= 0:
            return
        cfg[name] = value

    set_if("title", args.title)
    set_if("course", args.course)
    set_if("group", args.group)
    set_if("date", args.date)
    if args.pitch_minutes > 0:
        cfg["target_pitch_minutes"] = int(args.pitch_minutes)
    if args.members is not None and len(args.members) > 0:
        cfg["members"] = [str(m) for m in args.members]

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {cfg_path}")


if __name__ == "__main__":
    main()

