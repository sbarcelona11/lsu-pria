from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract downloaded iLSU-T .rar archives into a structured root.")
    p.add_argument("--archives-dir", required=True, help="Folder containing source2/, source3/, etc. with downloaded .rar files")
    p.add_argument("--out-root", required=True, help="Destination root for extracted files")
    p.add_argument("--sources", nargs="+", default=["source2", "source3"])
    p.add_argument("--skip-existing", action="store_true", help="Skip extraction when destination already has files")
    return p.parse_args()


def _require_bsdtar() -> str:
    tool = shutil.which("bsdtar")
    if not tool:
        raise SystemExit("`bsdtar` is required to extract iLSU-T archives on this machine.")
    return tool


def _extract_archive(tool: str, archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [tool, "-xf", str(archive), "-C", str(out_dir)]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"Extraction failed: {archive}")


def _looks_extracted(path: Path) -> bool:
    if not path.exists():
        return False
    for _ in path.rglob("*"):
        return True
    return False


def main() -> None:
    args = parse_args()
    tool = _require_bsdtar()
    archives_dir = Path(args.archives_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for source in args.sources:
        src_dir = archives_dir / source
        if not src_dir.exists():
            raise SystemExit(f"Missing source archive directory: {src_dir}")

        episode_part1 = sorted(src_dir.glob("*_episodes.part1.rar"))
        whisperx_rars = sorted(src_dir.glob("*_whisperx.rar"))

        if not episode_part1:
            raise SystemExit(f"Could not find *_episodes.part1.rar under {src_dir}")
        if not whisperx_rars:
            raise SystemExit(f"Could not find *_whisperx.rar under {src_dir}")

        episode_out = out_root / source / "episodes"
        whisperx_out = out_root / source / "whisperx"

        if not (args.skip_existing and _looks_extracted(episode_out)):
            print(f"extract: {episode_part1[0]} -> {episode_out}")
            _extract_archive(tool, episode_part1[0], episode_out)
            extracted += 1
        else:
            print(f"skip: {episode_out}")

        if not (args.skip_existing and _looks_extracted(whisperx_out)):
            print(f"extract: {whisperx_rars[0]} -> {whisperx_out}")
            _extract_archive(tool, whisperx_rars[0], whisperx_out)
            extracted += 1
        else:
            print(f"skip: {whisperx_out}")

    print(f"Done. extracted_archives={extracted} out_root={out_root}")


if __name__ == "__main__":
    main()

