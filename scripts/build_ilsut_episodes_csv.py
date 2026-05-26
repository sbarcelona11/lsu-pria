from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


@dataclass
class MatchResult:
    source: str
    episode_id: str
    video_path: str
    whisperx_path: str
    match_type: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build an iLSU-T episodes.csv-style file directly from extracted source folders "
            "(videos + WhisperX JSON)."
        )
    )
    p.add_argument("--root", required=True, help="Extracted iLSU-T root, e.g. data/ilsut_extracted")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--sources", nargs="*", default=None, help="Optional sources to scan, e.g. source2 source3")
    p.add_argument("--video-dir-name", default="episodes", help="Subdirectory name containing episode videos")
    p.add_argument("--whisperx-dir-name", default="whisperx", help="Subdirectory name containing WhisperX JSON files")
    p.add_argument("--include-unmatched", action="store_true", help="Keep video rows even when no WhisperX JSON matches")
    p.add_argument("--strict", action="store_true", help="Fail if any selected source has zero matched pairs")
    return p.parse_args()


def _iter_sources(root: Path, sources_arg: list[str] | None) -> list[Path]:
    if sources_arg:
        sources = [root / source for source in sources_arg]
    else:
        sources = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not sources:
        raise SystemExit(f"No source directories found under {root}")
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise SystemExit(f"Missing source directories: {', '.join(missing)}")
    return sources


def _relative_stem(base_dir: Path, path: Path) -> str:
    return path.relative_to(base_dir).with_suffix("").as_posix()


def _video_priority(path: Path) -> tuple[int, str]:
    ext = path.suffix.lower()
    try:
        idx = VIDEO_EXTENSIONS.index(ext)
    except ValueError:
        idx = len(VIDEO_EXTENSIONS)
    return idx, path.as_posix()


def _collect_videos(videos_dir: Path) -> list[Path]:
    by_rel_stem: dict[str, Path] = {}
    candidates = sorted(path for path in videos_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    for path in candidates:
        rel_stem = _relative_stem(videos_dir, path)
        current = by_rel_stem.get(rel_stem)
        if current is None or _video_priority(path) < _video_priority(current):
            by_rel_stem[rel_stem] = path
    return sorted(by_rel_stem.values(), key=lambda p: p.relative_to(videos_dir).as_posix())


def _collect_whisperx(whisperx_dir: Path) -> list[Path]:
    return sorted(path for path in whisperx_dir.rglob("*.json") if path.is_file())


def _build_json_indexes(whisperx_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    by_relative_stem: dict[str, Path] = {}
    by_name_stem: dict[str, Path] = {}
    for path in _collect_whisperx(whisperx_dir):
        by_relative_stem.setdefault(_relative_stem(whisperx_dir, path), path)
        by_name_stem.setdefault(path.stem, path)
    return by_relative_stem, by_name_stem


def _resolve_json(
    video_path: Path,
    videos_dir: Path,
    whisperx_dir: Path,
    by_relative_stem: dict[str, Path],
    by_name_stem: dict[str, Path],
) -> tuple[Path | None, str]:
    rel_stem = _relative_stem(videos_dir, video_path)
    if rel_stem in by_relative_stem:
        return by_relative_stem[rel_stem], "relative_stem"

    direct_json = whisperx_dir / f"{rel_stem}.json"
    if direct_json.exists():
        return direct_json, "relative_path"

    name_stem = video_path.stem
    if name_stem in by_name_stem:
        return by_name_stem[name_stem], "filename_stem"

    return None, "missing"


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    rows: list[MatchResult] = []
    total_videos = 0
    total_matched = 0
    total_unmatched = 0
    matched_by_source: dict[str, int] = {}

    for source_dir in _iter_sources(root, args.sources):
        source = source_dir.name
        videos_dir = source_dir / args.video_dir_name
        whisperx_dir = source_dir / args.whisperx_dir_name
        if not videos_dir.exists():
            print(f"skip: {source} has no {videos_dir.name}/ directory")
            continue
        if not whisperx_dir.exists():
            if args.include_unmatched:
                print(f"warn: {source} has no {whisperx_dir.name}/ directory; rows will be unmatched")
            else:
                print(f"skip: {source} has no {whisperx_dir.name}/ directory")
                continue

        videos = _collect_videos(videos_dir)
        by_relative_stem, by_name_stem = _build_json_indexes(whisperx_dir) if whisperx_dir.exists() else ({}, {})

        source_matched = 0
        for video_path in videos:
            total_videos += 1
            matched_json, match_type = _resolve_json(video_path, videos_dir, whisperx_dir, by_relative_stem, by_name_stem)
            if matched_json is None and not args.include_unmatched:
                total_unmatched += 1
                continue

            if matched_json is not None:
                source_matched += 1
                total_matched += 1
                whisperx_rel = matched_json.relative_to(root).as_posix()
            else:
                total_unmatched += 1
                whisperx_rel = ""

            rel_video = video_path.relative_to(root).as_posix()
            rel_stem = _relative_stem(videos_dir, video_path)
            rows.append(
                MatchResult(
                    source=source,
                    episode_id=f"{source}:{rel_stem}",
                    video_path=rel_video,
                    whisperx_path=whisperx_rel,
                    match_type=match_type,
                )
            )

        matched_by_source[source] = source_matched

    if not rows:
        raise SystemExit("No rows generated. Check that your extracted root contains source*/episodes and source*/whisperx.")
    if args.strict:
        empty_sources = [source for source, count in matched_by_source.items() if count == 0]
        if empty_sources:
            raise SystemExit(f"Strict mode: zero matched pairs for {', '.join(empty_sources)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode_id", "source", "video_path", "whisperx_path", "match_type"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "episode_id": row.episode_id,
                    "source": row.source,
                    "video_path": row.video_path,
                    "whisperx_path": row.whisperx_path,
                    "match_type": row.match_type,
                }
            )

    print(
        f"Wrote: {out_path} rows={len(rows)} matched={total_matched} "
        f"unmatched={total_unmatched} scanned_videos={total_videos}"
    )


if __name__ == "__main__":
    main()
