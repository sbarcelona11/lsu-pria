from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from ilsut_utils import (
    build_file_index,
    load_csv_rows,
    load_json,
    normalize_token,
    resolve_dataset_path,
)


@dataclass
class KeywordRule:
    label: str
    patterns: list[re.Pattern]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a weakly-labeled manifest from iLSU-T WhisperX JSON transcripts.\n"
            "This does NOT download iLSU-T; it expects you to have extracted the dataset locally."
        )
    )
    p.add_argument("--episodes-csv", required=True, help="Path to iLSU-T episodes CSV (from the iLSU-T repo data/)")
    p.add_argument("--root", required=True, help="Local root folder where iLSU-T files were extracted")
    p.add_argument("--whisperx-col", default="", help="Optional override for WhisperX JSON column")
    p.add_argument("--video-col", default="", help="Optional override for episode video column")
    p.add_argument("--episode-id-col", default="", help="Optional override for episode id column")
    p.add_argument("--source-col", default="", help="Optional override for source column")
    p.add_argument("--out", required=True, help="Output manifest CSV")
    p.add_argument("--keywords", required=True, help="JSON file mapping label -> list of patterns/keywords")
    p.add_argument("--pre-ms", type=int, default=400, help="Milliseconds to include before a matched word")
    p.add_argument("--post-ms", type=int, default=800, help="Milliseconds to include after a matched word")
    p.add_argument("--min-word-ms", type=int, default=80)
    p.add_argument("--max-seg-ms", type=int, default=4000, help="Clamp segments to this max duration")
    p.add_argument("--limit", type=int, default=0, help="Limit number of episodes (0 = no limit)")
    p.add_argument(
        "--path-mode",
        choices=["auto", "relative", "filename"],
        default="auto",
        help="How to resolve stale paths from the official episodes CSV",
    )
    return p.parse_args()


def _load_keyword_rules(path: Path) -> list[KeywordRule]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("keywords JSON must be an object: {label: [patterns...]}")
    rules: list[KeywordRule] = []
    for label, pats in raw.items():
        if not isinstance(pats, list) or not pats:
            continue
        compiled = []
        for p in pats:
            s = str(p).strip()
            if not s:
                continue
            # Treat as regex if it contains regex meta, otherwise match whole word.
            if any(ch in s for ch in [".", "?", "+", "*", "(", ")", "[", "]", "|", "^", "$", "\\"]):
                rx = re.compile(s, flags=re.IGNORECASE)
            else:
                s_norm = normalize_token(s)
                rx = re.compile(rf"\\b{re.escape(s_norm)}\\b", flags=re.IGNORECASE)
            compiled.append(rx)
        if compiled:
            rules.append(KeywordRule(label=str(label), patterns=compiled))
    if not rules:
        raise SystemExit("No keyword rules loaded. Provide a non-empty JSON mapping label -> patterns.")
    return rules


def _iter_whisperx_words(obj: dict) -> Iterable[dict]:
    # WhisperX JSON usually contains "segments": [{words:[{start,end,word},...]}, ...]
    segs = obj.get("segments")
    if isinstance(segs, list):
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            words = seg.get("words")
            if isinstance(words, list):
                for w in words:
                    if isinstance(w, dict):
                        yield w


def _to_ms(x: object) -> Optional[int]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        return int(round(v * 1000.0))
    except Exception:
        return None


def _match_label(word: str, rules: list[KeywordRule]) -> Optional[str]:
    w = normalize_token(word)
    if not w:
        return None
    for r in rules:
        for rx in r.patterns:
            if rx.search(w):
                return r.label
    return None


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    episodes_csv = Path(args.episodes_csv)
    out_path = Path(args.out)
    rules = _load_keyword_rules(Path(args.keywords))

    if not episodes_csv.exists():
        raise SystemExit(f"episodes csv not found: {episodes_csv}")
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    rows, detected_cols = load_csv_rows(episodes_csv)
    episode_id_col = args.episode_id_col or detected_cols.episode_id
    source_col = args.source_col or (detected_cols.source or "")
    video_col = args.video_col or detected_cols.video_path
    whisperx_col = args.whisperx_col or detected_cols.whisperx_path

    path_index = build_file_index(root) if args.path_mode in ("auto", "filename") else None
    rows_out: list[dict] = []
    skipped_missing = 0
    count = 0
    for row in rows:
        if args.limit and count >= int(args.limit):
            break
        ep_id = (row.get(episode_id_col) or "").strip() or f"row_{count}"
        source = (row.get(source_col) or "").strip() if source_col else ""

        w_raw = (row.get(whisperx_col) or "").strip()
        v_raw = (row.get(video_col) or "").strip()
        if not w_raw or not v_raw:
            skipped_missing += 1
            count += 1
            continue

        sibling_hints = [source] if source else []
        if args.path_mode == "relative":
            w_path = resolve_dataset_path(root, w_raw, sibling_hints=sibling_hints)
            v_path = resolve_dataset_path(root, v_raw, sibling_hints=sibling_hints)
        elif args.path_mode == "filename":
            w_path = resolve_dataset_path(root, Path(w_raw).name, file_index=path_index, sibling_hints=sibling_hints)
            v_path = resolve_dataset_path(root, Path(v_raw).name, file_index=path_index, sibling_hints=sibling_hints)
        else:
            w_path = resolve_dataset_path(root, w_raw, file_index=path_index, sibling_hints=sibling_hints)
            v_path = resolve_dataset_path(root, v_raw, file_index=path_index, sibling_hints=sibling_hints)

        if w_path is None or v_path is None:
            skipped_missing += 1
            count += 1
            continue

        try:
            w_json = load_json(w_path)
        except Exception:
            skipped_missing += 1
            count += 1
            continue

        for w in _iter_whisperx_words(w_json):
            word = str(w.get("word") or "").strip()
            t0 = _to_ms(w.get("start"))
            t1 = _to_ms(w.get("end"))
            if t0 is None or t1 is None:
                continue
            if t1 - t0 < int(args.min_word_ms):
                continue

            label = _match_label(word, rules)
            if not label:
                continue

            start_ms = max(0, int(t0) - int(args.pre_ms))
            end_ms = int(t1) + int(args.post_ms)
            if end_ms - start_ms > int(args.max_seg_ms):
                end_ms = start_ms + int(args.max_seg_ms)
            rows_out.append(
                {
                    "dataset": "iLSU-T",
                    "source": source,
                    "episode_id": ep_id,
                    "video_path": str(v_path),
                    "whisperx_path": str(w_path),
                    "label": label,
                    "matched_word": word,
                    "matched_word_norm": normalize_token(word),
                    "word_start_ms": int(t0),
                    "word_end_ms": int(t1),
                    "start_ms": int(start_ms),
                    "end_ms": int(end_ms),
                    "group_id": source or ep_id,
                }
            )

        count += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_out:
        raise SystemExit(
            "No matches produced. Check that your --episodes-csv columns match your file, "
            "paths are correct relative to --root, and --keywords patterns match Spanish text."
        )

    # Stable header
    fieldnames = [
        "dataset",
        "source",
        "episode_id",
        "video_path",
        "whisperx_path",
        "label",
        "matched_word",
        "matched_word_norm",
        "word_start_ms",
        "word_end_ms",
        "start_ms",
        "end_ms",
        "group_id",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r0 in rows_out:
            w.writerow(r0)

    print(f"Wrote: {out_path} ({len(rows_out)} segments, skipped_rows={skipped_missing})")


if __name__ == "__main__":
    main()
