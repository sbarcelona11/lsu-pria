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
    resolve_video_path_with_alternates,
)

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


@dataclass
class CompiledPattern:
    regex: re.Pattern
    token_len: int


@dataclass
class FuzzyPattern:
    text: str
    token_len: int
    score_cutoff: float


@dataclass
class KeywordRule:
    label: str
    word_patterns: list[CompiledPattern]
    phrase_patterns: list[CompiledPattern]
    exclude_word_patterns: list[CompiledPattern]
    exclude_phrase_patterns: list[CompiledPattern]
    fuzzy_word_patterns: list[FuzzyPattern]
    fuzzy_phrase_patterns: list[FuzzyPattern]


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
    p.add_argument(
        "--keywords",
        required=True,
        help=(
            "JSON file with keyword rules. Supports the legacy format {label:[patterns...]} "
            "and an extended object format with include/phrases/exclude."
        ),
    )
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


def _compile_pattern(raw: object) -> Optional[CompiledPattern]:
    s = str(raw).strip()
    if not s:
        return None
    token_len = max(1, len(normalize_token(s).split()))
    if any(ch in s for ch in [".", "?", "+", "*", "(", ")", "[", "]", "|", "^", "$", "\\"]):
        rx = re.compile(s, flags=re.IGNORECASE)
    else:
        s_norm = normalize_token(s)
        rx = re.compile(rf"\b{re.escape(s_norm)}\b", flags=re.IGNORECASE)
        token_len = max(1, len(s_norm.split()))
    return CompiledPattern(regex=rx, token_len=token_len)


def _compile_pattern_list(items: object) -> list[CompiledPattern]:
    if not isinstance(items, list):
        return []
    compiled: list[CompiledPattern] = []
    for item in items:
        pat = _compile_pattern(item)
        if pat is not None:
            compiled.append(pat)
    return compiled


def _compile_fuzzy_pattern_list(items: object, default_cutoff: float = 88.0) -> list[FuzzyPattern]:
    if not isinstance(items, list):
        return []
    compiled: list[FuzzyPattern] = []
    for item in items:
        if isinstance(item, dict):
            text = normalize_token(str(item.get("text") or "").strip())
            cutoff = float(item.get("score_cutoff", default_cutoff))
        else:
            text = normalize_token(str(item).strip())
            cutoff = float(default_cutoff)
        if not text:
            continue
        compiled.append(FuzzyPattern(text=text, token_len=max(1, len(text.split())), score_cutoff=cutoff))
    return compiled


def _coerce_rule_spec(spec: object) -> dict:
    if isinstance(spec, list):
        return {"include": spec}
    if isinstance(spec, dict):
        return spec
    return {}


def _load_keyword_rules(path: Path) -> list[KeywordRule]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(
            "keywords JSON must be an object. Supported formats: "
            "{label:[patterns...]} or "
            "{label:{include:[...], phrases:[...], exclude:[...], phrase_exclude:[...]}}"
        )
    rules: list[KeywordRule] = []
    for label, spec in raw.items():
        rule = _coerce_rule_spec(spec)
        word_patterns = _compile_pattern_list(rule.get("include", []))
        word_patterns += _compile_pattern_list(rule.get("variants", []))
        word_patterns += _compile_pattern_list(rule.get("regex", []))

        phrase_patterns = _compile_pattern_list(rule.get("phrases", []))
        phrase_patterns += _compile_pattern_list(rule.get("phrase_regex", []))

        exclude_word_patterns = _compile_pattern_list(rule.get("exclude", []))
        exclude_word_patterns += _compile_pattern_list(rule.get("exclude_regex", []))

        exclude_phrase_patterns = _compile_pattern_list(rule.get("phrase_exclude", []))
        exclude_phrase_patterns += _compile_pattern_list(rule.get("phrase_exclude_regex", []))
        fuzzy_word_patterns = _compile_fuzzy_pattern_list(rule.get("fuzzy", []), default_cutoff=88.0)
        fuzzy_phrase_patterns = _compile_fuzzy_pattern_list(rule.get("fuzzy_phrases", []), default_cutoff=85.0)

        if not word_patterns and not phrase_patterns and not fuzzy_word_patterns and not fuzzy_phrase_patterns:
            continue
        rules.append(
            KeywordRule(
                label=str(label),
                word_patterns=word_patterns,
                phrase_patterns=phrase_patterns,
                exclude_word_patterns=exclude_word_patterns,
                exclude_phrase_patterns=exclude_phrase_patterns,
                fuzzy_word_patterns=fuzzy_word_patterns,
                fuzzy_phrase_patterns=fuzzy_phrase_patterns,
            )
        )
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


def _matches_any(text: str, patterns: list[CompiledPattern]) -> bool:
    for pat in patterns:
        if pat.regex.search(text):
            return True
    return False


def _matches_any_fuzzy(text: str, patterns: list[FuzzyPattern]) -> bool:
    if fuzz is None:
        return False
    norm_text = normalize_token(text)
    for pat in patterns:
        if fuzz.ratio(norm_text, pat.text) >= pat.score_cutoff:
            return True
    return False


def _iter_segment_matches(words: list[dict], rules: list[KeywordRule]) -> Iterable[dict]:
    tokens: list[dict] = []
    for item in words:
        word = str(item.get("word") or "").strip()
        t0 = _to_ms(item.get("start"))
        t1 = _to_ms(item.get("end"))
        if t0 is None or t1 is None:
            continue
        norm = normalize_token(word)
        if not norm:
            continue
        tokens.append({"raw": word, "norm": norm, "start_ms": t0, "end_ms": t1})

    seen: set[tuple[str, int, int, str]] = set()
    for idx, tok in enumerate(tokens):
        norm_word = tok["norm"]
        for rule in rules:
            blocked_word_idx = False
            for pat in rule.exclude_phrase_patterns:
                end_idx = idx + pat.token_len
                if end_idx > len(tokens):
                    continue
                phrase_norm = " ".join(t["norm"] for t in tokens[idx:end_idx])
                if pat.regex.search(phrase_norm):
                    blocked_word_idx = True
                    break
            if not blocked_word_idx:
                for pat in rule.exclude_phrase_patterns:
                    start_idx = idx - pat.token_len + 1
                    if start_idx < 0:
                        continue
                    phrase_norm = " ".join(t["norm"] for t in tokens[start_idx : idx + 1])
                    if pat.regex.search(phrase_norm):
                        blocked_word_idx = True
                        break

            if blocked_word_idx:
                continue

            if rule.word_patterns and _matches_any(norm_word, rule.word_patterns):
                if not _matches_any(norm_word, rule.exclude_word_patterns):
                    key = (rule.label, tok["start_ms"], tok["end_ms"], norm_word)
                    if key not in seen:
                        seen.add(key)
                        yield {
                            "label": rule.label,
                            "matched_word": tok["raw"],
                            "matched_word_norm": norm_word,
                            "word_start_ms": tok["start_ms"],
                            "word_end_ms": tok["end_ms"],
                            "start_ms": tok["start_ms"],
                            "end_ms": tok["end_ms"],
                        }
            elif rule.fuzzy_word_patterns and _matches_any_fuzzy(norm_word, rule.fuzzy_word_patterns):
                if not _matches_any(norm_word, rule.exclude_word_patterns):
                    key = (rule.label, tok["start_ms"], tok["end_ms"], norm_word)
                    if key not in seen:
                        seen.add(key)
                        yield {
                            "label": rule.label,
                            "matched_word": tok["raw"],
                            "matched_word_norm": norm_word,
                            "word_start_ms": tok["start_ms"],
                            "word_end_ms": tok["end_ms"],
                            "start_ms": tok["start_ms"],
                            "end_ms": tok["end_ms"],
                        }

            for pat in rule.phrase_patterns:
                end_idx = idx + pat.token_len
                if end_idx > len(tokens):
                    continue
                phrase_tokens = tokens[idx:end_idx]
                phrase_norm = " ".join(t["norm"] for t in phrase_tokens)
                if not pat.regex.search(phrase_norm):
                    continue
                if _matches_any(phrase_norm, rule.exclude_phrase_patterns):
                    continue
                key = (rule.label, phrase_tokens[0]["start_ms"], phrase_tokens[-1]["end_ms"], phrase_norm)
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "label": rule.label,
                    "matched_word": " ".join(t["raw"] for t in phrase_tokens),
                    "matched_word_norm": phrase_norm,
                    "word_start_ms": phrase_tokens[0]["start_ms"],
                    "word_end_ms": phrase_tokens[-1]["end_ms"],
                    "start_ms": phrase_tokens[0]["start_ms"],
                    "end_ms": phrase_tokens[-1]["end_ms"],
                }

            for pat in rule.fuzzy_phrase_patterns:
                end_idx = idx + pat.token_len
                if end_idx > len(tokens):
                    continue
                phrase_tokens = tokens[idx:end_idx]
                phrase_norm = " ".join(t["norm"] for t in phrase_tokens)
                if fuzz is None or fuzz.ratio(phrase_norm, pat.text) < pat.score_cutoff:
                    continue
                if _matches_any(phrase_norm, rule.exclude_phrase_patterns):
                    continue
                key = (rule.label, phrase_tokens[0]["start_ms"], phrase_tokens[-1]["end_ms"], phrase_norm)
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "label": rule.label,
                    "matched_word": " ".join(t["raw"] for t in phrase_tokens),
                    "matched_word_norm": phrase_norm,
                    "word_start_ms": phrase_tokens[0]["start_ms"],
                    "word_end_ms": phrase_tokens[-1]["end_ms"],
                    "start_ms": phrase_tokens[0]["start_ms"],
                    "end_ms": phrase_tokens[-1]["end_ms"],
                }


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
        group_id = f"{source}:{ep_id}" if source else ep_id

        w_raw = (row.get(whisperx_col) or "").strip()
        v_raw = (row.get(video_col) or "").strip()
        if not w_raw or not v_raw:
            skipped_missing += 1
            count += 1
            continue

        sibling_hints = [source] if source else []
        if args.path_mode == "relative":
            w_path = resolve_dataset_path(root, w_raw, sibling_hints=sibling_hints)
            v_path = resolve_video_path_with_alternates(root, v_raw, sibling_hints=sibling_hints)
        elif args.path_mode == "filename":
            w_path = resolve_dataset_path(root, Path(w_raw).name, file_index=path_index, sibling_hints=sibling_hints)
            v_path = resolve_video_path_with_alternates(
                root, Path(v_raw).name, file_index=path_index, sibling_hints=sibling_hints
            )
        else:
            w_path = resolve_dataset_path(root, w_raw, file_index=path_index, sibling_hints=sibling_hints)
            v_path = resolve_video_path_with_alternates(
                root, v_raw, file_index=path_index, sibling_hints=sibling_hints
            )

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

        words = list(_iter_whisperx_words(w_json))
        for match in _iter_segment_matches(words, rules):
            t0 = int(match["word_start_ms"])
            t1 = int(match["word_end_ms"])
            if t1 - t0 < int(args.min_word_ms):
                continue

            start_ms = max(0, t0 - int(args.pre_ms))
            end_ms = t1 + int(args.post_ms)
            if end_ms - start_ms > int(args.max_seg_ms):
                end_ms = start_ms + int(args.max_seg_ms)
            rows_out.append(
                {
                    "dataset": "iLSU-T",
                    "source": source,
                    "episode_id": ep_id,
                    "video_path": str(v_path),
                    "whisperx_path": str(w_path),
                    "label": match["label"],
                    "matched_word": match["matched_word"],
                    "matched_word_norm": match["matched_word_norm"],
                    "word_start_ms": t0,
                    "word_end_ms": t1,
                    "start_ms": int(start_ms),
                    "end_ms": int(end_ms),
                    "group_id": group_id,
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
