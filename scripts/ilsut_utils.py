from __future__ import annotations

import csv
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


_EPISODE_ID_CANDIDATES = (
    "episode_id",
    "episode",
    "id",
    "sample_id",
    "name",
)
_SOURCE_CANDIDATES = (
    "source",
    "source_id",
    "split_source",
    "channel",
)
_VIDEO_PATH_CANDIDATES = (
    "video_path",
    "episode_path",
    "episode_file",
    "filepath",
    "file_path",
    "path",
    "video",
    "episode",
)
_WHISPERX_PATH_CANDIDATES = (
    "whisperx_path",
    "whisperx_json",
    "transcript_path",
    "transcription_path",
    "json_path",
    "asr_path",
)


@dataclass
class IlsutColumns:
    episode_id: str
    source: Optional[str]
    video_path: str
    whisperx_path: str


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip().replace("-", "_").replace(" ", "_")
    return s


def normalize_token(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()


def detect_columns(fieldnames: Iterable[str]) -> IlsutColumns:
    headers = list(fieldnames)
    by_norm = {_norm(h): h for h in headers}

    def pick(candidates: tuple[str, ...], required: bool = True) -> Optional[str]:
        for cand in candidates:
            if cand in by_norm:
                return by_norm[cand]
        # fallback by substring match
        for nk, original in by_norm.items():
            if any(cand in nk for cand in candidates):
                return original
        if required:
            raise SystemExit(
                "Could not detect expected column in episodes CSV. "
                f"Available columns: {headers}"
            )
        return None

    return IlsutColumns(
        episode_id=pick(_EPISODE_ID_CANDIDATES) or "episode_id",
        source=pick(_SOURCE_CANDIDATES, required=False),
        video_path=pick(_VIDEO_PATH_CANDIDATES) or "video_path",
        whisperx_path=pick(_WHISPERX_PATH_CANDIDATES) or "whisperx_path",
    )


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], IlsutColumns]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {path}")
        cols = detect_columns(reader.fieldnames)
    return rows, cols


def build_file_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for p in root.rglob("*"):
        if p.is_file():
            name = p.name
            if name in duplicates:
                continue
            if name in index:
                # Ambiguous basename: avoid silently resolving to an arbitrary file.
                duplicates.add(name)
                index.pop(name, None)
                continue
            index[name] = p
    return index


def resolve_dataset_path(
    root: Path,
    raw_value: str,
    file_index: Optional[dict[str, Path]] = None,
    sibling_hints: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    value = str(raw_value or "").strip()
    if not value:
        return None

    p = Path(value).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()

    rel = (root / value).resolve()
    if rel.exists():
        return rel

    # Try basename-only lookup in the extracted dataset tree.
    name = p.name
    if file_index is not None and name in file_index:
        return file_index[name].resolve()

    # Try sibling-hint names relative to root if the CSV path is stale.
    if sibling_hints is not None:
        for hint in sibling_hints:
            candidate = (root / hint / name).resolve()
            if candidate.exists():
                return candidate
    return None


def resolve_video_path_with_alternates(
    root: Path,
    raw_value: str,
    file_index: Optional[dict[str, Path]] = None,
    sibling_hints: Optional[Iterable[str]] = None,
    preferred_exts: tuple[str, ...] = (".mp4", ".mkv", ".avi"),
) -> Optional[Path]:
    primary = resolve_dataset_path(root, raw_value, file_index=file_index, sibling_hints=sibling_hints)
    if primary is not None:
        return primary

    raw = str(raw_value or "").strip()
    if not raw:
        return None

    p = Path(raw)
    stem = p.with_suffix("")
    original_ext = p.suffix.lower()

    candidates: list[str] = []
    for ext in preferred_exts:
        if ext != original_ext:
            candidates.append(str(stem) + ext)

    for alt in candidates:
        candidate = resolve_dataset_path(root, alt, file_index=file_index, sibling_hints=sibling_hints)
        if candidate is not None:
            return candidate

        # Also try basename-only substitution if the CSV path is stale.
        alt_name = Path(alt).name
        candidate = resolve_dataset_path(root, alt_name, file_index=file_index, sibling_hints=sibling_hints)
        if candidate is not None:
            return candidate

    return None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
