from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare an SLT subset directly from WhisperX segments (clip -> target_text), "
            "without keyword-based label reduction. This is intended for generative SLT training."
        )
    )
    p.add_argument("--root", required=True, help="Extracted iLSU-T root, e.g. data/ilsut_extracted")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--episodes-csv", default="auto", help="episodes.csv-like file (video_path + whisperx_path). If auto, it is generated.")
    p.add_argument("--sources", nargs="*", default=None)
    p.add_argument("--video-dir-name", default="episodes")
    p.add_argument("--whisperx-dir-name", default="whisperx")

    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--min-words", type=int, default=1)
    p.add_argument("--max-words", type=int, default=60, help="0 disables max length filtering")
    p.add_argument("--max-chars", type=int, default=240, help="0 disables")
    p.add_argument("--min-duration-ms", type=int, default=700)
    p.add_argument("--max-duration-ms", type=int, default=25000, help="0 disables")
    p.add_argument("--keep-punctuation", action="store_true", help="Do not strip basic punctuation from target_text")
    p.add_argument("--max-segments-per-episode", type=int, default=0, help="0 disables")

    p.add_argument("--export-clips", action="store_true")
    p.add_argument("--clip-ext", choices=[".mp4", ".mkv"], default=".mp4")
    p.add_argument("--max-clips", type=int, default=0)
    return p.parse_args()


def _run(cmd: list[str | Path]) -> None:
    cmd_str = [str(x) for x in cmd]
    print("+", " ".join(cmd_str))
    rc = subprocess.call(cmd_str)
    if rc != 0:
        raise SystemExit(rc)


def _to_ms(x: object) -> int | None:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        return int(round(v * 1000.0))
    except Exception:
        return None


_PUNCT_RE = re.compile(r"[\r\n\t]+")
_TRIM_PUNCT_RE = re.compile(r"^[\s\-–—,.;:!?¡¿()\[\]{}\"“”'’]+|[\s\-–—,.;:!?¡¿()\[\]{}\"“”'’]+$")


def _normalize_text(text: str, *, keep_punctuation: bool) -> str:
    t = _PUNCT_RE.sub(" ", str(text or ""))
    t = " ".join(t.strip().split())
    if not keep_punctuation:
        t = _TRIM_PUNCT_RE.sub("", t).strip()
    return t


def _assign_splits(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> pd.Series:
    groups = df["group_id"].astype(str).to_numpy()
    idx = df.index.to_numpy()
    y = df["label"].astype(str).to_numpy()
    unique_groups = pd.unique(groups)
    if len(unique_groups) <= 1:
        return pd.Series(index=df.index, data="train", dtype="object")

    splitter_test = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
    train_val_idx, test_idx = next(splitter_test.split(idx, y, groups=groups))
    train_val_abs = idx[train_val_idx]
    test_abs = idx[test_idx]

    train_val_df = df.loc[train_val_abs]
    train_val_groups = train_val_df["group_id"].astype(str).to_numpy()
    train_val_y = train_val_df["label"].astype(str).to_numpy()
    unique_train_val_groups = pd.unique(train_val_groups)

    val_share = float(val_size) / max(1e-9, 1.0 - float(test_size))
    val_share = min(max(val_share, 0.05), 0.5)
    if len(unique_train_val_groups) <= 1:
        train_abs = train_val_abs
        val_abs = train_val_abs[:0]
    else:
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_share, random_state=int(seed))
        train_rel, val_rel = next(splitter_val.split(train_val_abs, train_val_y, groups=train_val_groups))
        train_abs = train_val_abs[train_rel]
        val_abs = train_val_abs[val_rel]

    split = pd.Series(index=df.index, data="train", dtype="object")
    split.loc[test_abs] = "test"
    split.loc[val_abs] = "val"
    split.loc[train_abs] = "train"
    return split


def _write_clip(src: Path, dst: Path, start_ms: int, end_ms: int) -> bool:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        return False
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if dst.suffix.lower() == ".mp4" else "XVID"))
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, int(start_ms)))
    wrote = False
    while True:
        pos_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) or 0)
        if pos_ms > int(end_ms):
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        wrote = True
    writer.release()
    cap.release()
    if not wrote and dst.exists():
        dst.unlink()
    return wrote


@dataclass
class SegmentRow:
    sample_id: str
    split: str
    source: str
    episode_id: str
    group_id: str
    video_path: str
    whisperx_path: str
    start_ms: int
    end_ms: int
    label: str
    target_text: str
    clip_path: str


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    py = sys.executable or "python"

    root = Path(args.root)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    episodes_csv = args.episodes_csv
    if not episodes_csv or str(episodes_csv).strip().lower() == "auto":
        episodes_csv = work_dir / "episodes_generated.csv"
        cmd = [
            py,
            repo / "scripts" / "build_ilsut_episodes_csv.py",
            "--root",
            root,
            "--out",
            episodes_csv,
            "--video-dir-name",
            args.video_dir_name,
            "--whisperx-dir-name",
            args.whisperx_dir_name,
        ]
        if args.sources:
            cmd += ["--sources", *args.sources]
        _run(cmd)

    df_ep = pd.read_csv(Path(episodes_csv))
    df_ep = df_ep.dropna(subset=["video_path", "whisperx_path"]).copy()
    df_ep["video_abs"] = df_ep["video_path"].astype(str).map(lambda p: str((root / p).resolve()))
    df_ep["whisper_abs"] = df_ep["whisperx_path"].astype(str).map(lambda p: str((root / p).resolve()))

    rows: list[dict] = []
    for e in df_ep.to_dict(orient="records"):
        video_abs = Path(str(e["video_abs"]))
        whisper_abs = Path(str(e["whisper_abs"]))
        if not video_abs.exists() or not whisper_abs.exists():
            continue
        episode_id = str(e.get("episode_id") or "")
        source = str(e.get("source") or "")
        group_id = episode_id or source

        obj = json.loads(whisper_abs.read_text(encoding="utf-8"))
        segs = obj.get("segments") or []
        kept = 0
        for seg_idx, seg in enumerate(segs):
            if not isinstance(seg, dict):
                continue
            t0 = _to_ms(seg.get("start"))
            t1 = _to_ms(seg.get("end"))
            if t0 is None or t1 is None or t1 <= t0:
                continue
            dur = int(t1 - t0)
            if dur < int(args.min_duration_ms):
                continue
            if int(args.max_duration_ms) > 0 and dur > int(args.max_duration_ms):
                continue

            text = _normalize_text(str(seg.get("text") or ""), keep_punctuation=bool(args.keep_punctuation))
            if not text:
                continue
            n_words = len(text.split())
            if n_words < int(args.min_words):
                continue
            if int(args.max_words) > 0 and n_words > int(args.max_words):
                continue
            if int(args.max_chars) > 0 and len(text) > int(args.max_chars):
                continue

            sample_id = f"{source}_{Path(video_abs).stem}_{seg_idx:04d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "episode_id": episode_id,
                    "group_id": group_id,
                    "video_path": str(video_abs),
                    "whisperx_path": str(whisper_abs),
                    "start_ms": int(t0),
                    "end_ms": int(t1),
                    "label": "slt",
                    "target_text": text,
                }
            )
            kept += 1
            if int(args.max_segments_per_episode) > 0 and kept >= int(args.max_segments_per_episode):
                break

    if not rows:
        raise SystemExit("No segments kept. Check WhisperX JSON format and filtering thresholds.")

    df = pd.DataFrame(rows)
    df["split"] = _assign_splits(df, float(args.test_size), float(args.val_size), int(args.seed))

    subset_manifest = work_dir / "subset_manifest.csv"
    df.to_csv(subset_manifest, index=False)
    for split_name in ("train", "val", "test"):
        df[df["split"] == split_name].to_csv(work_dir / f"{split_name}.csv", index=False)

    if args.export_clips:
        clips_dir = work_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        out_rows: list[SegmentRow] = []
        exported = 0
        for row in df.to_dict(orient="records"):
            if int(args.max_clips) > 0 and exported >= int(args.max_clips):
                break
            split = str(row["split"])
            src = Path(str(row["video_path"]))
            sid = str(row["sample_id"])
            out_path = clips_dir / split / f"{sid}{args.clip_ext}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not _write_clip(src, out_path, int(row["start_ms"]), int(row["end_ms"])):
                continue
            row["clip_path"] = str(out_path)
            out_rows.append(SegmentRow(**{k: row.get(k, "") for k in SegmentRow.__annotations__.keys()}))  # type: ignore[arg-type]
            exported += 1
        pd.DataFrame([r.__dict__ for r in out_rows]).to_csv(work_dir / "clips_index.csv", index=False)
        print(f"Wrote: {work_dir / 'clips_index.csv'}")

    info = {
        "rows": int(len(df)),
        "splits": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "episodes": int(df["group_id"].astype(str).nunique()),
        "config": {
            "root": str(root),
            "episodes_csv": str(episodes_csv),
            "sources": list(args.sources or []),
            "min_words": int(args.min_words),
            "max_words": int(args.max_words),
            "max_chars": int(args.max_chars),
            "min_duration_ms": int(args.min_duration_ms),
            "max_duration_ms": int(args.max_duration_ms),
            "max_segments_per_episode": int(args.max_segments_per_episode),
            "export_clips": bool(args.export_clips),
            "clip_ext": str(args.clip_ext),
            "max_clips": int(args.max_clips),
        },
    }
    (work_dir / "subset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote: {subset_manifest}")
    print(f"Wrote: {work_dir / 'subset_info.json'}")


if __name__ == "__main__":
    main()
