from __future__ import annotations

import os
import urllib.request
from pathlib import Path


def has_legacy_solutions() -> bool:
    try:
        import mediapipe as mp  # noqa: F401

        return hasattr(mp, "solutions")
    except Exception:
        return False


_MODEL_URLS = {
    # Official model asset URLs used by MediaPipe Tasks examples.
    "holistic_landmarker.task": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def _default_cache_dir() -> Path:
    # Allow overriding to avoid writing into $HOME in some environments.
    env = os.environ.get("LSUPRIA_MEDIAPIPE_MODEL_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".cache" / "lsupria" / "mediapipe_models").resolve()


def ensure_mediapipe_task_model(filename: str) -> Path:
    """
    Ensure the requested MediaPipe Tasks model exists on disk and return its path.

    Models are downloaded on demand into a cache directory.
    """
    if filename not in _MODEL_URLS:
        raise ValueError(f"Unknown MediaPipe model '{filename}'. Known: {sorted(_MODEL_URLS)}")

    cache_dir = _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = _MODEL_URLS[filename]
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted public model host)
        tmp.replace(out_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    return out_path

