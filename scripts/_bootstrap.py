from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> None:
    """
    Allows running scripts as: `python scripts/foo.py` without setting PYTHONPATH.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    # Prefer src-root for importing vc_pria package.
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    # Keep repo-root too (for vcpria.py/webapp.py imports when running from repo).
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
