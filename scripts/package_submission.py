from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="deliverables/out/entrega_moodle.zip")
    p.add_argument("--include-data", action="store_true", help="Include data/ (can be large)")
    p.add_argument("--include-models", action="store_true", help="Include models/ (optional)")
    return p.parse_args()


def _add_dir(z: zipfile.ZipFile, root: Path, base: Path, ignore_ext: set[str] | None = None) -> None:
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        # Never include generated outputs or tmp artifacts inside deliverables/out
        p_posix = p.as_posix()
        if "deliverables/out/" in p_posix:
            continue
        if ignore_ext and p.suffix.lower() in ignore_ext:
            continue
        rel = p.relative_to(base).as_posix()
        z.write(p, arcname=rel)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ignore_ext = {".pyc", ".pyo", ".pyd"}
    ignore_dirs = {".git", ".venv", "node_modules", "dist", "build", "__pycache__"}

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # PDFs + PPTX (informe + pitch)
        pdf_dir = repo / "deliverables" / "out"
        if pdf_dir.exists():
            for p in pdf_dir.glob("*.pdf"):
                z.write(p, arcname=f"deliverables/out/{p.name}")
            for p in pdf_dir.glob("*.pptx"):
                z.write(p, arcname=f"deliverables/out/{p.name}")

        # Core code
        for p in ["README.md", "requirements.txt", "lsupria.py", "main.py", "webapp.py", "setup.py", "pyproject.toml"]:
            fp = repo / p
            if fp.exists():
                z.write(fp, arcname=p)

        _add_dir(z, repo / "src", base=repo, ignore_ext=ignore_ext)
        _add_dir(z, repo / "scripts", base=repo, ignore_ext=ignore_ext)
        # Include templates but skip deliverables/out (PDFs already added above).
        _add_dir(z, repo / "deliverables", base=repo, ignore_ext=ignore_ext)

        # Web UI source (not dist)
        if (repo / "web-ui").exists():
            for p in (repo / "web-ui").rglob("*"):
                if p.is_dir():
                    continue
                parts = p.relative_to(repo).parts
                if any(d in ignore_dirs for d in parts):
                    continue
                z.write(p, arcname=p.relative_to(repo).as_posix())

        if args.include_data and (repo / "data").exists():
            _add_dir(z, repo / "data", base=repo, ignore_ext=ignore_ext)
        if args.include_models and (repo / "models").exists():
            _add_dir(z, repo / "models", base=repo, ignore_ext=ignore_ext)

        # Optional results
        if (repo / "results").exists():
            _add_dir(z, repo / "results", base=repo, ignore_ext=ignore_ext)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
