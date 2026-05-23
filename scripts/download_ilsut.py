from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download iLSU-T archives from a local manifest JSON.")
    p.add_argument("--manifest", default="deliverables/ilsut_downloads.json")
    p.add_argument("--out-dir", required=True, help="Destination folder for downloaded .rar files")
    p.add_argument("--sources", nargs="+", default=["source2", "source3"], help="Sources to download from the manifest")
    p.add_argument("--skip-existing", action="store_true", help="Skip files that already exist and are non-empty")
    p.add_argument("--tool", choices=["auto", "curl"], default="auto")
    return p.parse_args()


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _download_with_curl(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--continue-at",
        "-",
        "-o",
        str(out_path),
        url,
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"Download failed for {url}")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("curl") is None:
        raise SystemExit("This downloader requires `curl` to be available in PATH.")

    data = _load_manifest(manifest_path)
    selected = []
    for source_key in args.sources:
        if source_key not in data:
            raise SystemExit(f"Unknown source in manifest: {source_key}")
        selected.append((source_key, data[source_key]))

    total = 0
    for source_key, source_cfg in selected:
        source_dir = out_dir / source_key
        source_dir.mkdir(parents=True, exist_ok=True)
        for item in source_cfg.get("files", []):
            filename = str(item["filename"])
            url = str(item["url"])
            out_path = source_dir / filename
            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                print(f"skip: {out_path}")
                continue
            print(f"download: {filename}")
            _download_with_curl(url, out_path)
            total += 1

    print(f"Done. downloaded_files={total} out_dir={out_dir}")


if __name__ == "__main__":
    main()

