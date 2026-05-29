from __future__ import annotations

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from lsu_pria.webapp.server import main


if __name__ == "__main__":
    main()
