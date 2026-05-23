from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation-json", type=str, required=True, help="results/ablation_results.json")
    p.add_argument("--out", type=str, required=True, help="Output PNG path")
    p.add_argument("--title", type=str, default="Precision vs FPS (Ablation)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.ablation_json).read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not cases:
        raise SystemExit("No cases found in ablation json")

    xs = []
    ys = []
    labels = []
    for c in cases:
        fps = c.get("fps")
        f1 = c.get("macro_f1")
        name = c.get("name", "case")
        if fps is None or f1 is None:
            continue
        xs.append(float(fps))
        ys.append(float(f1))
        labels.append(str(name))

    if not xs:
        raise SystemExit("No numeric fps/macro_f1 in cases")

    plt.figure(figsize=(7.2, 4.6), dpi=160)
    plt.scatter(xs, ys, s=60)
    for x, y, t in zip(xs, ys, labels):
        plt.annotate(t, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

    plt.title(args.title)
    plt.xlabel("FPS (higher is better)")
    plt.ylabel("Macro-F1 (higher is better)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

