from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _python() -> str:
    return sys.executable or "python"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _run(args: list[str]) -> int:
    return subprocess.call(args)


def main() -> None:
    p = argparse.ArgumentParser(prog="vcpria")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("web", help="Run FastAPI web demo")
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--sequence-model", default="")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", default="8000")
    sp.add_argument("--open-browser", action="store_true")

    sp = sub.add_parser("demo", help="Run OpenCV desktop demo")
    sp.add_argument("--pipeline", choices=["landmarks", "cnn", "sequence"], default="landmarks")
    sp.add_argument("--model", required=True)
    sp.add_argument("--camera", default="0")
    sp.add_argument("--width", default="1280")
    sp.add_argument("--height", default="720")
    sp.add_argument("--max-fps", default="0")

    sp = sub.add_parser("collect", help="Collect labeled webcam samples")
    sp.add_argument("--out", required=True)
    sp.add_argument("--labels", nargs="+", required=True)
    sp.add_argument("--subject-id", default="S1")
    sp.add_argument("--camera", default="0")

    sp = sub.add_parser("collect-seq", help="Collect short sequences for dynamic gestures")
    sp.add_argument("--out", required=True)
    sp.add_argument("--labels", nargs="+", required=True)
    sp.add_argument("--subject-id", default="S1")
    sp.add_argument("--camera", default="0")
    sp.add_argument("--duration", default="2.0")

    sp = sub.add_parser("train-landmarks", help="Train landmarks baseline model")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("train-cnn", help="Train CNN transfer-learning model")
    sp.add_argument("--img-dir", default="")
    sp.add_argument("--csv", default="")
    sp.add_argument("--image-col", default="auto")
    sp.add_argument("--out", required=True)
    sp.add_argument("--epochs", default="10")
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--unfreeze-backbone", action="store_true")
    sp.add_argument("--group-col", default="")

    sp = sub.add_parser("train-seq", help="Train temporal baseline from landmark sequences")
    sp.add_argument("--seq-dir", required=True)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("train-stack", help="Run the recommended local end-to-end training stack")
    sp.add_argument("--csv", default="")
    sp.add_argument("--csvs", nargs="*", default=None)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--cnn-image-col", default="img_raw_path")
    sp.add_argument("--group-col", default="subject_id")
    sp.add_argument("--cnn-epochs", default="10")
    sp.add_argument("--cnn-device", default="cpu")
    sp.add_argument("--skip-ablation", action="store_true")
    sp.add_argument("--ablation-seconds", default="10")
    sp.add_argument("--camera", default="0")

    sp = sub.add_parser("ilsut-train", help="Prepare iLSU-T weak labels and train project models")
    sp.add_argument("--episodes-csv", required=True)
    sp.add_argument("--root", required=True)
    sp.add_argument("--keywords", required=True)
    sp.add_argument("--work-dir", required=True)
    sp.add_argument("--pipelines", nargs="+", choices=["landmarks", "cnn"], default=["cnn"])
    sp.add_argument("--fps", default="5.0")
    sp.add_argument("--max-per-seg", default="40")
    sp.add_argument("--preprocess", action="store_true")
    sp.add_argument("--skin-mask", action="store_true")
    sp.add_argument("--camera-like", action="store_true")
    sp.add_argument("--group-col", default="group_id")
    sp.add_argument("--cnn-epochs", default="10")
    sp.add_argument("--cnn-device", default="cpu")
    sp.add_argument("--cnn-use-masked", action="store_true")
    sp.add_argument("--path-mode", choices=["auto", "relative", "filename"], default="auto")

    sp = sub.add_parser("ilsut-download", help="Download iLSU-T archives from the local manifest")
    sp.add_argument("--manifest", default="deliverables/ilsut_downloads.json")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--sources", nargs="+", default=["source2", "source3"])
    sp.add_argument("--skip-existing", action="store_true")

    sp = sub.add_parser("ilsut-extract", help="Extract downloaded iLSU-T .rar archives")
    sp.add_argument("--archives-dir", required=True)
    sp.add_argument("--out-root", required=True)
    sp.add_argument("--sources", nargs="+", default=["source2", "source3"])
    sp.add_argument("--skip-existing", action="store_true")

    sp = sub.add_parser("eval-split", help="Evaluate with random/group split")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--cnn-image-col", default="auto")
    sp.add_argument("--group-col", default="")
    sp.add_argument("--json-out", default="")

    sp = sub.add_parser("ablation", help="Run ablation (split macro-F1 + FPS profiling)")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--group-col", default="")
    sp.add_argument("--seconds", default="10")
    sp.add_argument("--camera", default="0")
    sp.add_argument("--fps-preprocess", choices=["on", "off"], default="on")
    sp.add_argument("--fps-tracker", choices=["on", "off"], default="on")
    sp.add_argument("--fps-skin-mask", choices=["on", "off"], default="off")
    sp.add_argument("--fps-mask-space", choices=["ycrcb", "hsv"], default="ycrcb")
    sp.add_argument("--out-md", default="results/ablation_table.md")
    sp.add_argument("--out-json", default="results/ablation_results.json")

    sp = sub.add_parser("ablation-grid", help="Sweep OpenCV toggles (Macro-F1 + FPS)")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--landmarks-model", default="")
    sp.add_argument("--cnn-model", default="")
    sp.add_argument("--group-col", default="")
    sp.add_argument("--seconds", default="10")
    sp.add_argument("--camera", default="0")
    sp.add_argument("--out-md", default="results/ablation_grid.md")
    sp.add_argument("--out-json", default="results/ablation_grid.json")

    sp = sub.add_parser("dataset-stats", help="Print dataset stats and optional markdown")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--by", choices=["label", "label_subject"], default="label_subject")
    sp.add_argument("--out-md", default="")

    sp = sub.add_parser("validate-multisubject", help="Validate min samples per label per subject")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--min-per-label-per-subject", default="30")
    sp.add_argument("--out-md", default="")

    sp = sub.add_parser("merge-csvs", help="Merge multiple landmarks.csv into one")
    sp.add_argument("--out", required=True)
    sp.add_argument("--inputs", nargs="+", required=True)

    sp = sub.add_parser("report", help="Generate quick report (md + plot)")
    sp.add_argument("--ablation-json", default="results/ablation_results.json")
    sp.add_argument("--ablation-table", default="results/ablation_table.md")
    sp.add_argument("--dataset-stats-md", default="results/dataset_stats.md")
    sp.add_argument("--out-md", default="results/report.md")
    sp.add_argument("--out-fig", default="results/precision_vs_fps.png")

    sp = sub.add_parser("deliverables", help="Build PDFs + zip and run checks")
    sp.add_argument("--config", default="deliverables/config.json")
    sp.add_argument("--include-data", action="store_true")
    sp.add_argument("--include-models", action="store_true")
    sp.add_argument("--set-title", default="")
    sp.add_argument("--set-course", default="")
    sp.add_argument("--set-group", default="")
    sp.add_argument("--set-date", default="")
    sp.add_argument("--set-pitch-minutes", default="0")
    sp.add_argument("--set-members", nargs="*", default=None)

    args = p.parse_args()
    repo = _repo_root()
    py = _python()

    if args.cmd == "web":
        cmd = [
            py,
            str(repo / "scripts" / "run_webapp.py"),
            "--host",
            str(args.host),
            "--port",
            str(args.port),
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.sequence_model:
            cmd += ["--sequence-model", args.sequence_model]
        if args.open_browser:
            cmd += ["--open-browser"]
        raise SystemExit(_run(cmd))

    if args.cmd == "demo":
        cmd = [
            py,
            str(repo / "main.py"),
            "--pipeline",
            args.pipeline,
            "--model",
            args.model,
            "--camera",
            int(args.camera),
            "--width",
            int(args.width),
            "--height",
            int(args.height),
            "--max-fps",
            float(args.max_fps),
        ]
        cmd = [str(x) for x in cmd]
        raise SystemExit(_run(cmd))

    if args.cmd == "collect":
        cmd = [
            py,
            str(repo / "scripts" / "collect_data.py"),
            "--out",
            args.out,
            "--subject-id",
            args.subject_id,
            "--camera",
            args.camera,
            "--labels",
            *args.labels,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "collect-seq":
        cmd = [
            py,
            str(repo / "scripts" / "collect_sequence.py"),
            "--out",
            args.out,
            "--subject-id",
            args.subject_id,
            "--camera",
            int(args.camera),
            "--duration",
            float(args.duration),
            "--labels",
            *args.labels,
        ]
        cmd = [str(x) for x in cmd]
        raise SystemExit(_run(cmd))

    if args.cmd == "train-landmarks":
        cmd = [py, str(repo / "scripts" / "train_landmarks.py"), "--csv", args.csv, "--out", args.out]
        raise SystemExit(_run(cmd))

    if args.cmd == "train-cnn":
        cmd = [
            py,
            str(repo / "scripts" / "train_cnn.py"),
            "--out",
            args.out,
            "--epochs",
            args.epochs,
            "--device",
            args.device,
        ]
        if args.img_dir:
            cmd += ["--img-dir", args.img_dir]
        if args.csv:
            cmd += ["--csv", args.csv, "--image-col", args.image_col]
        if args.group_col:
            cmd += ["--group-col", args.group_col]
        if args.unfreeze_backbone:
            cmd += ["--unfreeze-backbone"]
        raise SystemExit(_run(cmd))

    if args.cmd == "train-seq":
        cmd = [
            py,
            str(repo / "scripts" / "train_sequence.py"),
            "--seq-dir",
            args.seq_dir,
            "--out",
            args.out,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "train-stack":
        cmd = [
            py,
            str(repo / "scripts" / "train_stack.py"),
            "--work-dir",
            args.work_dir,
            "--cnn-image-col",
            args.cnn_image_col,
            "--group-col",
            args.group_col,
            "--cnn-epochs",
            args.cnn_epochs,
            "--cnn-device",
            args.cnn_device,
            "--ablation-seconds",
            args.ablation_seconds,
            "--camera",
            args.camera,
        ]
        if args.csv:
            cmd += ["--csv", args.csv]
        if args.csvs:
            cmd += ["--csvs", *args.csvs]
        if args.skip_ablation:
            cmd.append("--skip-ablation")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-train":
        cmd = [
            py,
            str(repo / "scripts" / "train_ilsut.py"),
            "--episodes-csv",
            args.episodes_csv,
            "--root",
            args.root,
            "--keywords",
            args.keywords,
            "--work-dir",
            args.work_dir,
            "--fps",
            args.fps,
            "--max-per-seg",
            args.max_per_seg,
            "--group-col",
            args.group_col,
            "--cnn-epochs",
            args.cnn_epochs,
            "--cnn-device",
            args.cnn_device,
            "--path-mode",
            args.path_mode,
            "--pipelines",
            *args.pipelines,
        ]
        if args.preprocess:
            cmd.append("--preprocess")
        if args.skin_mask:
            cmd.append("--skin-mask")
        if args.camera_like:
            cmd.append("--camera-like")
        if args.cnn_use_masked:
            cmd.append("--cnn-use-masked")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-download":
        cmd = [
            py,
            str(repo / "scripts" / "download_ilsut.py"),
            "--manifest",
            args.manifest,
            "--out-dir",
            args.out_dir,
            "--sources",
            *args.sources,
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        raise SystemExit(_run(cmd))

    if args.cmd == "ilsut-extract":
        cmd = [
            py,
            str(repo / "scripts" / "extract_ilsut.py"),
            "--archives-dir",
            args.archives_dir,
            "--out-root",
            args.out_root,
            "--sources",
            *args.sources,
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        raise SystemExit(_run(cmd))

    if args.cmd == "eval-split":
        cmd = [
            py,
            str(repo / "scripts" / "eval_split.py"),
            "--csv",
            args.csv,
            "--cnn-image-col",
            args.cnn_image_col,
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.group_col:
            cmd += ["--group-col", args.group_col]
        if args.json_out:
            cmd += ["--json-out", args.json_out]
        raise SystemExit(_run(cmd))

    if args.cmd == "ablation":
        cmd = [
            py,
            str(repo / "scripts" / "run_ablation.py"),
            "--csv",
            args.csv,
            "--seconds",
            args.seconds,
            "--camera",
            args.camera,
            "--fps-preprocess",
            args.fps_preprocess,
            "--fps-tracker",
            args.fps_tracker,
            "--fps-skin-mask",
            args.fps_skin_mask,
            "--fps-mask-space",
            args.fps_mask_space,
            "--out-md",
            args.out_md,
            "--out-json",
            args.out_json,
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.group_col:
            cmd += ["--group-col", args.group_col]
        raise SystemExit(_run(cmd))

    if args.cmd == "ablation-grid":
        cmd = [
            py,
            str(repo / "scripts" / "run_ablation_grid.py"),
            "--csv",
            args.csv,
            "--seconds",
            args.seconds,
            "--camera",
            args.camera,
            "--out-md",
            args.out_md,
            "--out-json",
            args.out_json,
        ]
        if args.landmarks_model:
            cmd += ["--landmarks-model", args.landmarks_model]
        if args.cnn_model:
            cmd += ["--cnn-model", args.cnn_model]
        if args.group_col:
            cmd += ["--group-col", args.group_col]
        raise SystemExit(_run(cmd))

    if args.cmd == "dataset-stats":
        cmd = [py, str(repo / "scripts" / "dataset_stats.py"), "--csv", args.csv, "--by", args.by]
        if args.out_md:
            cmd += ["--out-md", args.out_md]
        raise SystemExit(_run(cmd))

    if args.cmd == "validate-multisubject":
        cmd = [
            py,
            str(repo / "scripts" / "validate_multisubject.py"),
            "--csv",
            args.csv,
            "--min-per-label-per-subject",
            args.min_per_label_per_subject,
        ]
        if args.out_md:
            cmd += ["--out-md", args.out_md]
        raise SystemExit(_run(cmd))

    if args.cmd == "merge-csvs":
        cmd = [py, str(repo / "scripts" / "merge_subject_csvs.py"), "--out", args.out, "--inputs", *args.inputs]
        raise SystemExit(_run(cmd))

    if args.cmd == "report":
        cmd = [
            py,
            str(repo / "scripts" / "make_report.py"),
            "--ablation-json",
            args.ablation_json,
            "--ablation-table",
            args.ablation_table,
            "--dataset-stats-md",
            args.dataset_stats_md,
            "--out-md",
            args.out_md,
            "--out-fig",
            args.out_fig,
        ]
        raise SystemExit(_run(cmd))

    if args.cmd == "deliverables":
        # Optionally update config before building.
        if (
            args.set_title
            or args.set_course
            or args.set_group
            or args.set_date
            or (args.set_members is not None and len(args.set_members) > 0)
            or int(args.set_pitch_minutes) > 0
        ):
            cmd0 = [
                py,
                str(repo / "scripts" / "update_deliverables_config.py"),
                "--config",
                args.config,
            ]
            if args.set_title:
                cmd0 += ["--title", args.set_title]
            if args.set_course:
                cmd0 += ["--course", args.set_course]
            if args.set_group:
                cmd0 += ["--group", args.set_group]
            if args.set_date:
                cmd0 += ["--date", args.set_date]
            if int(args.set_pitch_minutes) > 0:
                cmd0 += ["--pitch-minutes", args.set_pitch_minutes]
            if args.set_members is not None and len(args.set_members) > 0:
                cmd0 += ["--members", *args.set_members]
            rc0 = _run(cmd0)
            if rc0 != 0:
                raise SystemExit(rc0)

        cmd1 = [py, str(repo / "scripts" / "build_deliverables.py"), "--config", args.config]
        rc1 = _run(cmd1)
        if rc1 != 0:
            raise SystemExit(rc1)
        cmd2 = [py, str(repo / "scripts" / "package_submission.py")]
        if args.include_data:
            cmd2.append("--include-data")
        if args.include_models:
            cmd2.append("--include-models")
        rc2 = _run(cmd2)
        if rc2 != 0:
            raise SystemExit(rc2)
        cmd3 = [py, str(repo / "scripts" / "check_deliverables.py"), "--config", args.config]
        raise SystemExit(_run(cmd3))


if __name__ == "__main__":
    main()
