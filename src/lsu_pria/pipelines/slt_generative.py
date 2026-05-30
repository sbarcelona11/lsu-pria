from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .slt import SltPrediction
from ..slt_features import (
    aggregate_sequence_embedding,
    extract_slt_features_from_frames,
    extract_slt_features_from_video,
    normalize_slt_text,
)


@dataclass
class SltGenerativePipeline:
    """
    Thin wrapper around the neccam/slt (SignJoey) backend.

    This pipeline runs inference by invoking `python -m signjoey test` in a given
    backend repo checkout. It is intended for demo/offline inference (uploaded
    video), not realtime.
    """

    backend_repo: Path
    base_config: Path
    ckpt: Path
    model_dir: Path
    sample_fps: float = 6.0
    max_frames: int = 0
    min_frames: int = 4
    preprocess: bool = True

    def _python(self) -> str:
        return os.environ.get("LSUPRIA_PY", "") or (os.sys.executable or "python")

    def _ensure_backend_present(self) -> None:
        if not self.backend_repo.exists():
            raise RuntimeError(f"Backend repo not found: {self.backend_repo}")
        if not self.base_config.exists():
            raise RuntimeError(f"Backend config not found: {self.base_config}")
        if not self.ckpt.exists():
            raise RuntimeError(f"Checkpoint not found: {self.ckpt}")
        if not (self.model_dir / "txt.vocab").exists() or not (self.model_dir / "gls.vocab").exists():
            raise RuntimeError(f"Missing vocabs under model_dir: {self.model_dir}")

    def _write_single_sample_dataset(self, out_dir: Path, *, sample_id: str, seq_features: np.ndarray) -> None:
        try:
            import torch
        except Exception as e:
            raise RuntimeError("Missing PyTorch; required for neccam/slt backend packaging.") from e

        out_dir.mkdir(parents=True, exist_ok=True)
        # SignJoey expects a gzip+pickle list of dicts with torch tensors.
        payload = [
            {
                "name": sample_id,
                "signer": "demo",
                "gloss": "",
                "text": "",
                "sign": torch.from_numpy(np.asarray(seq_features, dtype=np.float32).T).contiguous(),  # [F, T]
            }
        ]

        import gzip
        import pickle

        for split in ("train", "val", "test"):
            with gzip.open(out_dir / f"ilsut.{split}", "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _patch_config_for_single_sample(self, out_cfg: Path, *, data_path: Path, feature_size: int) -> None:
        """
        Patch a SignJoey YAML config by line-level replacements.
        We keep the model section intact (must match the checkpoint), and only
        override data paths + vocab files + beam sizes for faster inference.
        """

        txt = self.base_config.read_text(encoding="utf-8")

        def replace_data_scalar(key: str, value: str) -> None:
            nonlocal txt
            # Match "key: ..." within the data: block only.
            lines = txt.splitlines()
            out = []
            in_data = False
            replaced = False
            for ln in lines:
                if ln.startswith("data:"):
                    in_data = True
                    out.append(ln)
                    continue
                if in_data and ln and not ln.startswith(" "):
                    in_data = False
                if in_data and re.match(rf"\\s*{re.escape(key)}\\s*:\\s*", ln):
                    indent = ln[: len(ln) - len(ln.lstrip())]
                    out.append(f"{indent}{key}: {value}")
                    replaced = True
                else:
                    out.append(ln)
            if not replaced:
                out2 = []
                inserted = False
                for ln in out:
                    out2.append(ln)
                    if ln.startswith("data:") and not inserted:
                        out2.append(f"    {key}: {value}")
                        inserted = True
                out = out2
            txt = "\n".join(out)

        def replace_testing_lists() -> None:
            nonlocal txt
            # Brutal but effective: replace the entire `testing:` block if present.
            m = re.search(r"^testing:\\n(?:^[ ]+.*\\n?)*", txt, flags=re.MULTILINE)
            block = "\n".join(
                [
                    "testing:",
                    "    recognition_beam_sizes:",
                    "    - 1",
                    "    translation_beam_sizes:",
                    "    - 1",
                    "    translation_beam_alphas:",
                    "    - -1",
                    "",
                ]
            )
            if m:
                txt = txt[: m.start()] + block + txt[m.end() :]
            else:
                txt = txt.rstrip() + "\n\n" + block

        replace_data_scalar("data_path", json.dumps(str(data_path)))  # directory containing ilsut.{split}
        replace_data_scalar("train", "ilsut.train")
        replace_data_scalar("dev", "ilsut.val")
        replace_data_scalar("test", "ilsut.test")
        replace_data_scalar("feature_size", str(int(feature_size)))
        replace_data_scalar("level", "word")
        replace_data_scalar("txt_lowercase", "true")
        replace_data_scalar("max_sent_length", "400")
        replace_data_scalar("gls_vocab", json.dumps(str(self.model_dir / "gls.vocab")))
        replace_data_scalar("txt_vocab", json.dumps(str(self.model_dir / "txt.vocab")))
        replace_testing_lists()

        out_cfg.parent.mkdir(parents=True, exist_ok=True)
        out_cfg.write_text(txt.rstrip() + "\n", encoding="utf-8")

    def _run_backend_test(self, cfg_path: Path, out_prefix: Path) -> str:
        cmd = [
            self._python(),
            "-m",
            "signjoey",
            "test",
            str(cfg_path),
            "--ckpt",
            str(self.ckpt),
            "--output_path",
            str(out_prefix),
        ]
        proc = subprocess.run(cmd, cwd=str(self.backend_repo), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("neccam/slt inference failed:\n" + (proc.stderr or proc.stdout or ""))

        # Find the test translations file.
        matches = sorted(out_prefix.parent.glob(out_prefix.name + "*.test.txt"))
        if not matches:
            # SignJoey may also write without alpha in filename depending on config.
            matches = sorted(out_prefix.parent.glob(out_prefix.name + "*.test.txt"))
        if not matches:
            raise RuntimeError("Backend inference did not produce a test.txt output.")

        line = matches[0].read_text(encoding="utf-8").splitlines()[0].strip() if matches[0].exists() else ""
        if "|" in line:
            return line.split("|", 1)[1].strip()
        return line

    def predict_from_sequence(self, seq_features: np.ndarray) -> tuple[str, float, np.ndarray]:
        emb = aggregate_sequence_embedding(seq_features, min_frames=self.min_frames)
        if emb.size == 0:
            return "", 0.0, emb

        self._ensure_backend_present()
        feature_size = int(seq_features.shape[1]) if seq_features.ndim == 2 else 0
        if feature_size <= 0:
            return "", 0.0, emb

        with tempfile.TemporaryDirectory(prefix="lsupria_sltgen_") as tmp:
            tmp_dir = Path(tmp)
            data_dir = tmp_dir / "data" / "ilsut"
            self._write_single_sample_dataset(data_dir, sample_id="demo_000001", seq_features=seq_features)
            cfg_path = tmp_dir / "config.yaml"
            self._patch_config_for_single_sample(cfg_path, data_path=data_dir, feature_size=feature_size)
            out_prefix = tmp_dir / "out"
            pred_text = self._run_backend_test(cfg_path, out_prefix)

        pred_text = pred_text.strip()
        conf = 1.0 if normalize_slt_text(pred_text) else 0.0
        return pred_text, conf, emb

    def predict_video_file(
        self,
        video_path: Path,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        sample_fps: float | None = None,
        max_frames: int | None = None,
        preprocess: bool | None = None,
    ) -> SltPrediction:
        extraction = extract_slt_features_from_video(
            video_path,
            start_ms=start_ms,
            end_ms=end_ms,
            sample_fps=float(self.sample_fps if sample_fps is None else sample_fps),
            max_frames=int(self.max_frames if max_frames is None else max_frames),
            preprocess=bool(self.preprocess if preprocess is None else preprocess),
            include_debug=False,
        )
        text, conf, emb = self.predict_from_sequence(extraction.features)
        toks = normalize_slt_text(text).split() if text else []
        return SltPrediction(
            text=text,
            confidence=conf,
            token_sequence=toks,
            embedding=emb,
            frames_total=extraction.frames_total,
            frames_used=extraction.frames_used,
        )

    def predict_frames(
        self,
        frames_bgr: list[np.ndarray],
        ts_ms: list[int],
        *,
        sample_fps: float | None = None,
        max_frames: int | None = None,
        preprocess: bool | None = None,
    ) -> SltPrediction:
        extraction = extract_slt_features_from_frames(
            frames_bgr,
            ts_ms,
            sample_fps=float(self.sample_fps if sample_fps is None else sample_fps),
            max_frames=int(self.max_frames if max_frames is None else max_frames),
            preprocess=bool(self.preprocess if preprocess is None else preprocess),
            include_debug=False,
        )
        text, conf, emb = self.predict_from_sequence(extraction.features)
        toks = normalize_slt_text(text).split() if text else []
        return SltPrediction(
            text=text,
            confidence=conf,
            token_sequence=toks,
            embedding=emb,
            frames_total=extraction.frames_total,
            frames_used=extraction.frames_used,
        )
