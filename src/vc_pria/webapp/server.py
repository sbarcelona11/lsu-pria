from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import time
import uuid
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import sys

from ..hand import HandDetector
from ..multimodal import HolisticDetector, multimodal_bbox, multimodal_primary_hand_landmarks
from ..opencv_utils import SkinMaskConfig, apply_clahe, maybe_denoise, skin_mask_hsv, skin_mask_ycrcb
from ..pipelines.cnn import CnnPipeline
from ..pipelines.landmarks import LandmarksPipeline
from ..pipelines.multimodal_sequence import MultimodalSequencePipeline
from ..pipelines.sequence import SequencePipeline
from ..tracking import RoiTracker
from .composer import ComposeConfig, ComposeMode, ComposeState
from .tts import TtsEngine
from .video_analysis import VideoAnalysisConfig, analyze_video, download_video_source, load_video_pipeline, tools_status


INDEX_HTML = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>VC-pria Web Demo</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; background: #0b1020; color: #e8ecff;}
      .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start;}
      .card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 12px;}
      video, canvas { border-radius: 12px; background: #000; }
      .stack { display: grid; gap: 8px; }
      .stage { position: relative; width: 640px; height: 480px; }
      .stage video { position: absolute; left: 0; top: 0; width: 640px; height: 480px; }
      .stage canvas { position: absolute; left: 0; top: 0; width: 640px; height: 480px; pointer-events: none; }
      label { display: inline-flex; gap: 8px; align-items: center; }
      input[type="range"] { width: 220px; }
      button { padding: 8px 10px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.10); color: #e8ecff; cursor: pointer;}
      button:hover { background: rgba(255,255,255,0.16); }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
      textarea { width: 520px; height: 90px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.18); background: rgba(0,0,0,0.35); color: #e8ecff; padding: 10px;}
      .small { opacity: 0.9; font-size: 12px;}
      .pill { display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.16); background: rgba(0,0,0,0.25); }
      .muted { opacity: 0.85; }
      select { border-radius: 10px; padding: 6px 8px; border: 1px solid rgba(255,255,255,0.18); background: rgba(0,0,0,0.25); color: #e8ecff; }
    </style>
  </head>
  <body>
    <h2>VC-pria — Web demo (infer + frase + TTS)</h2>
    <div class="pill mono muted">session: <span id="sid">(creating...)</span> <button id="copySid">Copy</button> <button id="export">Export JSON</button> <button id="exportCsv">Export CSV</button></div>
    <div class="row">
      <div class="card stack">
        <div class="stage">
          <video id="vid" width="640" height="480" autoplay playsinline muted></video>
          <canvas id="overlay" width="640" height="480"></canvas>
          <canvas id="cv" width="640" height="480" style="display:none"></canvas>
        </div>
        <div class="small mono" id="status">starting...</div>
      </div>
      <div class="card stack">
        <div class="small mono muted" id="models">models: loading...</div>
        <div><b>Pipeline</b></div>
        <label><input id="pipe_landmarks" type="radio" name="pipeline" value="landmarks" checked/> landmarks</label>
        <label><input id="pipe_cnn" type="radio" name="pipeline" value="cnn"/> cnn</label>
        <label><input id="pipe_sequence" type="radio" name="pipeline" value="sequence"/> sequence</label>
        <label><input id="pipe_multimodal" type="radio" name="pipeline" value="multimodal"/> multimodal</label>
        <hr style="width:100%; border:none; border-top:1px solid rgba(255,255,255,0.12)"/>
        <div><b>Compose mode</b></div>
        <label><input type="radio" name="mode" value="both" checked/> both</label>
        <label><input type="radio" name="mode" value="words"/> words</label>
        <label><input type="radio" name="mode" value="spelling"/> spelling</label>
        <hr style="width:100%; border:none; border-top:1px solid rgba(255,255,255,0.12)"/>
        <label><input id="pre" type="checkbox" checked/> preprocess (CLAHE/denoise)</label>
        <label><input id="mask" type="checkbox"/> skin mask</label>
        <label>mask space
          <select id="maskSpace">
            <option value="ycrcb" selected>YCrCb</option>
            <option value="hsv">HSV</option>
          </select>
        </label>
        <label><input id="trk" type="checkbox" checked/> tracker (bbox fallback)</label>
        <label><input id="dbg" type="checkbox" checked/> draw landmarks/bbox</label>
        <label><input id="maskThumb" type="checkbox"/> mask thumbnail</label>
        <label><input id="roiThumb" type="checkbox"/> ROI thumbnail</label>
        <label><input id="composeDbg" type="checkbox"/> compose debug</label>
        <label>fps target <input id="fps" type="range" min="2" max="20" value="10"/></label>
        <label>conf thr <input id="thr" type="range" min="50" max="95" value="75"/></label>
        <label>stable frames <input id="stable" type="range" min="2" max="14" value="6"/></label>
        <label>pause ms <input id="pause" type="range" min="150" max="900" value="350"/></label>
        <label>cooldown ms <input id="cool" type="range" min="200" max="2000" value="800"/></label>
        <hr style="width:100%; border:none; border-top:1px solid rgba(255,255,255,0.12)"/>
        <div><b>Frase</b> <span class="small">(confirmación automática por pausa)</span></div>
        <textarea id="phrase" placeholder="tokens confirmados..."></textarea>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <button id="reset">Reset frase</button>
          <button id="space">Espacio</button>
          <button id="back">Borrar</button>
          <button id="speak">Hablar (TTS)</button>
        </div>
        <div class="small">Tip: hacé el gesto estable y sacá la mano (pausa) para confirmarlo.</div>
      </div>
    </div>
    <script>
      const vid = document.getElementById('vid');
      const overlay = document.getElementById('overlay');
      const octx = overlay.getContext('2d');
      const cv = document.getElementById('cv');
      const ctx = cv.getContext('2d');
      const statusEl = document.getElementById('status');
      const phraseEl = document.getElementById('phrase');
      const preEl = document.getElementById('pre');
      const maskEl = document.getElementById('mask');
      const maskSpaceEl = document.getElementById('maskSpace');
      const trkEl = document.getElementById('trk');
      const dbgEl = document.getElementById('dbg');
      const maskThumbEl = document.getElementById('maskThumb');
      const roiThumbEl = document.getElementById('roiThumb');
      const composeDbgEl = document.getElementById('composeDbg');
      const fpsEl = document.getElementById('fps');
      const thrEl = document.getElementById('thr');
      const stableEl = document.getElementById('stable');
      const pauseEl = document.getElementById('pause');
      const coolEl = document.getElementById('cool');
      const resetBtn = document.getElementById('reset');
      const spaceBtn = document.getElementById('space');
      const backBtn = document.getElementById('back');
      const speakBtn = document.getElementById('speak');
      const sidEl = document.getElementById('sid');
      const copySidBtn = document.getElementById('copySid');
      const exportBtn = document.getElementById('export');
      const exportCsvBtn = document.getElementById('exportCsv');
      const modelsEl = document.getElementById('models');
      const pipeLandmarksEl = document.getElementById('pipe_landmarks');
      const pipeCnnEl = document.getElementById('pipe_cnn');
      const pipeSequenceEl = document.getElementById('pipe_sequence');
      const pipeMultimodalEl = document.getElementById('pipe_multimodal');

      const getPipeline = () => document.querySelector('input[name="pipeline"]:checked').value;
      const getMode = () => document.querySelector('input[name="mode"]:checked').value;

      let sessionId = null;
      let running = true;

      // MediaPipe Hands connections (index pairs), for drawing.
      const HAND_CONNECTIONS = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [5,9],[9,10],[10,11],[11,12],
        [9,13],[13,14],[14,15],[15,16],
        [13,17],[17,18],[18,19],[19,20],
        [0,17]
      ];

      async function newSession() {
        const res = await fetch('/api/session/new', {method: 'POST'});
        const j = await res.json();
        sessionId = j.session_id;
        sidEl.textContent = sessionId;
      }

      async function startCam() {
        const stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false});
        vid.srcObject = stream;
        await vid.play();
      }

      async function loadModels() {
        try {
          const h = await (await fetch('/api/health')).json();
          const c = await (await fetch('/api/classes')).json();
          const hasL = !!(h.pipelines && h.pipelines.landmarks);
          const hasC = !!(h.pipelines && h.pipelines.cnn);
          const hasS = !!(h.pipelines && h.pipelines.sequence);
          const hasM = !!(h.pipelines && h.pipelines.multimodal);

          pipeLandmarksEl.disabled = !hasL;
          pipeCnnEl.disabled = !hasC;
          pipeSequenceEl.disabled = !hasS;
          pipeMultimodalEl.disabled = !hasM;
          if (!hasL && hasC) pipeCnnEl.checked = true;
          if (!hasL && !hasC && hasS) pipeSequenceEl.checked = true;
          if (!hasL && !hasC && !hasS && hasM) pipeMultimodalEl.checked = true;

          const lCount = (c.classes && c.classes.landmarks) ? c.classes.landmarks.length : 0;
          const cCount = (c.classes && c.classes.cnn) ? c.classes.cnn.length : 0;
          const sCount = (c.classes && c.classes.sequence) ? c.classes.sequence.length : 0;
          const mCount = (c.classes && c.classes.multimodal) ? c.classes.multimodal.length : 0;
          modelsEl.textContent = `models: landmarks=${hasL ? 'on' : 'off'}(${lCount}) cnn=${hasC ? 'on' : 'off'}(${cCount}) sequence=${hasS ? 'on' : 'off'}(${sCount}) multimodal=${hasM ? 'on' : 'off'}(${mCount})`;
        } catch (e) {
          modelsEl.textContent = 'models: error loading /health';
        }
      }

      function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

      function drawOverlay(payload) {
        octx.clearRect(0, 0, overlay.width, overlay.height);
        if (!dbgEl.checked) return;
        if (payload && payload.bbox) {
          const [x,y,w,h] = payload.bbox;
          octx.lineWidth = 2;
          octx.strokeStyle = payload.no_hand ? 'rgba(255,100,100,0.9)' : 'rgba(120,255,120,0.9)';
          octx.strokeRect(x, y, w, h);
        }
        if (payload && payload.landmarks_px) {
          const pts = payload.landmarks_px;
          // connections
          octx.lineWidth = 2;
          octx.strokeStyle = 'rgba(80,180,255,0.85)';
          for (const [a,b] of HAND_CONNECTIONS) {
            const pa = pts[a], pb = pts[b];
            if (!pa || !pb) continue;
            octx.beginPath();
            octx.moveTo(pa[0], pa[1]);
            octx.lineTo(pb[0], pb[1]);
            octx.stroke();
          }
          // points
          for (let i=0; i<pts.length; i++) {
            const p = pts[i];
            octx.fillStyle = (i===0) ? 'rgba(255,220,120,0.95)' : 'rgba(255,255,255,0.9)';
            octx.beginPath();
            octx.arc(p[0], p[1], 3, 0, Math.PI*2);
            octx.fill();
          }
        }

        // Debug thumbnails (mask/ROI) bottom-left.
        if (payload && payload.debug_mask_png_b64) {
          const img = new Image();
          img.onload = () => octx.drawImage(img, 8, overlay.height - img.height - 8);
          img.src = 'data:image/png;base64,' + payload.debug_mask_png_b64;
        }
        if (payload && payload.debug_roi_png_b64) {
          const img = new Image();
          img.onload = () => octx.drawImage(img, 140, overlay.height - img.height - 8);
          img.src = 'data:image/png;base64,' + payload.debug_roi_png_b64;
        }
      }

      async function loop() {
        await newSession();
        while (running) {
          const t0 = performance.now();
          ctx.drawImage(vid, 0, 0, cv.width, cv.height);
          const blob = await new Promise(res => cv.toBlob(res, 'image/jpeg', 0.85));

          const fd = new FormData();
          fd.append('image', blob, 'frame.jpg');
          fd.append('pipeline', getPipeline());
          fd.append('mode', getMode());
          fd.append('preprocess', preEl.checked ? '1' : '0');
          fd.append('skin_mask', maskEl.checked ? '1' : '0');
          fd.append('mask_space', maskSpaceEl.value || 'ycrcb');
          fd.append('use_tracker', trkEl.checked ? '1' : '0');
          fd.append('debug_mask_thumb', maskThumbEl.checked ? '1' : '0');
          fd.append('debug_roi_thumb', roiThumbEl.checked ? '1' : '0');
          fd.append('debug_compose', composeDbgEl.checked ? '1' : '0');
          fd.append('confidence_threshold', (parseInt(thrEl.value,10)/100).toString());
          fd.append('stable_frames_min', stableEl.value);
          fd.append('pause_ms_min', pauseEl.value);
          fd.append('cooldown_ms', coolEl.value);
          fd.append('ts_ms', Math.floor(Date.now()).toString());
          fd.append('session_id', sessionId);

          let j = null;
          try {
            const res = await fetch('/api/infer_frame', {method: 'POST', body: fd});
            j = await res.json();
          } catch (e) {
            statusEl.textContent = 'error: ' + e;
            await sleep(500);
            continue;
          }

          phraseEl.value = j.compose_text || '';
          const clientFps = 1000.0 / Math.max(1.0, (performance.now() - t0));
          let extra = '';
          if (j.compose_debug) {
            extra = ` stable=${j.compose_debug.stable_count} cand=${j.compose_debug.candidate_ready ? j.compose_debug.candidate_label : '-'}`;
          }
          statusEl.textContent = `pred=${j.label} conf=${j.confidence.toFixed(2)} no_hand=${j.no_hand} tracker=${j.tracker_status} client_fps=${clientFps.toFixed(1)} server_ms=${j.server_ms.toFixed(1)}${extra}`;
          drawOverlay(j);

          const fpsTarget = parseInt(fpsEl.value,10);
          const elapsed = performance.now() - t0;
          const wait = Math.max(0, (1000/fpsTarget) - elapsed);
          await sleep(wait);
        }
      }

      resetBtn.onclick = async () => {
        if (!sessionId) return;
        await fetch(`/api/session/${sessionId}/reset`, {method:'POST'});
        phraseEl.value = '';
      };
      spaceBtn.onclick = async () => {
        if (!sessionId) return;
        await fetch(`/api/session/${sessionId}/action`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:'space'})});
      };
      backBtn.onclick = async () => {
        if (!sessionId) return;
        await fetch(`/api/session/${sessionId}/action`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action:'backspace'})});
      };
      speakBtn.onclick = async () => {
        const text = phraseEl.value.trim();
        if (!text) return;
        await fetch('/api/tts', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
      };

      copySidBtn.onclick = async () => {
        if (!sessionId) return;
        try { await navigator.clipboard.writeText(sessionId); } catch(e) {}
      };
      exportBtn.onclick = async () => {
        if (!sessionId) return;
        const res = await fetch(`/api/session/${sessionId}/export`);
        const j = await res.json();
        const blob = new Blob([JSON.stringify(j, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `session_${sessionId}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      };
      exportCsvBtn.onclick = async () => {
        if (!sessionId) return;
        const res = await fetch(`/api/session/${sessionId}/export.csv`);
        const txt = await res.text();
        const blob = new Blob([txt], {type: 'text/csv'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `session_${sessionId}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      };

      startCam().then(loop).catch(e => statusEl.textContent = 'camera error: ' + e);
      loadModels();
    </script>
  </body>
</html>
"""


@dataclass
class LoadedPipelines:
    landmarks: Optional[LandmarksPipeline] = None
    cnn: Optional[CnnPipeline] = None
    sequence: Optional[SequencePipeline] = None
    multimodal: Optional[MultimodalSequencePipeline] = None

    def get(self, name: str) -> object:
        if name == "landmarks":
            if self.landmarks is None:
                raise HTTPException(status_code=400, detail="landmarks model not loaded")
            return self.landmarks
        if name == "cnn":
            if self.cnn is None:
                raise HTTPException(status_code=400, detail="cnn model not loaded")
            return self.cnn
        if name == "sequence":
            if self.sequence is None:
                raise HTTPException(status_code=400, detail="sequence model not loaded")
            return self.sequence
        if name == "multimodal":
            if self.multimodal is None:
                raise HTTPException(status_code=400, detail="multimodal model not loaded")
            return self.multimodal
        raise HTTPException(status_code=400, detail=f"unknown pipeline: {name}")


def create_app(pipelines: LoadedPipelines) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    detector = HandDetector(max_num_hands=1)
    holistic = HolisticDetector()
    skin_cfg = SkinMaskConfig()
    tts = TtsEngine()
    artifacts_dir = Path(__file__).resolve().parents[3] / "web-artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    @dataclass
    class SessionRuntime:
        composer: ComposeState
        tracker: RoiTracker
        last_bbox: Optional[tuple[int, int, int, int]] = None
        log: list[dict] = field(default_factory=list)

    sessions: dict[str, SessionRuntime] = {}

    # Legacy inline UI lives at /legacy (React build will be served at / when available).
    @app.get("/legacy", response_class=HTMLResponse)
    def legacy_index() -> str:
        return INDEX_HTML

    api = APIRouter(prefix="/api")

    @api.get("/health")
    def health() -> dict:
        l_count = len(pipelines.landmarks.labels) if pipelines.landmarks is not None else 0
        c_count = len(pipelines.cnn.labels) if pipelines.cnn is not None else 0
        s_count = len(pipelines.sequence.labels) if pipelines.sequence is not None else 0
        m_count = len(pipelines.multimodal.labels) if pipelines.multimodal is not None else 0
        return {
            "ok": True,
            "pipelines": {
                "landmarks": pipelines.landmarks is not None,
                "cnn": pipelines.cnn is not None,
                "sequence": pipelines.sequence is not None,
                "multimodal": pipelines.multimodal is not None,
            },
            "classes_count": {"landmarks": l_count, "cnn": c_count, "sequence": s_count, "multimodal": m_count},
            "tools": tools_status(),
        }

    @api.get("/classes")
    def classes() -> dict:
        out = {}
        if pipelines.landmarks is not None:
            out["landmarks"] = list(pipelines.landmarks.labels)
        if pipelines.cnn is not None:
            out["cnn"] = list(pipelines.cnn.labels)
        if pipelines.sequence is not None:
            out["sequence"] = list(pipelines.sequence.labels)
        if pipelines.multimodal is not None:
            out["multimodal"] = list(pipelines.multimodal.labels)
        return {"classes": out}

    @api.post("/session/new")
    def session_new() -> dict:
        sid = str(uuid.uuid4())
        sessions[sid] = SessionRuntime(composer=ComposeState(), tracker=RoiTracker(), last_bbox=None)
        return {"session_id": sid}

    @api.post("/session/{session_id}/reset")
    def session_reset(session_id: str) -> dict:
        rt = sessions.get(session_id)
        if rt is None:
            raise HTTPException(status_code=404, detail="session not found")
        rt.composer.reset()
        rt.tracker.reset()
        rt.last_bbox = None
        rt.log.clear()
        return {"ok": True}

    @api.get("/session/{session_id}/export")
    def session_export(session_id: str) -> dict:
        rt = sessions.get(session_id)
        if rt is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {
            "session_id": session_id,
            "compose": {
                "text": rt.composer.text,
                "tokens": list(rt.composer.tokens),
                "current_word": rt.composer.current_word,
                "mode": rt.composer.mode.name,
                "config": {
                    "confidence_threshold": rt.composer.config.confidence_threshold,
                    "stable_frames_min": rt.composer.config.stable_frames_min,
                    "pause_ms_min": rt.composer.config.pause_ms_min,
                    "cooldown_ms": rt.composer.config.cooldown_ms,
                },
            },
            "log": rt.log,
        }

    @api.get("/session/{session_id}/export.csv")
    def session_export_csv(session_id: str) -> Response:
        rt = sessions.get(session_id)
        if rt is None:
            raise HTTPException(status_code=404, detail="session not found")
        rows = rt.log
        if not rows:
            return Response(content="", media_type="text/csv")
        # Stable header
        fieldnames = sorted({k for r in rows for k in r.keys()})
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        content = buf.getvalue()
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="session_{session_id}.csv"'},
        )

    @api.post("/session/{session_id}/action")
    def session_action(session_id: str, payload: dict) -> dict:
        rt = sessions.get(session_id)
        if rt is None:
            raise HTTPException(status_code=404, detail="session not found")
        action = str(payload.get("action", ""))
        if action == "space":
            rt.composer.add_space()
        elif action == "backspace":
            rt.composer.backspace()
        elif action == "reset":
            rt.composer.reset()
            rt.tracker.reset()
            rt.last_bbox = None
        else:
            raise HTTPException(status_code=400, detail="unknown action")
        return {"ok": True, "compose_text": rt.composer.text}

    @api.post("/tts")
    def tts_speak(payload: dict) -> dict:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=400, detail="missing text")
        tts.speak(text)
        return {"ok": True}

    @api.post("/video/analyze")
    async def analyze_video_endpoint(
        file: Optional[UploadFile] = File(default=None),
        source_url: str = Form(""),
        pipeline: str = Form("landmarks"),
        mode: str = Form("both"),
        preprocess: str = Form("1"),
        skin_mask: str = Form("0"),
        mask_space: str = Form("ycrcb"),
        use_tracker: str = Form("1"),
        confidence_threshold: float = Form(0.75),
        stable_frames_min: int = Form(6),
        pause_ms_min: int = Form(350),
        cooldown_ms: int = Form(800),
        sample_fps: float = Form(4.0),
        max_frames: int = Form(0),
    ) -> JSONResponse:
        source_url = str(source_url or "").strip()
        if file is None and not source_url:
            raise HTTPException(status_code=400, detail="subí un archivo o pegá una URL")

        job_id = str(uuid.uuid4())
        job_dir = artifacts_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            if file is not None:
                suffix = Path(file.filename or "uploaded.mp4").suffix or ".mp4"
                input_path = job_dir / f"input{suffix}"
                raw = await file.read()
                input_path.write_bytes(raw)
            else:
                input_path = download_video_source(source_url, job_dir)

            pipe = load_video_pipeline(pipelines, pipeline)
            config = VideoAnalysisConfig(
                pipeline_name=pipeline,
                mode=mode if mode in ("both", "words", "spelling") else "both",
                preprocess=preprocess == "1",
                skin_mask=skin_mask == "1",
                mask_space=mask_space if mask_space in ("ycrcb", "hsv") else "ycrcb",
                use_tracker=use_tracker == "1",
                confidence_threshold=float(confidence_threshold),
                stable_frames_min=int(stable_frames_min),
                pause_ms_min=int(pause_ms_min),
                cooldown_ms=int(cooldown_ms),
                sample_fps=float(sample_fps),
                max_frames=int(max_frames),
            )
            output_video_path = job_dir / "processed.mp4"
            result = analyze_video(
                video_path=input_path,
                pipeline=pipe,
                detector=detector,
                skin_cfg=skin_cfg,
                config=config,
                output_video_path=output_video_path,
            )

            summary_path = job_dir / "summary.json"
            summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return JSONResponse(
                {
                    "ok": True,
                    "job_id": job_id,
                    "input_video_url": f"/artifacts/{job_id}/{input_path.name}",
                    "processed_video_url": f"/artifacts/{job_id}/processed.mp4",
                    "summary_url": f"/artifacts/{job_id}/summary.json",
                    "predicted_text": result["predicted_text"],
                    "predicted_tokens": result["predicted_tokens"],
                    "frames_total": result["frames_total"],
                    "frames_used": result["frames_used"],
                    "predictions_count": result["predictions_count"],
                    "avg_confidence": result["avg_confidence"],
                    "effective_fps": result["effective_fps"],
                    "elapsed_s": result["elapsed_s"],
                    "config": result["config"],
                }
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @api.post("/infer_frame")
    async def infer_frame(
        image: UploadFile = File(...),
        pipeline: str = Form("landmarks"),
        mode: str = Form("both"),
        preprocess: str = Form("1"),
        skin_mask: str = Form("0"),
        mask_space: str = Form("ycrcb"),
        use_tracker: str = Form("1"),
        debug_mask_thumb: str = Form("0"),
        debug_roi_thumb: str = Form("0"),
        debug_compose: str = Form("0"),
        confidence_threshold: float = Form(0.75),
        stable_frames_min: int = Form(6),
        pause_ms_min: int = Form(350),
        cooldown_ms: int = Form(800),
        ts_ms: int = Form(0),
        session_id: str = Form(""),
    ) -> JSONResponse:
        t0 = time.perf_counter()
        raw = await image.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise HTTPException(status_code=400, detail="bad image")

        work = frame_bgr
        if preprocess == "1":
            work = apply_clahe(work)
            work = maybe_denoise(work)

        mask = None
        if skin_mask == "1":
            if mask_space == "hsv":
                mask = skin_mask_hsv(work, skin_cfg)
            else:
                mask = skin_mask_ycrcb(work, skin_cfg)

        frame_rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        hand = None
        holistic_res = None
        if pipeline == "multimodal":
            holistic_res = holistic.detect(frame_rgb)
        else:
            hand = detector.detect(frame_rgb)

        no_hand = hand is None and not (holistic_res is not None and holistic_res.any_hand())
        label = "no_hand"
        conf = 0.0
        bbox = None
        tracker_status = "off"

        rt: Optional[SessionRuntime] = None
        if session_id:
            rt = sessions.get(session_id)
            if rt is None:
                sessions[session_id] = SessionRuntime(composer=ComposeState(), tracker=RoiTracker(), last_bbox=None)
                rt = sessions[session_id]
            rt.composer.config = ComposeConfig(
                confidence_threshold=float(confidence_threshold),
                stable_frames_min=int(stable_frames_min),
                pause_ms_min=int(pause_ms_min),
                cooldown_ms=int(cooldown_ms),
            )
            if mode not in ("words", "spelling", "both"):
                mode = "both"
            rt.composer.mode = ComposeMode(name=mode)

        if pipeline == "multimodal":
            if holistic_res is not None:
                bbox = multimodal_bbox(holistic_res, work.shape[1], work.shape[0])
            if bbox is not None and rt is not None and use_tracker == "1":
                rt.tracker.update_from_detection(work, bbox)
                rt.last_bbox = bbox
                tracker_status = rt.tracker.status
            elif rt is not None and use_tracker == "1":
                tracked = rt.tracker.track(work)
                tracker_status = rt.tracker.status
                if tracked is not None:
                    bbox = tracked
                    rt.last_bbox = tracked
            elif rt is not None:
                tracker_status = "off"
        elif hand is not None:
            bbox = hand.bbox
            if rt is not None and bbox is not None and use_tracker == "1":
                rt.tracker.update_from_detection(work, bbox)
                rt.last_bbox = bbox
                tracker_status = rt.tracker.status
        else:
            if rt is not None and use_tracker == "1":
                tracked = rt.tracker.track(work)
                tracker_status = rt.tracker.status
                if tracked is not None:
                    bbox = tracked
                    rt.last_bbox = tracked

        # If we have no landmarks but do have a tracked bbox, allow CNN inference to run.
        if pipeline != "multimodal" and hand is None and bbox is not None:
            from ..hand import HandResult

            hand = HandResult(landmarks=None, handedness=None, bbox=bbox, score=0.0)

        if pipeline == "multimodal":
            pipe = pipelines.get(pipeline)
            label, conf = pipe.predict_multimodal(
                holistic_res.left_hand if holistic_res is not None else None,
                holistic_res.right_hand if holistic_res is not None else None,
                holistic_res.pose if holistic_res is not None else None,
                holistic_res.face if holistic_res is not None else None,
            )
            no_hand = label == "no_hand"
            if bbox is None and rt is not None and rt.last_bbox is not None:
                bbox = rt.last_bbox
        elif hand is not None:
            pipe = pipelines.get(pipeline)
            label, conf = pipe.predict(work, hand, skin_mask=mask)
            no_hand = label == "no_hand" or (hand.landmarks is None and pipeline == "landmarks")

        landmarks_px = None
        if pipeline == "multimodal" and holistic_res is not None:
            primary = multimodal_primary_hand_landmarks(holistic_res)
            if primary is not None:
                h_img, w_img = work.shape[:2]
                pts = primary[:, :2].copy()
                pts[:, 0] *= w_img
                pts[:, 1] *= h_img
                landmarks_px = pts.astype(np.int32).tolist()
        elif hand is not None and hand.landmarks is not None:
            h_img, w_img = work.shape[:2]
            pts = hand.landmarks[:, :2].copy()
            pts[:, 0] *= w_img
            pts[:, 1] *= h_img
            landmarks_px = pts.astype(np.int32).tolist()

        def _encode_png_b64(img_bgr_or_gray: np.ndarray, size: int = 120) -> Optional[str]:
            if img_bgr_or_gray is None:
                return None
            if img_bgr_or_gray.ndim == 2:
                thumb = img_bgr_or_gray
            else:
                thumb = img_bgr_or_gray
            h0, w0 = thumb.shape[:2]
            scale = size / max(h0, w0)
            new_w = max(1, int(w0 * scale))
            new_h = max(1, int(h0 * scale))
            thumb = cv2.resize(thumb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if thumb.ndim == 2:
                enc_ok, buf = cv2.imencode(".png", thumb)
            else:
                enc_ok, buf = cv2.imencode(".png", thumb)
            if not enc_ok:
                return None
            return base64.b64encode(buf.tobytes()).decode("ascii")

        debug_mask_png_b64 = None
        if debug_mask_thumb == "1" and mask is not None:
            debug_mask_png_b64 = _encode_png_b64(mask)

        debug_roi_png_b64 = None
        if debug_roi_thumb == "1" and bbox is not None:
            x, y, w, h = bbox
            roi = work[y : y + h, x : x + w]
            if roi is not None and roi.size:
                debug_roi_png_b64 = _encode_png_b64(roi)

        compose_text = ""
        new_token = None
        if rt is not None:
            new_token = rt.composer.update(label=label, confidence=conf, no_hand=no_hand, ts_ms=int(ts_ms or (time.time() * 1000)))
            compose_text = rt.composer.text
        compose_debug = rt.composer.debug_state() if (rt is not None and debug_compose == "1") else None

        if rt is not None:
            rt.log.append(
                {
                    "ts_ms": int(ts_ms or (time.time() * 1000)),
                    "pipeline": pipeline,
                    "mode": mode,
                    "preprocess": preprocess == "1",
                    "skin_mask": skin_mask == "1",
                    "mask_space": mask_space,
                    "use_tracker": use_tracker == "1",
                    "label": label,
                    "confidence": float(conf),
                    "no_hand": bool(no_hand),
                    "bbox": bbox,
                    "tracker_status": tracker_status,
                    "new_token": new_token,
                }
            )
            if len(rt.log) > 2000:
                rt.log = rt.log[-2000:]

        server_ms = (time.perf_counter() - t0) * 1000.0
        return JSONResponse(
            {
                "label": label,
                "confidence": float(conf),
                "no_hand": bool(no_hand),
                "new_token": new_token,
                "compose_text": compose_text,
                "server_ms": server_ms,
                "bbox": bbox,
                "tracker_status": tracker_status,
                "landmarks_px": landmarks_px,
                "debug_mask_png_b64": debug_mask_png_b64,
                "debug_roi_png_b64": debug_roi_png_b64,
                "compose_debug": compose_debug,
            }
        )

    app.include_router(api)
    app.mount("/artifacts", StaticFiles(directory=str(artifacts_dir)), name="artifacts")
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks-model", type=str, default="")
    ap.add_argument("--cnn-model", type=str, default="")
    ap.add_argument("--sequence-model", type=str, default="")
    ap.add_argument("--multimodal-model", type=str, default="")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open-browser", action="store_true", help="Open the app URL in the default browser")
    ap.add_argument(
        "--ui-dir",
        type=str,
        default="",
        help="If set (or if web-ui/dist exists), serve built static frontend from this directory",
    )
    args = ap.parse_args()

    pipelines = LoadedPipelines()
    if args.landmarks_model:
        pipelines.landmarks = LandmarksPipeline.load(Path(args.landmarks_model))
    if args.cnn_model:
        pipelines.cnn = CnnPipeline.load(Path(args.cnn_model))
    if args.sequence_model:
        pipelines.sequence = SequencePipeline.load(Path(args.sequence_model))
    if args.multimodal_model:
        pipelines.multimodal = MultimodalSequencePipeline.load(Path(args.multimodal_model))

    import uvicorn

    app = create_app(pipelines)

    def _default_ui_dir() -> Path:
        # 1) PyInstaller bundle (sys._MEIPASS)
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            p = Path(meipass) / "web-ui" / "dist"
            if p.exists():
                return p
        # 2) Repo checkout
        return Path(__file__).resolve().parents[3] / "web-ui" / "dist"

    # Serve built React UI if available.
    ui_dir = Path(args.ui_dir) if args.ui_dir else _default_ui_dir()
    if ui_dir.exists() and ui_dir.is_dir():
        # Serve React build at root. API is under /api so there is no path clash.
        app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui_root")

    if args.open_browser:
        import threading
        import webbrowser

        url = f"http://{args.host}:{args.port}"

        def _open() -> None:
            time.sleep(0.8)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
