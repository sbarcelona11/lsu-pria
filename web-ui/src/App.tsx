import { useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeVideo,
  exportCsv,
  exportJson,
  getClasses,
  getHealth,
  inferFrame,
  newSession,
  sessionAction,
  sltRealtimeReset,
  sltRealtimeStep,
  ttsSpeak,
  type AnalyzeVideoResponse,
  type ComposeMode,
  type InferResponse,
  type MaskSpace,
  type PipelineName,
  type SltRealtimeStepResponse,
} from "./api";
import { HeaderBar } from "./components/HeaderBar";
import { Badge, Button, Card, Toggle } from "./components/ui";
import { VideoStage } from "./components/VideoStage";
import { StatusBar } from "./components/StatusBar";
import { SectionTitle } from "./components/SectionTitle";

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

function downloadText(filename: string, text: string, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function App() {
  const baseUrl = useMemo(() => {
    const env = (import.meta as any).env?.VITE_API_BASE as string | undefined;
    // In dev, Vite proxy can be used by leaving this empty, so calls hit same-origin.
    return env?.trim() || "";
  }, []);

  const videoRef = useRef<HTMLVideoElement>(null!);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null!);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null!);

  const [sessionId, setSessionId] = useState<string>("");
  const [modelsText, setModelsText] = useState<string>("models: loading...");

  const [pipeline, setPipeline] = useState<PipelineName>("landmarks");
  const [videoPipeline, setVideoPipeline] = useState<PipelineName>("landmarks");
  const [mode, setMode] = useState<ComposeMode>("both");
  const [preprocess, setPreprocess] = useState(true);
  const [skinMask, setSkinMask] = useState(false);
  const [maskSpace, setMaskSpace] = useState<MaskSpace>("ycrcb");
  const [useTracker, setUseTracker] = useState(true);

  const [drawDebug, setDrawDebug] = useState(true);
  const [maskThumb, setMaskThumb] = useState(false);
  const [roiThumb, setRoiThumb] = useState(false);
  const [composeDebug, setComposeDebug] = useState(false);

  const [fpsTarget, setFpsTarget] = useState(10);
  const [confThr, setConfThr] = useState(0.75);
  const [stableFrames, setStableFrames] = useState(6);
  const [pauseMs, setPauseMs] = useState(350);
  const [cooldownMs, setCooldownMs] = useState(800);

  const [status, setStatus] = useState("starting...");
  const [phrase, setPhrase] = useState("");
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [autoSpeakLetters, setAutoSpeakLetters] = useState(false);
  const [lastInfer, setLastInfer] = useState<InferResponse | null>(null);
  const [toolsText, setToolsText] = useState<string>("");
  const [videoUrlInput, setVideoUrlInput] = useState("");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoSampleFps, setVideoSampleFps] = useState(4);
  const [videoMaxFrames, setVideoMaxFrames] = useState(0);
  const [videoJob, setVideoJob] = useState<AnalyzeVideoResponse | null>(null);
  const [videoAnalyzeStatus, setVideoAnalyzeStatus] = useState("idle");
  const [videoAnalyzeError, setVideoAnalyzeError] = useState("");

  const [sltEnabled, setSltEnabled] = useState(false);
  const [sltSessionId, setSltSessionId] = useState("");
  const [sltWindowMs, setSltWindowMs] = useState(9000);
  const [sltStepMs, setSltStepMs] = useState(900);
  const [sltSampleFps, setSltSampleFps] = useState(6);
  const [sltMaxFrames, setSltMaxFrames] = useState(0);
  const [sltConfThr, setSltConfThr] = useState(0.5);
  const [sltStableTicks, setSltStableTicks] = useState(3);
  const [sltPauseMs, setSltPauseMs] = useState(900);
  const [sltCooldownMs, setSltCooldownMs] = useState(800);
  const [sltStatus, setSltStatus] = useState("idle");
  const [sltLast, setSltLast] = useState<SltRealtimeStepResponse | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const h = await getHealth(baseUrl);
        const c = await getClasses(baseUrl);
        const hasL = !!h?.pipelines?.landmarks;
        const hasC = !!h?.pipelines?.cnn;
        const hasS = !!h?.pipelines?.sequence;
        const hasM = !!h?.pipelines?.multimodal;
        const hasT = !!h?.pipelines?.slt;

        const p = [];
        p.push(`models: landmarks=${hasL ? "on" : "off"} cnn=${hasC ? "on" : "off"} sequence=${hasS ? "on" : "off"} multimodal=${hasM ? "on" : "off"} slt=${hasT ? "on" : "off"}`);
        const classes = c?.classes || {};
        const lCount = (classes.landmarks || []).length;
        const cCount = (classes.cnn || []).length;
        const sCount = (classes.sequence || []).length;
        const mCount = (classes.multimodal || []).length;
        const tCount = (classes.slt || []).length;
        p.push(`classes: landmarks=${lCount} cnn=${cCount} sequence=${sCount} multimodal=${mCount} slt=${tCount}`);
        setModelsText(p.join(" | "));

        const tools = h?.tools || {};
        setToolsText(`tools: ffmpeg=${tools.ffmpeg ? "on" : "off"} yt-dlp=${tools.yt_dlp_module || tools.yt_dlp_exe ? "on" : "off"}`);

        if (!hasL && hasC) setPipeline("cnn");
        if (!hasL && !hasC && hasS) setPipeline("sequence");
        if (!hasL && !hasC && !hasS && hasM) setPipeline("multimodal");
        if (!hasL && !hasC && !hasS && !hasM && hasT) setVideoPipeline("slt");
      } catch {
        setModelsText("models: error loading /health");
        setToolsText("tools: error loading /health");
      }
    })();
  }, [baseUrl]);

  useEffect(() => {
    (async () => {
      const sid = (await newSession(baseUrl)).session_id;
      setSessionId(sid);
    })();
  }, [baseUrl]);

  useEffect(() => {
    (async () => {
      const v = videoRef.current!;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      v.srcObject = stream;
      await v.play();
    })().catch((e) => setStatus(`camera error: ${String(e)}`));
  }, []);

  useEffect(() => {
    if (!sltEnabled) return;
    let running = true;
    const loop = async () => {
      if (!running) return;
      try {
        if (!videoRef.current || !captureCanvasRef.current) {
          setTimeout(loop, sltStepMs);
          return;
        }
        const v = videoRef.current;
        const cv = captureCanvasRef.current;
        const ctx = cv.getContext("2d")!;
        ctx.drawImage(v, 0, 0, cv.width, cv.height);
        const blob: Blob = await new Promise((res) => cv.toBlob((b) => res(b!), "image/jpeg", 0.85));
        setSltStatus("running");
        const resp = await sltRealtimeStep(baseUrl, {
          blob,
          session_id: sltSessionId,
          ts_ms: Date.now(),
          window_ms: sltWindowMs,
          sample_fps: sltSampleFps,
          max_frames: sltMaxFrames,
          preprocess,
          confidence_threshold: sltConfThr,
          stable_frames_min: sltStableTicks,
          pause_ms_min: sltPauseMs,
          cooldown_ms: sltCooldownMs,
          debug_compose: false,
        });
        setSltLast(resp);
        setSltSessionId(resp.session_id);
      } catch (e) {
        setSltStatus(`error: ${String(e)}`);
      } finally {
        setTimeout(loop, Math.max(250, sltStepMs));
      }
    };
    loop();
    return () => {
      running = false;
    };
  }, [
    sltEnabled,
    baseUrl,
    sltSessionId,
    sltWindowMs,
    sltStepMs,
    sltSampleFps,
    sltMaxFrames,
    sltConfThr,
    sltStableTicks,
    sltPauseMs,
    sltCooldownMs,
    preprocess,
  ]);

  function drawOverlay(payload: InferResponse) {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!drawDebug) return;

    if (payload.bbox) {
      const [x, y, w, h] = payload.bbox;
      ctx.lineWidth = 2;
      ctx.strokeStyle = payload.no_hand ? "rgba(255,100,100,0.9)" : "rgba(120,255,120,0.9)";
      ctx.strokeRect(x, y, w, h);
    }
    if (payload.landmarks_px) {
      const pts = payload.landmarks_px;
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(80,180,255,0.85)";
      for (const [a, b] of HAND_CONNECTIONS) {
        const pa = pts[a];
        const pb = pts[b];
        if (!pa || !pb) continue;
        ctx.beginPath();
        ctx.moveTo(pa[0], pa[1]);
        ctx.lineTo(pb[0], pb[1]);
        ctx.stroke();
      }
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        ctx.fillStyle = i === 0 ? "rgba(255,220,120,0.95)" : "rgba(255,255,255,0.9)";
        ctx.beginPath();
        ctx.arc(p[0], p[1], 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const drawThumb = (b64: string, x: number) =>
      new Promise<void>((resolve) => {
        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, x, canvas.height - img.height - 8);
          resolve();
        };
        img.src = `data:image/png;base64,${b64}`;
      });

    (async () => {
      if (payload.debug_mask_png_b64) await drawThumb(payload.debug_mask_png_b64, 8);
      if (payload.debug_roi_png_b64) await drawThumb(payload.debug_roi_png_b64, 140);
    })();
  }

  useEffect(() => {
    let running = true;
    const loop = async () => {
      if (!videoRef.current || !captureCanvasRef.current || !sessionId) {
        if (running) requestAnimationFrame(loop);
        return;
      }
      const t0 = performance.now();
      const v = videoRef.current;
      const cv = captureCanvasRef.current;
      const ctx = cv.getContext("2d")!;
      ctx.drawImage(v, 0, 0, cv.width, cv.height);
      const blob: Blob = await new Promise((res) => cv.toBlob((b) => res(b!), "image/jpeg", 0.85));

      try {
        const resp = await inferFrame(baseUrl, {
          blob,
          pipeline,
          mode,
          preprocess,
          skin_mask: skinMask,
          mask_space: maskSpace,
          use_tracker: useTracker,
          debug_mask_thumb: maskThumb,
          debug_roi_thumb: roiThumb,
          debug_compose: composeDebug,
          confidence_threshold: confThr,
          stable_frames_min: stableFrames,
          pause_ms_min: pauseMs,
          cooldown_ms: cooldownMs,
          ts_ms: Date.now(),
          session_id: sessionId,
        });
        setLastInfer(resp);
        setPhrase(resp.compose_text || "");
        if (autoSpeak && resp.new_token) {
          const tok = String(resp.new_token || "").trim();
          const isLetter = tok.length === 1 && /^[a-zA-Z]$/.test(tok);
          if (!isLetter || autoSpeakLetters) {
            // Fire-and-forget; avoid blocking the realtime loop.
            ttsSpeak(baseUrl, tok).catch(() => {});
          }
        }
        const clientFps = 1000 / Math.max(1, performance.now() - t0);
        const extra = resp.compose_debug
          ? ` stable=${resp.compose_debug.stable_count} cand=${resp.compose_debug.candidate_ready ? resp.compose_debug.candidate_label : "-"}`
          : "";
        setStatus(
          `pred=${resp.label} conf=${resp.confidence.toFixed(2)} no_hand=${resp.no_hand} tracker=${resp.tracker_status} client_fps=${clientFps.toFixed(1)} server_ms=${resp.server_ms.toFixed(1)}${extra}`,
        );
        drawOverlay(resp);
      } catch (e) {
        setStatus(`infer error: ${String(e)}`);
      }

      const elapsed = performance.now() - t0;
      const wait = Math.max(0, 1000 / fpsTarget - elapsed);
      setTimeout(() => {
        if (running) loop();
      }, wait);
    };
    loop();
    return () => {
      running = false;
    };
  }, [
    baseUrl,
    sessionId,
    pipeline,
    mode,
    preprocess,
    skinMask,
    maskSpace,
    useTracker,
    maskThumb,
    roiThumb,
    composeDebug,
    confThr,
    stableFrames,
    pauseMs,
    cooldownMs,
    fpsTarget,
    drawDebug,
    autoSpeak,
    autoSpeakLetters,
  ]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-5">
      <HeaderBar
        modelsText={modelsText}
        toolsText={toolsText}
        sessionId={sessionId}
        onCopySession={async () => {
          if (!sessionId) return;
          try {
            await navigator.clipboard.writeText(sessionId);
          } catch {
            // ignore
          }
        }}
        onExportJson={async () => {
          if (!sessionId) return;
          const j = await exportJson(baseUrl, sessionId);
          downloadText(`session_${sessionId}.json`, JSON.stringify(j, null, 2), "application/json");
        }}
        onExportCsv={async () => {
          if (!sessionId) return;
          const txt = await exportCsv(baseUrl, sessionId);
          downloadText(`session_${sessionId}.csv`, txt, "text/csv");
        }}
      />

      <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-[1.15fr,0.85fr]">
        <Card
          title="Realtime camera"
          subtitle="Inferencia frame-a-frame (webcam) + overlay de landmarks/bbox"
          right={
            <div className="flex items-center gap-2">
              <Badge tone={lastInfer?.no_hand ? "red" : "green"}>{lastInfer?.no_hand ? "no hand" : "hand"}</Badge>
              <Badge tone="gray">
                <span className="mono">{pipeline}</span>
              </Badge>
            </div>
          }
        >
          <VideoStage videoRef={videoRef} overlayCanvasRef={overlayCanvasRef} captureCanvasRef={captureCanvasRef} />
          <StatusBar text={status} />
        </Card>

        <div className="flex flex-col gap-4">
          <Card
            title="Realtime SLT (beta)"
            subtitle="Ventana móvil sobre la webcam (más lento que el loop normal). Requiere backend con --slt-model."
            right={<Badge tone={sltEnabled ? "green" : "gray"}>{sltEnabled ? "on" : "off"}</Badge>}
          >
            <div className="grid grid-cols-1 gap-2">
              <Toggle checked={sltEnabled} onChange={setSltEnabled} label="Enable realtime SLT" hint="Envía frames al backend y muestra predicción/compose." />
              <div className="grid grid-cols-2 gap-3">
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">window ms</div>
                  <input
                    type="number"
                    min={1000}
                    step={500}
                    value={sltWindowMs}
                    onChange={(e) => setSltWindowMs(Number(e.target.value) || 0)}
                    className="mono mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">step ms</div>
                  <input
                    type="number"
                    min={250}
                    step={50}
                    value={sltStepMs}
                    onChange={(e) => setSltStepMs(Number(e.target.value) || 0)}
                    className="mono mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">sample fps</div>
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={sltSampleFps}
                    onChange={(e) => setSltSampleFps(Number(e.target.value) || 0)}
                    className="mono mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">max frames (0=auto)</div>
                  <input
                    type="number"
                    min={0}
                    step={10}
                    value={sltMaxFrames}
                    onChange={(e) => setSltMaxFrames(Number(e.target.value) || 0)}
                    className="mono mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">conf thr</div>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={sltConfThr}
                    onChange={(e) => setSltConfThr(Number(e.target.value) || 0)}
                    className="mono mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">stable ticks</div>
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={sltStableTicks}
                    onChange={(e) => setSltStableTicks(Number(e.target.value) || 0)}
                    className="mono mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">pause ms</div>
                  <input
                    type="number"
                    min={0}
                    step={50}
                    value={sltPauseMs}
                    onChange={(e) => setSltPauseMs(Number(e.target.value) || 0)}
                    className="mono mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">cooldown ms</div>
                  <input
                    type="number"
                    min={0}
                    step={50}
                    value={sltCooldownMs}
                    onChange={(e) => setSltCooldownMs(Number(e.target.value) || 0)}
                    className="mono mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
              </div>

              <div className="mono rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-xs text-slate-200/90">
                {sltStatus} | session={sltSessionId || "(new)"} | pred={sltLast?.predicted_text || "-"} conf=
                {sltLast ? sltLast.confidence.toFixed(2) : "-"} | win_frames={sltLast?.frames_in_window ?? "-"} used=
                {sltLast?.frames_used ?? "-"} | server_ms={sltLast ? sltLast.server_ms.toFixed(1) : "-"}
              </div>

              <div>
                <SectionTitle title="Compose text (SLT)" hint="Se confirma cuando el pred se estabiliza y hay una pausa (conf baja/vacío)." />
                <textarea
                  readOnly
                  value={sltLast?.compose_text || ""}
                  className="mono mt-2 w-full rounded-2xl border border-white/10 bg-black/20 p-3 text-sm text-slate-100 outline-none"
                />
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    onClick={async () => {
                      if (!sltSessionId) return;
                      await sltRealtimeReset(baseUrl, sltSessionId);
                      setSltLast(null);
                    }}
                    disabled={!sltSessionId}
                  >
                    Reset SLT session
                  </Button>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Pipeline" subtitle="Elegí el modelo/pipeline para el loop realtime">
            <div className="grid grid-cols-2 gap-2">
              {(["landmarks", "cnn", "sequence", "multimodal"] as PipelineName[]).map((p) => (
                <label
                  key={p}
                  className={`flex cursor-pointer items-center justify-between rounded-xl border px-3 py-2 text-sm transition ${
                    pipeline === p ? "border-sky-400/40 bg-sky-500/10" : "border-white/10 bg-white/5 hover:bg-white/10"
                  }`}
                >
                  <span className="mono">{p}</span>
                  <input type="radio" checked={pipeline === p} onChange={() => setPipeline(p)} className="accent-sky-500" />
                </label>
              ))}
            </div>
            <div className="mt-3 text-xs text-slate-300/70">SLT queda reservado para análisis de video offline.</div>
          </Card>

          <Card title="Compose" subtitle="Cómo se construye la frase (tokens)">
            <div className="grid grid-cols-3 gap-2">
              {(["both", "words", "spelling"] as ComposeMode[]).map((m) => (
                <label
                  key={m}
                  className={`flex cursor-pointer items-center justify-between rounded-xl border px-3 py-2 text-sm transition ${
                    mode === m ? "border-sky-400/40 bg-sky-500/10" : "border-white/10 bg-white/5 hover:bg-white/10"
                  }`}
                >
                  <span className="mono">{m}</span>
                  <input type="radio" checked={mode === m} onChange={() => setMode(m)} className="accent-sky-500" />
                </label>
              ))}
            </div>

            <div className="mt-4 grid grid-cols-1 gap-2">
              <Toggle checked={autoSpeak} onChange={setAutoSpeak} label="Auto speak new tokens" hint="Dispara TTS cuando se confirma un token." />
              <Toggle checked={autoSpeakLetters} onChange={setAutoSpeakLetters} label="Include letters" hint="Incluye tokens de 1 letra en auto-speak." />
            </div>

            <div className="mt-4">
              <SectionTitle title="Phrase" hint="Se actualiza en vivo desde /api/infer." />
              <textarea
                readOnly
                value={phrase}
                className="mono mt-2 w-full rounded-2xl border border-white/10 bg-black/20 p-3 text-sm text-slate-100 outline-none"
              />
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  onClick={() => {
                    if (!sessionId) return;
                    sessionAction(baseUrl, sessionId, "space").catch(() => {});
                  }}
                  disabled={!sessionId}
                >
                  Space
                </Button>
                <Button
                  onClick={() => {
                    if (!sessionId) return;
                    sessionAction(baseUrl, sessionId, "backspace").catch(() => {});
                  }}
                  disabled={!sessionId}
                >
                  Backspace
                </Button>
                <Button
                  onClick={() => {
                    if (!sessionId) return;
                    sessionAction(baseUrl, sessionId, "reset").catch(() => {});
                  }}
                  disabled={!sessionId}
                >
                  Reset
                </Button>
                <Button
                  variant="primary"
                  onClick={() => {
                    const txt = phrase.trim();
                    if (txt) ttsSpeak(baseUrl, txt).catch(() => {});
                  }}
                  disabled={!phrase.trim()}
                >
                  Speak (TTS)
                </Button>
              </div>
            </div>
          </Card>

          <Card title="OpenCV + debug" subtitle="Toggles que afectan preprocesamiento y overlay">
            <div className="grid grid-cols-1 gap-2">
              <Toggle checked={preprocess} onChange={setPreprocess} label="preprocess" hint="Resize/blur/normalización según pipeline." />
              <Toggle checked={skinMask} onChange={setSkinMask} label="skin mask" hint="Máscara de piel para ROI (puede ayudar/dañar según luz)." />

              <label className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm">
                <div>
                  <div className="font-medium text-slate-100">mask space</div>
                  <div className="mt-0.5 text-xs text-slate-300/70">Espacio de color usado por la máscara.</div>
                </div>
                <select
                  value={maskSpace}
                  onChange={(e) => setMaskSpace(e.target.value as MaskSpace)}
                  className="rounded-lg border border-white/10 bg-black/30 px-2 py-1 text-sm text-slate-100 outline-none"
                >
                  <option value="ycrcb">YCrCb</option>
                  <option value="hsv">HSV</option>
                </select>
              </label>

              <Toggle checked={useTracker} onChange={setUseTracker} label="tracker" hint="BBox fallback cuando MediaPipe pierde la mano." />
              <Toggle checked={drawDebug} onChange={setDrawDebug} label="draw landmarks/bbox" />
              <Toggle checked={maskThumb} onChange={setMaskThumb} label="mask thumbnail" hint="Miniatura de máscara en overlay." />
              <Toggle checked={roiThumb} onChange={setRoiThumb} label="ROI thumbnail" hint="Miniatura de ROI en overlay." />
              <Toggle checked={composeDebug} onChange={setComposeDebug} label="compose debug" hint="Incluye contadores internos en status." />
            </div>

            <div className="mt-4 grid grid-cols-1 gap-3">
              <div>
                <SectionTitle title="FPS target" right={<span className="mono text-xs text-slate-300/80">{fpsTarget}</span>} />
                <input
                  type="range"
                  min={2}
                  max={24}
                  value={fpsTarget}
                  onChange={(e) => setFpsTarget(Number(e.target.value))}
                  className="mt-2 w-full accent-sky-500"
                />
              </div>
              <div>
                <SectionTitle title="Confidence threshold" right={<span className="mono text-xs text-slate-300/80">{confThr.toFixed(2)}</span>} />
                <input
                  type="range"
                  min={0.35}
                  max={0.95}
                  step={0.01}
                  value={confThr}
                  onChange={(e) => setConfThr(Number(e.target.value))}
                  className="mt-2 w-full accent-sky-500"
                />
              </div>
              <div>
                <SectionTitle title="Stable frames" right={<span className="mono text-xs text-slate-300/80">{stableFrames}</span>} />
                <input
                  type="range"
                  min={2}
                  max={14}
                  value={stableFrames}
                  onChange={(e) => setStableFrames(Number(e.target.value))}
                  className="mt-2 w-full accent-sky-500"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">pause ms min</div>
                  <input
                    type="number"
                    min={0}
                    step={25}
                    value={pauseMs}
                    onChange={(e) => setPauseMs(Number(e.target.value) || 0)}
                    className="mono mt-1 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">cooldown ms</div>
                  <input
                    type="number"
                    min={0}
                    step={50}
                    value={cooldownMs}
                    onChange={(e) => setCooldownMs(Number(e.target.value) || 0)}
                    className="mono mt-1 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
              </div>
            </div>
          </Card>

          <Card title="Offline video analysis" subtitle="Procesa un video o URL (YouTube si yt-dlp está disponible). El resultado queda en /artifacts.">
            <div className="grid grid-cols-1 gap-3">
              <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm">
                <div className="text-xs font-medium text-slate-100">pipeline</div>
                <select
                  value={videoPipeline}
                  onChange={(e) => setVideoPipeline(e.target.value as PipelineName)}
                  className="mono mt-1 w-full rounded-lg border border-white/10 bg-black/30 px-2 py-1 text-sm outline-none"
                >
                  <option value="landmarks">landmarks</option>
                  <option value="cnn">cnn</option>
                  <option value="sequence">sequence</option>
                  <option value="multimodal">multimodal</option>
                  <option value="slt">slt</option>
                </select>
              </label>

              <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm">
                <div className="text-xs font-medium text-slate-100">YouTube / URL</div>
                <input
                  type="text"
                  value={videoUrlInput}
                  onChange={(e) => setVideoUrlInput(e.target.value)}
                  placeholder="https://youtu.be/..."
                  className="mono mt-1 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-2 text-sm outline-none placeholder:text-slate-500"
                />
              </label>

              <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm">
                <div className="text-xs font-medium text-slate-100">o subir archivo</div>
                <input type="file" accept="video/*" onChange={(e) => setVideoFile(e.target.files?.[0] || null)} className="mt-2 w-full text-xs" />
              </label>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="flex items-center justify-between gap-2 text-xs font-medium text-slate-100">
                    <span>sample fps</span>
                    <span className="mono text-slate-300/80">{videoSampleFps}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={12}
                    value={videoSampleFps}
                    onChange={(e) => setVideoSampleFps(Number(e.target.value))}
                    className="mt-2 w-full accent-sky-500"
                  />
                </label>
                <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
                  <div className="text-xs font-medium text-slate-100">max frames (0 = todo)</div>
                  <input
                    type="number"
                    min={0}
                    step={100}
                    value={videoMaxFrames}
                    onChange={(e) => setVideoMaxFrames(Number(e.target.value) || 0)}
                    className="mono mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-2 py-1 text-sm outline-none"
                  />
                </label>
              </div>

              <div className="flex flex-wrap gap-2">
                <Button
                  variant="primary"
                  disabled={!videoFile && !videoUrlInput.trim()}
                  onClick={async () => {
                    setVideoAnalyzeError("");
                    setVideoAnalyzeStatus("processing...");
                    setVideoJob(null);
                    try {
                      const result = await analyzeVideo(baseUrl, {
                        file: videoFile,
                        sourceUrl: videoUrlInput,
                        pipeline: videoPipeline,
                        mode,
                        preprocess,
                        skin_mask: skinMask,
                        mask_space: maskSpace,
                        use_tracker: useTracker,
                        confidence_threshold: confThr,
                        stable_frames_min: stableFrames,
                        pause_ms_min: pauseMs,
                        cooldown_ms: cooldownMs,
                        sample_fps: videoSampleFps,
                        max_frames: videoMaxFrames,
                      });
                      setVideoJob(result);
                      setVideoAnalyzeStatus("done");
                    } catch (e) {
                      setVideoAnalyzeError(String(e));
                      setVideoAnalyzeStatus("error");
                    }
                  }}
                >
                  Analyze video
                </Button>
                <Button
                  disabled={!videoJob?.predicted_text?.trim()}
                  onClick={() => {
                    if (videoJob?.predicted_text?.trim()) ttsSpeak(baseUrl, videoJob.predicted_text.trim()).catch(() => {});
                  }}
                >
                  Speak result
                </Button>
              </div>

              <div className="mono rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-xs text-slate-200/90">
                {videoAnalyzeStatus}
              </div>

              {videoAnalyzeError ? (
                <div className="mono rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-xs text-rose-200">
                  {videoAnalyzeError}
                </div>
              ) : null}

              {videoJob ? (
                <div className="mt-1 grid gap-3">
                  <video className="w-full rounded-2xl border border-white/10" controls src={`${baseUrl}${videoJob.processed_video_url}`} />
                  <textarea
                    readOnly
                    value={videoJob.predicted_text || ""}
                    className="mono w-full rounded-2xl border border-white/10 bg-black/20 p-3 text-sm outline-none"
                  />
                  <div className="mono text-xs text-slate-300/80">
                    frames={videoJob.frames_used}/{videoJob.frames_total} preds={videoJob.predictions_count} avg_conf=
                    {videoJob.avg_confidence.toFixed(2)} eff_fps={videoJob.effective_fps.toFixed(2)} time=
                    {videoJob.elapsed_s.toFixed(1)}s
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      onClick={async () => {
                        const res = await fetch(`${baseUrl}${videoJob.summary_url}`);
                        const txt = await res.text();
                        downloadText(`video_job_${videoJob.job_id}.json`, txt, "application/json");
                      }}
                    >
                      Export summary
                    </Button>
                    <Button
                      onClick={() => {
                        window.open(`${baseUrl}${videoJob.processed_video_url}`, "_blank");
                      }}
                    >
                      Open video
                    </Button>
                  </div>
                </div>
              ) : null}
            </div>
          </Card>
        </div>
      </div>

      <footer className="mt-6 text-xs text-slate-300/70">
        API base: <span className="mono text-slate-200/90">{baseUrl || "(same-origin/proxy)"}</span>
      </footer>
    </div>
  );
}
