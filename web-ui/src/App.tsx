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
  ttsSpeak,
  type ComposeMode,
  type AnalyzeVideoResponse,
  type InferResponse,
  type MaskSpace,
  type PipelineName,
} from "./api";

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

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [sessionId, setSessionId] = useState<string>("");
  const [modelsText, setModelsText] = useState<string>("models: loading...");

  const [pipeline, setPipeline] = useState<PipelineName>("landmarks");
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

  useEffect(() => {
    (async () => {
      try {
        const h = await getHealth(baseUrl);
        const c = await getClasses(baseUrl);
        const hasL = !!h?.pipelines?.landmarks;
        const hasC = !!h?.pipelines?.cnn;
        const hasS = !!h?.pipelines?.sequence;
        const hasM = !!h?.pipelines?.multimodal;
        const lCount = c?.classes?.landmarks?.length ?? 0;
        const cCount = c?.classes?.cnn?.length ?? 0;
        const sCount = c?.classes?.sequence?.length ?? 0;
        const mCount = c?.classes?.multimodal?.length ?? 0;
        setModelsText(
          `models: landmarks=${hasL ? "on" : "off"}(${lCount}) cnn=${hasC ? "on" : "off"}(${cCount}) sequence=${hasS ? "on" : "off"}(${sCount}) multimodal=${hasM ? "on" : "off"}(${mCount})`,
        );
        const tools = h?.tools || {};
        setToolsText(
          `tools: ffmpeg=${tools.ffmpeg ? "on" : "off"} yt-dlp=${tools.yt_dlp_module || tools.yt_dlp_exe ? "on" : "off"}`,
        );
        if (!hasL && hasC) setPipeline("cnn");
        if (!hasL && !hasC && hasS) setPipeline("sequence");
        if (!hasL && !hasC && !hasS && hasM) setPipeline("multimodal");
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
        const extra = resp.compose_debug ? ` stable=${resp.compose_debug.stable_count} cand=${resp.compose_debug.candidate_ready ? resp.compose_debug.candidate_label : "-"}` : "";
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
    <div className="page">
      <header className="header">
        <div>
          <h2>VC-pria — Web UI</h2>
          <div className="muted mono">{modelsText}</div>
          <div className="muted mono">{toolsText}</div>
        </div>
        <div className="pill mono">
          session: <span className="mono">{sessionId || "(creating...)"}</span>
          <button
            onClick={async () => {
              if (!sessionId) return;
              try {
                await navigator.clipboard.writeText(sessionId);
              } catch {
                // ignore
              }
            }}
          >
            Copy
          </button>
          <button
            onClick={async () => {
              if (!sessionId) return;
              const j = await exportJson(baseUrl, sessionId);
              downloadText(`session_${sessionId}.json`, JSON.stringify(j, null, 2), "application/json");
            }}
          >
            Export JSON
          </button>
          <button
            onClick={async () => {
              if (!sessionId) return;
              const txt = await exportCsv(baseUrl, sessionId);
              downloadText(`session_${sessionId}.csv`, txt, "text/csv");
            }}
          >
            Export CSV
          </button>
        </div>
      </header>

      <div className="grid">
        <div className="card">
          <div className="stage">
            <video ref={videoRef} width={640} height={480} playsInline muted />
            <canvas ref={overlayCanvasRef} width={640} height={480} />
            <canvas ref={captureCanvasRef} width={640} height={480} style={{ display: "none" }} />
          </div>
          <div className="status mono">{status}</div>
        </div>

        <div className="card controls">
          <section>
            <h3>Pipeline</h3>
            <div className="row">
              <label>
                <input type="radio" checked={pipeline === "landmarks"} onChange={() => setPipeline("landmarks")} />
                landmarks
              </label>
              <label>
                <input type="radio" checked={pipeline === "cnn"} onChange={() => setPipeline("cnn")} />
                cnn
              </label>
              <label>
                <input type="radio" checked={pipeline === "sequence"} onChange={() => setPipeline("sequence")} />
                sequence
              </label>
              <label>
                <input type="radio" checked={pipeline === "multimodal"} onChange={() => setPipeline("multimodal")} />
                multimodal
              </label>
            </div>
          </section>

          <section>
            <h3>Compose mode</h3>
            <div className="row">
              <label>
                <input type="radio" checked={mode === "both"} onChange={() => setMode("both")} />
                both
              </label>
              <label>
                <input type="radio" checked={mode === "words"} onChange={() => setMode("words")} />
                words
              </label>
              <label>
                <input type="radio" checked={mode === "spelling"} onChange={() => setMode("spelling")} />
                spelling
              </label>
            </div>
          </section>

          <section>
            <h3>OpenCV toggles</h3>
            <div className="row">
              <label>
                <input type="checkbox" checked={preprocess} onChange={(e) => setPreprocess(e.target.checked)} />
                preprocess
              </label>
              <label>
                <input type="checkbox" checked={skinMask} onChange={(e) => setSkinMask(e.target.checked)} />
                skin mask
              </label>
              <label>
                mask space{" "}
                <select value={maskSpace} onChange={(e) => setMaskSpace(e.target.value as MaskSpace)}>
                  <option value="ycrcb">YCrCb</option>
                  <option value="hsv">HSV</option>
                </select>
              </label>
              <label>
                <input type="checkbox" checked={useTracker} onChange={(e) => setUseTracker(e.target.checked)} />
                tracker
              </label>
            </div>
          </section>

          <section>
            <h3>Debug</h3>
            <div className="row">
              <label>
                <input type="checkbox" checked={drawDebug} onChange={(e) => setDrawDebug(e.target.checked)} />
                draw overlay
              </label>
              <label>
                <input type="checkbox" checked={maskThumb} onChange={(e) => setMaskThumb(e.target.checked)} />
                mask thumb
              </label>
              <label>
                <input type="checkbox" checked={roiThumb} onChange={(e) => setRoiThumb(e.target.checked)} />
                ROI thumb
              </label>
              <label>
                <input type="checkbox" checked={composeDebug} onChange={(e) => setComposeDebug(e.target.checked)} />
                compose debug
              </label>
            </div>
          </section>

          <section>
            <h3>Thresholds</h3>
            <div className="sliders">
              <label>
                fps target <span className="mono">{fpsTarget}</span>
                <input type="range" min={2} max={20} value={fpsTarget} onChange={(e) => setFpsTarget(Number(e.target.value))} />
              </label>
              <label>
                conf thr <span className="mono">{confThr.toFixed(2)}</span>
                <input
                  type="range"
                  min={0.5}
                  max={0.95}
                  step={0.01}
                  value={confThr}
                  onChange={(e) => setConfThr(Number(e.target.value))}
                />
              </label>
              <label>
                stable frames <span className="mono">{stableFrames}</span>
                <input type="range" min={2} max={14} value={stableFrames} onChange={(e) => setStableFrames(Number(e.target.value))} />
              </label>
              <label>
                pause ms <span className="mono">{pauseMs}</span>
                <input type="range" min={150} max={900} value={pauseMs} onChange={(e) => setPauseMs(Number(e.target.value))} />
              </label>
              <label>
                cooldown ms <span className="mono">{cooldownMs}</span>
                <input type="range" min={200} max={2000} value={cooldownMs} onChange={(e) => setCooldownMs(Number(e.target.value))} />
              </label>
            </div>
          </section>

          <section>
            <h3>Phrase</h3>
            <textarea value={phrase} onChange={(e) => setPhrase(e.target.value)} placeholder="tokens..." />
            <div className="row">
              <button onClick={() => sessionAction(baseUrl, sessionId, "reset").then(() => setPhrase(""))}>Reset</button>
              <button onClick={() => sessionAction(baseUrl, sessionId, "space")}>Space</button>
              <button onClick={() => sessionAction(baseUrl, sessionId, "backspace")}>Backspace</button>
              <button onClick={() => ttsSpeak(baseUrl, phrase.trim())} disabled={!phrase.trim()}>
                Speak (TTS)
              </button>
            </div>
            <div className="row">
              <label>
                <input type="checkbox" checked={autoSpeak} onChange={(e) => setAutoSpeak(e.target.checked)} />
                auto-speak new tokens
              </label>
              <label>
                <input type="checkbox" checked={autoSpeakLetters} onChange={(e) => setAutoSpeakLetters(e.target.checked)} />
                include letters
              </label>
            </div>
          </section>

          <section>
            <h3>Video analysis</h3>
            <div className="stack-gap">
              <label className="stacked">
                YouTube / URL directa
                <input
                  type="text"
                  value={videoUrlInput}
                  onChange={(e) => setVideoUrlInput(e.target.value)}
                  placeholder="https://youtu.be/..."
                />
              </label>
              <label className="stacked">
                o subir archivo
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
                />
              </label>
              <div className="row">
                <label>
                  sample fps <span className="mono">{videoSampleFps}</span>
                  <input
                    type="range"
                    min={1}
                    max={12}
                    value={videoSampleFps}
                    onChange={(e) => setVideoSampleFps(Number(e.target.value))}
                  />
                </label>
                <label className="stacked narrow">
                  max frames (0 = todo)
                  <input
                    type="number"
                    min={0}
                    step={100}
                    value={videoMaxFrames}
                    onChange={(e) => setVideoMaxFrames(Number(e.target.value) || 0)}
                  />
                </label>
              </div>
              <div className="row">
                <button
                  onClick={async () => {
                    setVideoAnalyzeError("");
                    setVideoAnalyzeStatus("processing...");
                    setVideoJob(null);
                    try {
                      const result = await analyzeVideo(baseUrl, {
                        file: videoFile,
                        sourceUrl: videoUrlInput,
                        pipeline,
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
                  disabled={!videoFile && !videoUrlInput.trim()}
                >
                  Analyze video
                </button>
                <button
                  onClick={() => {
                    if (videoJob?.predicted_text?.trim()) {
                      ttsSpeak(baseUrl, videoJob.predicted_text.trim()).catch(() => {});
                    }
                  }}
                  disabled={!videoJob?.predicted_text?.trim()}
                >
                  Speak result
                </button>
              </div>
              <div className="status mono">{videoAnalyzeStatus}</div>
              {videoAnalyzeError ? <div className="error-box mono">{videoAnalyzeError}</div> : null}
              {videoJob ? (
                <div className="stack-gap">
                  <video
                    className="result-video"
                    controls
                    src={`${baseUrl}${videoJob.processed_video_url}`}
                  />
                  <textarea readOnly value={videoJob.predicted_text || ""} />
                  <div className="small mono">
                    frames={videoJob.frames_used}/{videoJob.frames_total} preds={videoJob.predictions_count} avg_conf=
                    {videoJob.avg_confidence.toFixed(2)} eff_fps={videoJob.effective_fps.toFixed(2)} time=
                    {videoJob.elapsed_s.toFixed(1)}s
                  </div>
                  <div className="row">
                    <button
                      onClick={async () => {
                        const res = await fetch(`${baseUrl}${videoJob.summary_url}`);
                        const txt = await res.text();
                        downloadText(`video_job_${videoJob.job_id}.json`, txt, "application/json");
                      }}
                    >
                      Export summary
                    </button>
                    <button
                      onClick={() => {
                        window.open(`${baseUrl}${videoJob.processed_video_url}`, "_blank");
                      }}
                    >
                      Open video
                    </button>
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        </div>
      </div>

      <footer className="muted small">
        API base: <span className="mono">{baseUrl || "(same-origin/proxy)"}</span>
      </footer>
    </div>
  );
}
