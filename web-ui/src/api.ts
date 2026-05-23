export type PipelineName = "landmarks" | "cnn" | "sequence";
export type ComposeMode = "both" | "words" | "spelling";
export type MaskSpace = "ycrcb" | "hsv";

export type InferResponse = {
  label: string;
  confidence: number;
  no_hand: boolean;
  new_token: string | null;
  compose_text: string;
  server_ms: number;
  bbox: [number, number, number, number] | null;
  tracker_status: string;
  landmarks_px: [number, number][] | null;
  debug_mask_png_b64?: string | null;
  debug_roi_png_b64?: string | null;
  compose_debug?: any;
};

export async function getHealth(baseUrl: string) {
  const r = await fetch(`${baseUrl}/api/health`);
  if (!r.ok) throw new Error("health failed");
  return (await r.json()) as any;
}

export async function getClasses(baseUrl: string) {
  const r = await fetch(`${baseUrl}/api/classes`);
  if (!r.ok) throw new Error("classes failed");
  return (await r.json()) as any;
}

export async function newSession(baseUrl: string) {
  const r = await fetch(`${baseUrl}/api/session/new`, { method: "POST" });
  if (!r.ok) throw new Error("session/new failed");
  return (await r.json()) as { session_id: string };
}

export async function sessionAction(baseUrl: string, sessionId: string, action: string) {
  const r = await fetch(`${baseUrl}/api/session/${sessionId}/action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  if (!r.ok) throw new Error("session action failed");
  return await r.json();
}

export async function ttsSpeak(baseUrl: string, text: string) {
  const r = await fetch(`${baseUrl}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error("tts failed");
  return await r.json();
}

export async function exportJson(baseUrl: string, sessionId: string) {
  const r = await fetch(`${baseUrl}/api/session/${sessionId}/export`);
  if (!r.ok) throw new Error("export json failed");
  return await r.json();
}

export async function exportCsv(baseUrl: string, sessionId: string) {
  const r = await fetch(`${baseUrl}/api/session/${sessionId}/export.csv`);
  if (!r.ok) throw new Error("export csv failed");
  return await r.text();
}

export async function inferFrame(
  baseUrl: string,
  payload: {
    blob: Blob;
    pipeline: PipelineName;
    mode: ComposeMode;
    preprocess: boolean;
    skin_mask: boolean;
    mask_space: MaskSpace;
    use_tracker: boolean;
    debug_mask_thumb: boolean;
    debug_roi_thumb: boolean;
    debug_compose: boolean;
    confidence_threshold: number;
    stable_frames_min: number;
    pause_ms_min: number;
    cooldown_ms: number;
    ts_ms: number;
    session_id: string;
  },
): Promise<InferResponse> {
  const fd = new FormData();
  fd.append("image", payload.blob, "frame.jpg");
  fd.append("pipeline", payload.pipeline);
  fd.append("mode", payload.mode);
  fd.append("preprocess", payload.preprocess ? "1" : "0");
  fd.append("skin_mask", payload.skin_mask ? "1" : "0");
  fd.append("mask_space", payload.mask_space);
  fd.append("use_tracker", payload.use_tracker ? "1" : "0");
  fd.append("debug_mask_thumb", payload.debug_mask_thumb ? "1" : "0");
  fd.append("debug_roi_thumb", payload.debug_roi_thumb ? "1" : "0");
  fd.append("debug_compose", payload.debug_compose ? "1" : "0");
  fd.append("confidence_threshold", String(payload.confidence_threshold));
  fd.append("stable_frames_min", String(payload.stable_frames_min));
  fd.append("pause_ms_min", String(payload.pause_ms_min));
  fd.append("cooldown_ms", String(payload.cooldown_ms));
  fd.append("ts_ms", String(payload.ts_ms));
  fd.append("session_id", payload.session_id);

  const r = await fetch(`${baseUrl}/api/infer_frame`, { method: "POST", body: fd });
  if (!r.ok) throw new Error("infer_frame failed");
  return (await r.json()) as InferResponse;
}
