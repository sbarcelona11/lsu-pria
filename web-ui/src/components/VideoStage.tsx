import type { RefObject } from "react";

export function VideoStage(props: {
  videoRef: RefObject<HTMLVideoElement>;
  overlayCanvasRef: RefObject<HTMLCanvasElement>;
  captureCanvasRef: RefObject<HTMLCanvasElement>;
}) {
  return (
    <div className="relative aspect-[4/3] w-full overflow-hidden rounded-2xl border border-white/10 bg-black/30">
      <video ref={props.videoRef} playsInline muted className="absolute inset-0 h-full w-full object-cover" />
      <canvas ref={props.overlayCanvasRef} width={640} height={480} className="absolute inset-0 h-full w-full" />
      <canvas ref={props.captureCanvasRef} width={640} height={480} className="hidden" />
    </div>
  );
}
