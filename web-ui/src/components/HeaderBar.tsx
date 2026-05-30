import { Badge, Button } from "./ui";

export function HeaderBar(props: {
  modelsText: string;
  toolsText: string;
  sessionId: string;
  onCopySession: () => void;
  onExportJson: () => void;
  onExportCsv: () => void;
}) {
  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-white/10 bg-gradient-to-br from-white/10 to-white/5 px-5 py-4 shadow-soft backdrop-blur">
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between gap-4">
          <h1 className="text-lg font-semibold tracking-tight text-slate-100">lsu-pria — Web UI</h1>
          <Badge tone="blue">
            <span className="mono">session</span>
            <span className="mono">{props.sessionId || "(creating...)"}</span>
          </Badge>
        </div>
        <div className="mono text-xs text-slate-300/80">{props.modelsText}</div>
        <div className="mono text-xs text-slate-300/80">{props.toolsText}</div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="ghost" disabled={!props.sessionId} onClick={props.onCopySession}>
          Copy session
        </Button>
        <Button variant="secondary" disabled={!props.sessionId} onClick={props.onExportJson}>
          Export JSON
        </Button>
        <Button variant="secondary" disabled={!props.sessionId} onClick={props.onExportCsv}>
          Export CSV
        </Button>
      </div>
    </div>
  );
}

