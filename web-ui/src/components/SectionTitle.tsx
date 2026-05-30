import type { ReactNode } from "react";

export function SectionTitle(props: { title: string; hint?: string; right?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div>
        <div className="text-sm font-semibold text-slate-100">{props.title}</div>
        {props.hint ? <div className="mt-0.5 text-xs text-slate-300/70">{props.hint}</div> : null}
      </div>
      {props.right ? <div className="shrink-0">{props.right}</div> : null}
    </div>
  );
}

