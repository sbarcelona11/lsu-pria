import type { ReactNode } from "react";

export function Card(props: { title?: string; subtitle?: string; right?: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-2xl border border-white/10 bg-white/5 shadow-soft backdrop-blur">
      {props.title ? (
        <header className="flex items-start justify-between gap-4 border-b border-white/10 px-5 py-4">
          <div className="min-w-0">
            <div className="text-sm font-semibold text-slate-100">{props.title}</div>
            {props.subtitle ? <div className="mt-0.5 text-xs text-slate-300/80">{props.subtitle}</div> : null}
          </div>
          {props.right ? <div className="shrink-0">{props.right}</div> : null}
        </header>
      ) : null}
      <div className="px-5 py-4">{props.children}</div>
    </section>
  );
}

export function Badge(props: { tone?: "gray" | "green" | "amber" | "red" | "blue"; children: ReactNode }) {
  const tone = props.tone || "gray";
  const toneClass =
    tone === "green"
      ? "bg-emerald-500/15 text-emerald-200 ring-emerald-500/30"
      : tone === "amber"
        ? "bg-amber-500/15 text-amber-200 ring-amber-500/30"
        : tone === "red"
          ? "bg-rose-500/15 text-rose-200 ring-rose-500/30"
          : tone === "blue"
            ? "bg-sky-500/15 text-sky-200 ring-sky-500/30"
            : "bg-white/10 text-slate-200 ring-white/20";
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs ring-1 ${toneClass}`}>
      {props.children}
    </span>
  );
}

export function Button(props: {
  variant?: "primary" | "secondary" | "ghost";
  disabled?: boolean;
  onClick?: () => void;
  children: ReactNode;
}) {
  const variant = props.variant || "secondary";
  const cls =
    variant === "primary"
      ? "bg-sky-500/90 hover:bg-sky-500 text-white ring-sky-400/40"
      : variant === "ghost"
        ? "bg-transparent hover:bg-white/5 text-slate-100 ring-white/10"
        : "bg-white/5 hover:bg-white/10 text-slate-100 ring-white/10";
  return (
    <button
      disabled={props.disabled}
      onClick={props.onClick}
      className={`inline-flex items-center justify-center gap-2 rounded-xl px-3 py-2 text-sm font-medium ring-1 transition disabled:cursor-not-allowed disabled:opacity-50 ${cls}`}
    >
      {props.children}
    </button>
  );
}

export function Toggle(props: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <label className="flex items-center justify-between gap-4 rounded-xl border border-white/10 bg-white/5 px-3 py-2">
      <div className="min-w-0">
        <div className="text-sm font-medium text-slate-100">{props.label}</div>
        {props.hint ? <div className="mt-0.5 text-xs text-slate-300/70">{props.hint}</div> : null}
      </div>
      <input
        type="checkbox"
        checked={props.checked}
        onChange={(e) => props.onChange(e.target.checked)}
        className="h-4 w-4 accent-sky-500"
      />
    </label>
  );
}

