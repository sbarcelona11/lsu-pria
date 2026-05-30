export function StatusBar(props: { text: string }) {
  return (
    <div className="mono mt-3 rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-xs text-slate-200/90">
      {props.text}
    </div>
  );
}

