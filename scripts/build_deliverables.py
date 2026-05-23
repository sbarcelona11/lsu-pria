from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from fpdf import FPDF  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="deliverables/config.json")
    p.add_argument("--report-md", default="deliverables/report.md")
    p.add_argument("--pitch-md", default="deliverables/pitch.md")
    p.add_argument("--out-dir", default="deliverables/out")
    p.add_argument("--results-fig", default="results/precision_vs_fps.png")
    p.add_argument("--ablation-table", default="results/ablation_table.md")
    p.add_argument("--dataset-stats", default="results/dataset_stats.md")
    return p.parse_args()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _apply_placeholders(md: str, cfg: dict) -> str:
    members = cfg.get("members", [])
    if isinstance(members, list):
        members_s = ", ".join(str(x) for x in members)
    else:
        members_s = str(members)
    mapping = {
        "{{TITLE}}": str(cfg.get("title", "VC-pria")),
        "{{COURSE}}": str(cfg.get("course", "")),
        "{{GROUP}}": str(cfg.get("group", "")),
        "{{MEMBERS}}": members_s,
        "{{DATE}}": str(cfg.get("date", "")),
        "{{PITCH_MAX_MIN}}": str(cfg.get("target_pitch_minutes", 8)),
    }
    out = md
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def _md_to_lines(md: str) -> list[tuple[str, str]]:
    """
    Very small markdown subset:
    - # / ## / ### headings
    - bullets "- "
    - plain paragraphs
    Returns list of (kind, text): h1,h2,h3,bullet,para,blank
    """
    out: list[tuple[str, str]] = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            out.append(("blank", ""))
            continue
        if line.startswith("### "):
            out.append(("h3", line[4:].strip()))
        elif line.startswith("## "):
            out.append(("h2", line[3:].strip()))
        elif line.startswith("# "):
            out.append(("h1", line[2:].strip()))
        elif line.lstrip().startswith("- "):
            out.append(("bullet", line.lstrip()[2:].strip()))
        else:
            out.append(("para", line.strip()))
    return out


def _sanitize_pdf_text(s: str) -> str:
    # fpdf core fonts are Latin-1; replace common Unicode punctuation.
    s = (
        s.replace("—", "-")
        .replace("–", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("•", "*")
        .replace("→", "->")
        .replace("≤", "<=")
        .replace("≥", ">=")
        .replace("\u00a0", " ")
    )
    # Help line wrapping for long tokens (paths/flags) by adding break opportunities.
    if " " not in s and len(s) > 32:
        for ch in ["/", "_", "-", ".", ":", ",", ")"]:
            s = s.replace(ch, ch + " ")
    return s


class PdfDoc(FPDF):
    def header(self) -> None:
        # minimal header (empty)
        return

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", size=9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"{self.page_no()}", align="C")

    def safe_multi_cell(self, w: int, h: float, txt: str) -> None:
        # Ensure we always have width available (multi_cell uses remaining width when w==0).
        self.set_x(self.l_margin)
        try:
            self.multi_cell(w, h, txt)
        except Exception:
            # Last-resort: force break opportunities between characters.
            forced = " ".join(list(txt))
            self.set_x(self.l_margin)
            self.multi_cell(w, h, forced)


def render_report(md_path: Path, out_pdf: Path, fig_path: Path) -> None:
    pdf = PdfDoc(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    lines = _md_to_lines(_read(md_path))
    for kind, text in lines:
        text = _sanitize_pdf_text(text)
        if kind == "blank":
            pdf.ln(3)
            continue
        if kind == "h1":
            pdf.set_font("Helvetica", style="B", size=18)
            pdf.safe_multi_cell(0, 9, text)
            pdf.ln(2)
        elif kind == "h2":
            pdf.set_font("Helvetica", style="B", size=14)
            pdf.safe_multi_cell(0, 7, text)
            pdf.ln(1)
        elif kind == "h3":
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.safe_multi_cell(0, 6, text)
        elif kind == "bullet":
            pdf.set_font("Helvetica", size=11)
            pdf.safe_multi_cell(0, 5.5, f"- {text}")
        else:
            pdf.set_font("Helvetica", size=11)
            # Remove basic markdown emphasis markers for readability
            clean = re.sub(r"[*_`]", "", text)
            pdf.safe_multi_cell(0, 5.5, clean)

    if fig_path.exists():
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.multi_cell(0, 7, "Figura — Precisión vs FPS")
        pdf.ln(2)
        # Fit image within page margins.
        max_w = 180
        pdf.image(str(fig_path), w=max_w)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_pdf))


def render_pitch(md_path: Path, out_pdf: Path) -> None:
    md = _read(md_path)

    # Parse slides from "## Slide X — Title" headings.
    slides: list[tuple[str, list[str]]] = []
    current_title = "Pitch"
    current_body: list[str] = []
    for kind, text in _md_to_lines(md):
        if kind == "h2" and text.lower().startswith("slide"):
            if current_body or current_title:
                slides.append((current_title, current_body))
            current_title = text
            current_body = []
        elif kind in ("bullet", "para"):
            current_body.append(text)
    if current_body:
        slides.append((current_title, current_body))
    slides = [s for s in slides if s[0] and (s[1] or "slide" in s[0].lower())]

    class PitchPdf(FPDF):
        pass

    # 16:9 presentation canvas (approx 13.33" x 7.5" => 338.7mm x 190.5mm)
    pdf = PitchPdf(orientation="L", unit="mm", format=(338.7, 190.5))
    pdf.set_auto_page_break(auto=False)

    def draw_slide(title: str, bullets: list[str], idx: int, total: int) -> None:
        pdf.add_page()

        # Background
        pdf.set_fill_color(11, 16, 32)  # #0b1020
        pdf.rect(0, 0, 338.7, 190.5, style="F")

        # Header bar
        pdf.set_fill_color(255, 255, 255)
        # fpdf2 alpha support may be missing depending on build; emulate with darker solid.
        pdf.set_fill_color(30, 40, 80)
        pdf.rect(12, 12, 314.7, 22, style="F")

        pdf.set_xy(18, 16)
        pdf.set_text_color(232, 236, 255)
        pdf.set_font("Helvetica", style="B", size=22)
        pdf.cell(0, 10, _sanitize_pdf_text(title), ln=1)

        # Content area
        x0, y0 = 24, 46
        pdf.set_xy(x0, y0)
        pdf.set_font("Helvetica", size=18)
        pdf.set_text_color(232, 236, 255)

        clean_bullets = []
        for b in bullets:
            b2 = _sanitize_pdf_text(re.sub(r"[*_`]", "", b)).strip()
            if not b2:
                continue
            clean_bullets.append(b2)

        # Keep slides concise: show at most 8 bullets.
        if len(clean_bullets) > 8:
            clean_bullets = clean_bullets[:8] + ["(ver detalle en informe)"]

        line_h = 10.5
        for b in clean_bullets:
            pdf.set_x(x0)
            pdf.multi_cell(0, line_h, f"- {b}")
            pdf.ln(1.5)

        # Footer
        pdf.set_text_color(160, 166, 198)
        pdf.set_font("Helvetica", size=11)
        pdf.set_xy(12, 182)
        pdf.cell(0, 6, _sanitize_pdf_text("VC-pria — Pitch"), align="L")
        pdf.set_xy(12, 182)
        pdf.cell(314.7, 6, f"{idx}/{total}", align="R")

    total = max(1, len(slides))
    for i, (title, body) in enumerate(slides, start=1):
        # Trim "Slide X —" prefix for display
        display_title = title
        if "—" in display_title:
            display_title = display_title.split("—", 1)[1].strip()
        elif "-" in display_title and display_title.lower().startswith("slide"):
            # "Slide 1 - Title"
            parts = display_title.split("-", 1)
            if len(parts) == 2:
                display_title = parts[1].strip()
        if not display_title:
            display_title = "Slide"
        draw_slide(display_title, body, i, total)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_pdf))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_pdf = out_dir / "informe.pdf"
    pitch_pdf = out_dir / "pitch.pdf"
    pitch_pptx = out_dir / "pitch.pptx"

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    # Auto-append artifacts if present.
    report_md = _apply_placeholders(_read(Path(args.report_md)), cfg)
    pitch_md = _apply_placeholders(_read(Path(args.pitch_md)), cfg)
    ablation = Path(args.ablation_table)
    ds = Path(args.dataset_stats)
    if ablation.exists():
        report_md += "\n\n## Anexo — Tabla ablation\n\n" + ablation.read_text(encoding="utf-8").strip() + "\n"
    if ds.exists():
        report_md += "\n\n## Anexo — Stats dataset\n\n" + ds.read_text(encoding="utf-8").strip() + "\n"

    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    report_tmp = tmp_dir / "report_resolved.md"
    pitch_tmp = tmp_dir / "pitch_resolved.md"
    report_tmp.write_text(report_md, encoding="utf-8")
    pitch_tmp.write_text(pitch_md, encoding="utf-8")

    render_report(report_tmp, report_pdf, Path(args.results_fig))
    render_pitch(pitch_tmp, pitch_pdf)
    # Also generate a PPTX deck (editable) for the pitch.
    try:
        subprocess.check_call(
            [
                sys.executable,
                str((Path(__file__).resolve().parent / "build_pitch_pptx.py").resolve()),
                "--config",
                args.config,
                "--pitch-md",
                str(pitch_tmp),
                "--out",
                str(pitch_pptx),
            ]
        )
    except Exception as e:
        raise SystemExit(
            "Failed to build pitch.pptx. Install deps with: pip install -r requirements.txt "
            f"(needs python-pptx). Underlying error: {e}"
        )
    print(f"Wrote: {report_pdf}")
    print(f"Wrote: {pitch_pdf}")
    print(f"Wrote: {pitch_pptx}")


if __name__ == "__main__":
    main()
