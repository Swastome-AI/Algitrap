"""PDF (landscape A4), HTML and ZIP report builders."""
from __future__ import annotations

import base64
import datetime as dt
import html
import io
import json
import os
import re
import zipfile

import pandas as pd

from .constants import APP_NAME, APP_TAGLINE, APP_VERSION, FOOTER_LINES
from .figs import Fig, export_bytes


def safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_")
    return s or "User"


def _register_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    try:
        import matplotlib
        base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(base, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(base, "DejaVuSans-Bold.ttf")))
        from reportlab.lib.fonts import addMapping
        addMapping("DejaVu", 0, 0, "DejaVu")
        addMapping("DejaVu", 1, 0, "DejaVu-Bold")
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


_EMOJI = {"✅": "[OK]", "⚠️": "[!]", "⚠": "[!]", "❌": "[X]", "ℹ️": "[i]", "🟢": "", "🔴": "", "\ufe0f": ""}


def pdf_text(s: str) -> str:
    s = str(s)
    for k, v in _EMOJI.items():
        s = s.replace(k, v)
    return "".join(ch for ch in s if ord(ch) <= 0xFFFF and not (0x2600 <= ord(ch) <= 0x27BF))


def _fmt(v):
    if isinstance(v, float):
        if v != v:
            return "–"
        if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
            return f"{v:.3e}"
        return f"{v:,.4g}"
    return str(v)


def build_pdf(user: str, sections: list[dict], cover_metrics: list[tuple], findings: list[str],
              limitations: list[str], dpi: int = 200, progress=None, exporter=None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    font, bold = _register_font()
    ss = getSampleStyleSheet()
    st = {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName=bold, fontSize=26, leading=31, alignment=0),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontName=font, fontSize=13, leading=17,
                              textColor=colors.HexColor("#333333")),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName=bold, fontSize=17, leading=21, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName=bold, fontSize=12.5, leading=16, spaceAfter=3),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontName=font, fontSize=9.5, leading=13),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontName=font, fontSize=7.8, leading=10),
        "cap": ParagraphStyle("c", parent=ss["Normal"], fontName=font, fontSize=8.5, leading=11,
                              textColor=colors.HexColor("#222222")),
        "box": ParagraphStyle("x", parent=ss["Normal"], fontName=font, fontSize=9.5, leading=13,
                              backColor=colors.HexColor("#eef6fb"), borderColor=colors.HexColor("#0072B2"),
                              borderWidth=0.8, borderPadding=6, spaceBefore=4, spaceAfter=8),
    }
    buf = io.BytesIO()
    W, H = landscape(A4)

    def on_page(c, doc):
        c.saveState()
        c.setFont(font, 6.8)
        y = 0.95 * cm
        from reportlab.lib.utils import simpleSplit
        lines = []
        for ln in FOOTER_LINES:
            lines += simpleSplit(ln, font, 6.8, W - 4 * cm)
        for k, ln in enumerate(reversed(lines)):
            c.drawString(1.5 * cm, y + k * 8, ln)
        c.setStrokeColor(colors.HexColor("#999999"))
        c.line(1.5 * cm, y + len(lines) * 8 + 2, W - 1.5 * cm, y + len(lines) * 8 + 2)
        c.setFont(font, 8)
        c.drawRightString(W - 1.5 * cm, H - 1.0 * cm, f"{APP_NAME} report · Prepared by {user} · page {doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=2.4 * cm, title=f"{APP_NAME} report – {user}",
                            author=user)
    story = []
    # ------------------------------------------------------------------ cover
    story.append(Paragraph(f"{APP_NAME}", st["title"]))
    story.append(Paragraph(APP_TAGLINE, st["sub"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Prepared by {html.escape(user)}</b> · {dt.datetime.now():%d %B %Y, %H:%M} · "
                           f"{APP_NAME} v{APP_VERSION}", st["sub"]))
    story.append(Spacer(1, 14))
    rows = [[Paragraph("<b>Key metric</b>", st["body"]), Paragraph("<b>Value</b>", st["body"])]]
    rows += [[Paragraph(html.escape(str(k)), st["body"]), Paragraph(html.escape(_fmt(v)), st["body"])]
             for k, v in cover_metrics]
    t = Table(rows, colWidths=[11 * cm, 7 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9ecf7")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Key findings", st["h1"]))
    for f in findings:
        story.append(Paragraph("• " + html.escape(pdf_text(f)), st["body"]))
    story.append(PageBreak())
    # --------------------------------------------------------------- sections
    nfig = sum(len(s.get("figures", [])) for s in sections)
    done = 0
    for s in sections:
        story.append(Paragraph(html.escape(s["title"]), st["h1"]))
        if s.get("summary"):
            story.append(Paragraph(html.escape(pdf_text(s["summary"])), st["box"]))
        figs: list[Fig] = s.get("figures", [])
        cells = []
        for f in figs:
            png = (exporter or export_bytes)(f, "png", dpi=dpi)
            img = Image(io.BytesIO(png), width=12.6 * cm, height=8.82 * cm)
            cells.append([img, Paragraph(f"<b>Figure – {html.escape(f.title)}</b>", st["cap"])])
            done += 1
            if progress:
                progress(done / max(nfig, 1), f"Rendering figure {done}/{nfig}")
        for k in range(0, len(cells), 2):
            pair = cells[k:k + 2]
            row_img = [c[0] for c in pair] + ([""] if len(pair) == 1 else [])
            row_cap = [c[1] for c in pair] + ([""] if len(pair) == 1 else [])
            tt = Table([row_img, row_cap], colWidths=[13.2 * cm, 13.2 * cm])
            tt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(KeepTogether(tt))
            story.append(Spacer(1, 6))
        for cap, df in s.get("tables", []):
            if df is None or len(df) == 0:
                continue
            story.append(Paragraph(html.escape(cap), st["h2"]))
            d2 = df.head(30)
            cols = [str(c) for c in d2.columns][:9]
            data = [[Paragraph(f"<b>{html.escape(c)}</b>", st["small"]) for c in cols]]
            for _, r in d2.iterrows():
                data.append([Paragraph(html.escape(_fmt(r[c]))[:120], st["small"]) for c in d2.columns[:9]])
            cw = (W - 3 * cm) / len(cols)
            tb = Table(data, colWidths=[cw] * len(cols), repeatRows=1)
            tb.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.black),
                                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                                    ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(tb)
            if len(df) > 30:
                story.append(Paragraph(f"(first 30 of {len(df)} rows – full table in the ZIP)", st["small"]))
            story.append(Spacer(1, 8))
        if s.get("interpretation"):
            story.append(Paragraph("How to read these results", st["h2"]))
            story.append(Paragraph(html.escape(pdf_text(s["interpretation"])), st["body"]))
        story.append(PageBreak())
    story.append(Paragraph("Limitations and good practice", st["h1"]))
    for l_ in limitations:
        story.append(Paragraph("• " + html.escape(pdf_text(l_)), st["body"]))
        story.append(Spacer(1, 3))
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()


def build_html(user: str, sections: list[dict], cover_metrics: list[tuple], findings: list[str],
               limitations: list[str]) -> str:
    try:
        from plotly.offline import get_plotlyjs
        from .figs import to_plotly
        plotly_ok = True
    except Exception:
        plotly_ok = False
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{APP_NAME} report – {html.escape(user)}</title>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             "<style>body{font-family:Arial,Helvetica,sans-serif;background:#fff;color:#000;max-width:1200px;margin:auto;"
             "padding:16px}h1{border-bottom:3px solid #0072B2}h2{margin-top:36px;border-bottom:1px solid #999}"
             ".box{background:#eef6fb;border-left:5px solid #0072B2;padding:10px 14px;margin:10px 0}"
             "table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{border:1px solid #444;padding:4px 7px}"
             "th{background:#e9e9e9}.foot{font-size:11px;color:#333;border-top:1px solid #999;margin-top:40px;padding-top:8px}"
             ".fig{margin:14px 0}</style>"]
    if plotly_ok:
        parts.append(f"<script type='text/javascript'>{get_plotlyjs()}</script>")
    parts.append("</head><body>")
    parts.append(f"<h1>{APP_NAME} – {html.escape(APP_TAGLINE)}</h1>")
    parts.append(f"<p><b>Prepared by {html.escape(user)}</b> · {dt.datetime.now():%d %B %Y %H:%M} · v{APP_VERSION}</p>")
    parts.append("<table><tr><th>Key metric</th><th>Value</th></tr>" +
                 "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(_fmt(v))}</td></tr>"
                         for k, v in cover_metrics) + "</table>")
    parts.append("<h2>Key findings</h2><ul>" + "".join(f"<li>{html.escape(pdf_text(f))}</li>" for f in findings) + "</ul>")
    for s in sections:
        parts.append(f"<h2>{html.escape(s['title'])}</h2>")
        if s.get("summary"):
            parts.append(f"<div class='box'>{html.escape(s['summary'])}</div>")
        for f in s.get("figures", []):
            if plotly_ok:
                fig = to_plotly(f)
                parts.append("<div class='fig'>" + fig.to_html(full_html=False, include_plotlyjs=False) + "</div>")
            else:
                b64 = base64.b64encode(export_bytes(f, "png", dpi=110)).decode()
                parts.append(f"<div class='fig'><img style='max-width:100%' src='data:image/png;base64,{b64}'></div>")
        for cap, df in s.get("tables", []):
            if df is not None and len(df):
                parts.append(f"<h3>{html.escape(cap)}</h3>" + df.head(200).to_html(index=False, border=0))
        if s.get("interpretation"):
            parts.append(f"<p><b>How to read these results.</b> {html.escape(s['interpretation'])}</p>")
    parts.append("<h2>Limitations</h2><ul>" + "".join(f"<li>{html.escape(pdf_text(l_))}</li>" for l_ in limitations) + "</ul>")
    parts.append("<div class='foot'>" + "<br>".join(html.escape(x) for x in FOOTER_LINES) + "</div></body></html>")
    return "\n".join(parts)


def build_zip(user: str, figures: list[Fig], fig_formats: list[str], tables: dict[str, pd.DataFrame],
              structures: dict[str, str], reports: dict[str, bytes | str], params: dict, readme_extra: str = "",
              dpi: int = 300, progress=None, exporter=None) -> bytes:
    buf = io.BytesIO()
    root = f"{APP_NAME}_{safe_name(user)}_{dt.datetime.now():%Y%m%d_%H%M}"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        n = len(figures) * len(fig_formats)
        k = 0
        for f in figures:
            for fmt in fig_formats:
                z.writestr(f"{root}/figures/{fmt}/{f.id}.{fmt}", (exporter or export_bytes)(f, fmt, dpi=dpi))
                k += 1
                if progress:
                    progress(k / max(n, 1), f"Exporting {f.id}.{fmt}")
        for name, df in tables.items():
            z.writestr(f"{root}/tables/{name}.csv", df.to_csv(index=False))
        for name, txt in structures.items():
            z.writestr(f"{root}/structures/{name}", txt)
        for name, data in reports.items():
            z.writestr(f"{root}/reports/{name}", data)
        z.writestr(f"{root}/parameters.json", json.dumps(params, indent=2, default=str))
        readme = [f"{APP_NAME} v{APP_VERSION} – analysis bundle", f"Prepared by: {user}",
                  f"Created: {dt.datetime.now():%Y-%m-%d %H:%M}", "",
                  "Folders:", "  figures/<format>/  publication-quality plots (PNG/TIFF at "
                  f"{dpi} dpi, SVG and PDF are vector)", "  tables/            every results table as CSV",
                  "  structures/        PDB/SDF structures and multi-model trajectories (open in PyMOL, ChimeraX, VMD)",
                  "  reports/           PDF and interactive HTML report", "  parameters.json    every setting used",
                  "", readme_extra, "", *FOOTER_LINES]
        z.writestr(f"{root}/README.txt", "\n".join(readme))
    return buf.getvalue()
