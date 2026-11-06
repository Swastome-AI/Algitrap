"""Streamlit helpers: version-safe widgets, figure/table blocks with
publication downloads, the 3Dmol.js viewer, banners and the footer."""
from __future__ import annotations

import html as _html
import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from .constants import FOOTER_LINES
from .figs import MIME, Fig, export_bytes, plotly_config, to_plotly

try:
    ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:  # pragma: no cover
    ST_VER = (1, 40)

FORMATS = {"PNG (300 dpi)": "png", "TIFF (300 dpi)": "tiff", "SVG (vector)": "svg", "PDF (vector)": "pdf"}


def stretch() -> dict:
    return {"width": "stretch"} if ST_VER >= (1, 50) else {"use_container_width": True}


def show_plotly(fig, key: str, config: dict | None = None):
    """st.plotly_chart that works across Streamlit versions (width='stretch' vs use_container_width)."""
    kw = {"key": key}
    if config is not None:
        kw["config"] = config
    try:
        st.plotly_chart(fig, **kw, **stretch())
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kw)


def show_df(df, **kw):
    try:
        st.dataframe(df, **kw, **stretch())
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


CSS = """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .block-container {
  background-color: #ffffff !important; color: #000000 !important; }
[data-testid="stHeader"] { background: #ffffff !important; }
p, li, label, span, div, h1, h2, h3, h4, h5, h6, td, th { color: #000000; }
.block-container { padding-top: 1.2rem; max-width: 1500px; }
.at-hero { border: 2px solid #0072B2; border-radius: 12px; padding: 14px 18px; margin-bottom: 8px;
  background: linear-gradient(90deg, #eef6fb 0%, #ffffff 70%); }
.at-hero h1 { margin: 0; font-size: 2.1rem; color: #003b5c !important; }
.at-hero p { margin: 2px 0 0 0; font-size: 1.02rem; }
.at-strip { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 12px 0; }
.at-chip { border: 1px solid #9bb9cf; border-radius: 8px; padding: 6px 10px; background: #f7fafc; min-width: 150px;
  flex: 1 1 150px; }
.at-chip b { font-size: 0.78rem; text-transform: uppercase; letter-spacing: .03em; color: #0b4f7a !important; }
.at-chip div { font-size: 0.92rem; }
.at-box { border-left: 6px solid #0072B2; background: #eef6fb; padding: 10px 14px; border-radius: 6px; margin: 6px 0 12px 0; }
.at-interp { border-left: 6px solid #009E73; background: #eefaf5; padding: 10px 14px; border-radius: 6px; margin: 10px 0; }
.at-warnbox { border-left: 6px solid #E69F00; background: #fff8e8; padding: 10px 14px; border-radius: 6px; margin: 8px 0; }
.at-verdict-ok { border: 3px solid #009E73; background: #e9f9f1; border-radius: 12px; padding: 18px 22px; margin: 10px 0; }
.at-verdict-bad { border: 3px solid #D55E00; background: #fdeee6; border-radius: 12px; padding: 18px 22px; margin: 10px 0; }
.at-verdict-ok h2, .at-verdict-bad h2 { margin: 0 0 6px 0; font-size: 2.0rem; }
.at-footer { border-top: 1px solid #999; margin-top: 40px; padding-top: 10px; font-size: 0.8rem; color: #222 !important; }
[data-testid="stMetricValue"] { color: #000000 !important; font-size: 1.45rem; }
[data-testid="stMetricLabel"] p { color: #222 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { background: #f1f5f9; border-radius: 8px 8px 0 0; padding: 6px 12px; }
.stTabs [aria-selected="true"] { background: #d9ecf7 !important; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def box(text: str, kind: str = "box"):
    cls = {"box": "at-box", "interp": "at-interp", "warn": "at-warnbox"}[kind]
    st.markdown(f"<div class='{cls}'>{text}</div>", unsafe_allow_html=True)


def summary_box(title: str, body_md: str):
    st.markdown(f"<div class='at-box'><b>🧭 {title}</b><br>{body_md}</div>", unsafe_allow_html=True)


def interpretation(body_md: str, title: str = "How to interpret this"):
    st.markdown(f"<div class='at-interp'><b>🔎 {title}</b><br>{body_md}</div>", unsafe_allow_html=True)


def verdict(ok: bool, headline: str, body: str):
    cls = "at-verdict-ok" if ok else "at-verdict-bad"
    icon = "✅" if ok else "⚠️"
    st.markdown(f"<div class='{cls}'><h2>{icon} {headline}</h2><div>{body}</div></div>", unsafe_allow_html=True)


def chips(items: list[tuple[str, str]]):
    inner = "".join(f"<div class='at-chip'><b>{_html.escape(k)}</b><div>{_html.escape(v)}</div></div>" for k, v in items)
    st.markdown(f"<div class='at-strip'>{inner}</div>", unsafe_allow_html=True)


def footer():
    st.markdown("<div class='at-footer'>" + "<br>".join(_html.escape(x) for x in FOOTER_LINES) + "</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- registries
def register_fig(f: Fig, tab: str, caption: str = ""):
    st.session_state.setdefault("figs", {})
    st.session_state["figs"][f.id] = {"json": f.to_json(), "tab": tab, "title": f.title, "caption": caption}


def register_table(df: pd.DataFrame, tid: str, tab: str, caption: str):
    st.session_state.setdefault("tables", {})
    st.session_state["tables"][tid] = {"df": df, "tab": tab, "caption": caption}


@st.cache_data(max_entries=400, show_spinner=False)
def cached_export(fig_json: str, fmt: str, dpi: int) -> bytes:
    return export_bytes(Fig.from_json(fig_json), fmt, dpi=dpi)


def fig_block(f: Fig, tab: str, caption: str = "", height: int | None = None):
    register_fig(f, tab, caption)
    fig = to_plotly(f)
    if height:
        fig.update_layout(height=height)
    show_plotly(fig, f"pl_{f.id}", plotly_config(f.id))
    c1, c2, c3 = st.columns([1.3, 1.3, 2.4])
    with c1:
        choice = st.selectbox("Download this plot (publication quality)", ["— choose format —"] + list(FORMATS),
                              key=f"fmt_{f.id}")
    if choice in FORMATS:
        ext = FORMATS[choice]
        dpi = int(st.session_state.get("export_dpi", 300))
        with st.spinner("Rendering publication figure…"):
            data = cached_export(f.to_json(), ext, dpi)
        with c2:
            st.write("")
            st.download_button(f"⬇️ {f.id}.{ext}", data, file_name=f"{f.id}.{ext}", mime=MIME[ext],
                               key=f"dl_{f.id}_{ext}")
    if caption:
        with c3:
            st.caption(caption)


def table_block(df: pd.DataFrame, tid: str, tab: str, caption: str, height: int | None = None):
    register_table(df, tid, tab, caption)
    st.markdown(f"**{caption}**")
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    show_df(df, **kw)
    st.download_button(f"⬇️ {tid}.csv", df.to_csv(index=False).encode("utf-8"), file_name=f"{tid}.csv",
                       mime="text/csv", key=f"csv_{tid}")


def dict_table(d: dict, key_name="Metric", val_name="Value") -> pd.DataFrame:
    rows = []
    for k, v in d.items():
        if str(k).startswith("_") or isinstance(v, (list, dict, np.ndarray)):
            continue
        rows.append({key_name: k, val_name: v if not isinstance(v, float) else round(v, 4)})
    df = pd.DataFrame(rows)
    if len(df):
        df[val_name] = df[val_name].astype(str)
    return df


# ---------------------------------------------------------------- 3Dmol viewer
VIEWER = """
<div id="wrap" style="position:relative;width:100%;height:{h}px;border:1px solid #bbb;border-radius:8px;background:#fff">
  <div id="v" style="position:absolute;inset:0"></div>
  <div id="msg" style="position:absolute;left:10px;top:8px;font:13px Arial;color:#333">Loading 3-D viewer…</div>
  <div style="position:absolute;right:10px;bottom:6px;font:12px Arial;color:#444;background:rgba(255,255,255,.8);padding:2px 6px;border-radius:4px">{legend}</div>
</div>
<script>
const SRC = ["https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.0/3Dmol-min.js",
             "https://unpkg.com/3dmol@2.4.0/build/3Dmol-min.js",
             "https://3Dmol.org/build/3Dmol-min.js"];
const DATA = {data};
const STATIC = {static};
const FRAMES = {frames};
const STYLES = {styles};
const SPIN = {spin};
function init() {{
  try {{
    document.getElementById('msg').innerText = '';
    const viewer = $3Dmol.createViewer(document.getElementById('v'), {{backgroundColor: 'white'}});
    if (STATIC) {{ viewer.addModel(STATIC, 'pdb'); }}
    if (FRAMES) {{ viewer.addModelsAsFrames(DATA, 'pdb'); }} else {{ viewer.addModel(DATA, 'pdb'); }}
    viewer.setStyle({{}}, {{}});
    for (const s of STYLES) {{ viewer.addStyle(s[0], s[1]); }}
    viewer.zoomTo(); viewer.render();
    if (FRAMES) {{ viewer.animate({{loop: 'forward', interval: {interval}, reps: 0}}); }}
    if (SPIN) {{ viewer.spin('y', 0.4); }}
  }} catch (e) {{ document.getElementById('msg').innerText = 'Viewer error: ' + e; }}
}}
function load(i) {{
  if (typeof $3Dmol !== 'undefined') {{ init(); return; }}
  if (i >= SRC.length) {{ document.getElementById('msg').innerText =
      '3-D viewer could not be loaded (no internet?). The Plotly 3-D views still work.'; return; }}
  const s = document.createElement('script'); s.src = SRC[i]; s.onload = init; s.onerror = () => load(i + 1);
  document.head.appendChild(s);
}}
load(0);
</script>
"""


def _js(obj) -> str:
    return json.dumps(obj).replace("</", "<\\/")


def style_sets(kind: str, n_chains: int = 1, stick_r: float = 0.16, ca_r: float = 1.0, lines: bool = False) -> list:
    rep = "line" if lines else "stick"
    rep_args = {} if lines else {"radius": stick_r}
    if kind == "mg":
        return [[{"resn": "BEM"}, {rep: dict(color="#0072B2", **rep_args)}],
                [{"resn": "LGU"}, {rep: dict(color="#E69F00", **rep_args)}],
                [{"resn": ["BEM", "LGU", "CA", "MPL"], "invert": True}, {rep: dict(colorscheme="greenCarbon", **rep_args)}],
                [{"elem": "Ca"}, {"sphere": {"color": "#56B4E9", "radius": ca_r}}]]
    if kind == "element":
        return [[{}, {rep: dict(colorscheme="greenCarbon", **rep_args)}],
                [{"elem": "Ca"}, {"sphere": {"color": "#56B4E9", "radius": ca_r}}]]
    # rainbow by chain index stored in the B-factor column
    return [[{"resn": ["CA", "MPL"], "invert": True},
             {rep: dict(colorscheme={"prop": "b", "gradient": "roygb", "min": 0, "max": max(n_chains - 1, 1)}, **rep_args)}],
            [{"elem": "Ca"}, {"sphere": {"color": "#56B4E9", "radius": ca_r}}],
            [{"resn": "MPL"}, {"sphere": {"color": "#CC79A7", "radius": 0.9}}]]


def mol_viewer(pdb: str, styles: list, height: int = 480, frames: bool = False, static_pdb: str | None = None,
               legend: str = "", spin: bool = False, interval: int = 220):
    html = VIEWER.format(h=height, data=_js(pdb), static=_js(static_pdb) if static_pdb else "null",
                         frames="true" if frames else "false", styles=_js(styles), spin="true" if spin else "false",
                         interval=int(interval), legend=_html.escape(legend))
    components.html(html, height=height + 12)
