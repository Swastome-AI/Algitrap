"""One figure description → two renderers.

`Fig` stores a plot as plain data (JSON-serialisable). `to_plotly` draws it
on screen (interactive toolbar), `to_matplotlib` / `export_bytes` render the
identical figure for publication (PNG/TIFF at 300 dpi, SVG, PDF) without
needing Kaleido/Chrome. All styling lives in this file so every tab matches:
white background, black text, grid lines on both axes, black frame, outside
ticks, colour-blind-safe Okabe–Ito palette.
"""
from __future__ import annotations

import hashlib
import io
import json
import textwrap

import numpy as np

from .constants import (EXPORT_DPI, EXPORT_HEIGHT_IN, EXPORT_WIDTH_IN, FONT_AXIS, FONT_LEGEND, FONT_TICK,
                        FONT_TITLE, PALETTE)

GRID = "#d0d0d0"
LIGHT_SCALES = {"Blues": ["#f7fbff", "#c6dbef", "#6baed6"], "Greens": ["#f7fcf5", "#c7e9c0", "#74c476"],
                "Oranges": ["#fff5eb", "#fdd0a2", "#fd8d3c"], "Purples": ["#fcfbfd", "#dadaeb", "#9e9ac8"]}


def _clean(v):
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if hasattr(v, "to_numpy") and not isinstance(v, (str, bytes)):   # pandas Series / Index
        return [_clean(x) for x in v.to_numpy().tolist()]
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def hex_rgba(hexcol: str, a: float) -> str:
    h = hexcol.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def wrap_title(t: str, width: int = 68, sep: str = "<br>") -> str:
    return sep.join(textwrap.wrap(t, width)) if len(t) > width else t


class Fig:
    def __init__(self, fig_id: str, title: str, xlabel: str = "", ylabel: str = "", kind: str = "xy",
                 logx: bool = False, logy: bool = False, xrange=None, yrange=None, barmode: str = "group",
                 legend: bool = True, height: int = 520, zlabel: str = "", categorical_x: bool = False,
                 xreverse: bool = False):
        self.d = {"id": fig_id, "title": title, "xlabel": xlabel, "ylabel": ylabel, "zlabel": zlabel, "kind": kind,
                  "logx": logx, "logy": logy, "xrange": xrange, "yrange": yrange, "barmode": barmode,
                  "legend": legend, "height": height, "traces": [], "lines": [], "categorical_x": categorical_x,
                  "xreverse": xreverse}
        self._ci = 0

    # ------------------------------------------------------------- helpers
    def _color(self, color):
        if color is not None:
            return color
        c = PALETTE[self._ci % len(PALETTE)]
        self._ci += 1
        return c

    @property
    def id(self):
        return self.d["id"]

    @property
    def title(self):
        return self.d["title"]

    def key(self) -> str:
        return hashlib.sha1(self.to_json().encode()).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps(_clean(self.d), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Fig":
        f = cls("x", "x")
        f.d = json.loads(s)
        return f

    # ------------------------------------------------------------- traces
    def line(self, x, y, name=None, color=None, dash="solid", width=2.5, markers=False, yerr=None):
        self.d["traces"].append(dict(t="line", x=_clean(x), y=_clean(y), name=name, color=self._color(color),
                                     dash=dash, width=width, markers=markers, yerr=_clean(yerr)))
        return self

    def scatter(self, x, y, name=None, color=None, size=9, symbol="circle", text=None, yerr=None, opacity=0.9,
                textpos="top center"):
        self.d["traces"].append(dict(t="scatter", x=_clean(x), y=_clean(y), name=name, color=self._color(color)
                                     if not isinstance(color, (list, tuple, np.ndarray)) else _clean(color),
                                     size=size, symbol=symbol, text=_clean(text), yerr=_clean(yerr),
                                     opacity=opacity, textpos=textpos))
        return self

    def bar(self, x, y, name=None, color=None, yerr=None, labels=True, fmt="{:.2f}", horizontal=False):
        col = _clean(color) if isinstance(color, (list, tuple, np.ndarray)) else self._color(color)
        self.d["traces"].append(dict(t="bar", x=_clean(x), y=_clean(y), name=name, color=col, yerr=_clean(yerr),
                                     labels=labels, fmt=fmt, horizontal=horizontal))
        return self

    def hist(self, values, bins=20, name=None, color=None, density=False, rng=None, labels=False):
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            v = np.zeros(1)
        counts, edges = np.histogram(v, bins=bins, range=rng)
        if density:
            counts = counts / max(counts.sum(), 1)
        self.d["traces"].append(dict(t="hist", x=_clean(0.5 * (edges[1:] + edges[:-1])), y=_clean(counts),
                                     w=float(edges[1] - edges[0]), name=name, color=self._color(color),
                                     labels=labels))
        return self

    def band(self, x, ylo, yhi, name=None, color=None, alpha=0.2):
        self.d["traces"].append(dict(t="band", x=_clean(x), ylo=_clean(ylo), yhi=_clean(yhi), name=name,
                                     color=self._color(color), alpha=alpha))
        return self

    def heatmap(self, z, x, y, scale="Blues", zlabel="", fmt="{:.2f}", annotate=True):
        self.d["kind"] = "heatmap"
        self.d["traces"].append(dict(t="heatmap", z=_clean(z), x=_clean(x), y=_clean(y), scale=scale,
                                     zlabel=zlabel, fmt=fmt, annotate=annotate))
        return self

    def scatter3d(self, x, y, z, name=None, color=None, size=4, opacity=0.9):
        self.d["kind"] = "3d"
        col = _clean(color) if isinstance(color, (list, tuple, np.ndarray)) else self._color(color)
        self.d["traces"].append(dict(t="scatter3d", x=_clean(x), y=_clean(y), z=_clean(z), name=name, color=col,
                                     size=size, opacity=opacity))
        return self

    def vline(self, x, label="", color="#000000", dash="dash"):
        self.d["lines"].append(dict(o="v", v=float(x), label=label, color=color, dash=dash))
        return self

    def hline(self, y, label="", color="#000000", dash="dash"):
        self.d["lines"].append(dict(o="h", v=float(y), label=label, color=color, dash=dash))
        return self


# =============================================================================
# Plotly
# =============================================================================
PLOTLY_DASH = {"solid": "solid", "dash": "dash", "dot": "dot", "dashdot": "dashdot"}


def plotly_config(fig_id: str) -> dict:
    return {"displaylogo": False, "responsive": True,
            "toImageButtonOptions": {"format": "png", "filename": fig_id, "width": 1500, "height": 1050, "scale": 2}}


def style_plotly(fig, d: dict):
    two_line = len(d["title"]) > 68
    fig.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", height=d.get("height", 520),
        font=dict(family="Arial, Helvetica, sans-serif", size=FONT_TICK, color="black"),
        title=dict(text=f"<b>{wrap_title(d['title'])}</b>", font=dict(size=FONT_TITLE, color="black"), x=0.01,
                   xanchor="left", y=0.97, yanchor="top"),
        legend=dict(font=dict(size=FONT_LEGEND, color="black"), bgcolor="rgba(255,255,255,0.9)", bordercolor="black",
                    borderwidth=1),
        showlegend=bool(d.get("legend", True)),
        margin=dict(l=85, r=35, t=110 if two_line else 80, b=80),
        hoverlabel=dict(bgcolor="white", font_size=14, font_color="black"),
    )
    if d["kind"] == "3d":
        ax = dict(showgrid=True, gridcolor=GRID, showline=True, linecolor="black", backgroundcolor="white",
                  showbackground=True, color="black", title_font=dict(size=FONT_AXIS), tickfont=dict(size=12))
        fig.update_layout(scene=dict(xaxis=dict(title="x (Å)", **ax), yaxis=dict(title="y (Å)", **ax),
                                     zaxis=dict(title="z (Å)", **ax), aspectmode="data"))
        return fig
    axis = dict(showgrid=True, gridcolor=GRID, gridwidth=1, showline=True, linecolor="black", linewidth=1.5,
                mirror=True, ticks="outside", tickcolor="black", ticklen=6, tickwidth=1.5, zeroline=False,
                title_font=dict(size=FONT_AXIS, color="black"), tickfont=dict(size=FONT_TICK, color="black"))
    fig.update_xaxes(title_text=d["xlabel"], type="log" if d["logx"] else None, **axis)
    fig.update_yaxes(title_text=d["ylabel"], type="log" if d["logy"] else None, **axis)
    if d.get("xrange"):
        fig.update_xaxes(range=d["xrange"])
    if d.get("yrange"):
        fig.update_yaxes(range=d["yrange"])
    if d.get("xreverse"):
        fig.update_xaxes(autorange="reversed")
    if d.get("hide_yticks"):
        fig.update_yaxes(showticklabels=False, ticks="")
    return fig


def to_plotly(f: Fig):
    import plotly.graph_objects as go
    d = f.d
    fig = go.Figure()
    has_bar_labels, ymax_bar, ymin_bar = False, None, 0.0
    for tr in d["traces"]:
        t = tr["t"]
        if t == "line":
            fig.add_trace(go.Scatter(
                x=tr["x"], y=tr["y"], name=tr["name"], mode="lines+markers" if tr["markers"] else "lines",
                line=dict(color=tr["color"], width=tr["width"], dash=PLOTLY_DASH.get(tr["dash"], "solid")),
                marker=dict(size=8, color=tr["color"], line=dict(color="black", width=1)),
                error_y=dict(type="data", array=tr["yerr"], visible=True, color="black") if tr["yerr"] else None,
                showlegend=tr["name"] is not None))
        elif t == "scatter":
            fig.add_trace(go.Scatter(
                x=tr["x"], y=tr["y"], name=tr["name"], mode="markers+text" if tr["text"] else "markers",
                text=tr["text"], textposition=tr.get("textpos", "top center"), textfont=dict(size=13, color="black"),
                marker=dict(size=tr["size"], color=tr["color"], symbol=tr["symbol"], opacity=tr["opacity"],
                            line=dict(color="black", width=1)),
                error_y=dict(type="data", array=tr["yerr"], visible=True, color="black") if tr["yerr"] else None,
                showlegend=tr["name"] is not None))
        elif t in ("bar", "hist"):
            horiz = tr.get("horizontal", False)
            txt = None
            if tr.get("labels"):
                fmt = tr.get("fmt", "{:.2f}")
                txt = [fmt.format(v) if v is not None else "" for v in tr["y"]]
                has_bar_labels = True
            yv = [v for v in tr["y"] if v is not None]
            if yv:
                errs = tr.get("yerr") or [0] * len(tr["y"])
                top = max((v or 0) + (e or 0) for v, e in zip(tr["y"], errs))
                ymax_bar = top if ymax_bar is None else max(ymax_bar, top)
                ymin_bar = min(ymin_bar, min(yv))
            kw = dict(name=tr["name"], marker=dict(color=tr["color"], line=dict(color="black", width=1)),
                      text=txt, textposition="outside", textfont=dict(color="black", size=13), cliponaxis=False,
                      showlegend=tr["name"] is not None)
            if t == "hist":
                kw["width"] = tr["w"] * 0.98
            if horiz:
                fig.add_trace(go.Bar(x=tr["y"], y=tr["x"], orientation="h", **kw))
            else:
                fig.add_trace(go.Bar(x=tr["x"], y=tr["y"],
                                     error_y=dict(type="data", array=tr.get("yerr"), visible=True, color="black")
                                     if tr.get("yerr") else None, **kw))
        elif t == "band":
            fig.add_trace(go.Scatter(x=tr["x"], y=tr["yhi"], mode="lines", line=dict(width=0), showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=tr["x"], y=tr["ylo"], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=hex_rgba(tr["color"], tr["alpha"]), name=tr["name"],
                                     showlegend=tr["name"] is not None))
        elif t == "heatmap":
            sc = LIGHT_SCALES.get(tr["scale"], LIGHT_SCALES["Blues"])
            cs = [[0, sc[0]], [0.5, sc[1]], [1, sc[2]]]
            kw = {}
            if tr["annotate"]:
                kw = dict(text=[[tr["fmt"].format(v) if v is not None else "" for v in row] for row in tr["z"]],
                          texttemplate="%{text}", textfont=dict(color="black", size=14))
            fig.add_trace(go.Heatmap(z=tr["z"], x=tr["x"], y=tr["y"], colorscale=cs,
                                     colorbar=dict(title=dict(text=tr["zlabel"], font=dict(size=14))), **kw))
        elif t == "scatter3d":
            fig.add_trace(go.Scatter3d(x=tr["x"], y=tr["y"], z=tr["z"], mode="markers", name=tr["name"],
                                       marker=dict(size=tr["size"], color=tr["color"], opacity=tr["opacity"],
                                                   line=dict(width=0)), showlegend=tr["name"] is not None))
    kv = 0
    for ln in d["lines"]:
        dash = PLOTLY_DASH.get(ln["dash"], "dash")
        if ln["o"] == "v":
            fig.add_vline(x=ln["v"], line=dict(color=ln["color"], dash=dash, width=2),
                          annotation=dict(text=ln["label"], font=dict(size=13, color="black"), yshift=-20 * kv,
                                          bgcolor="rgba(255,255,255,0.85)"), annotation_position="top right")
            kv += 1 if ln["label"] else 0
        else:
            fig.add_hline(y=ln["v"], line=dict(color=ln["color"], dash=dash, width=2),
                          annotation=dict(text=ln["label"], font=dict(size=13, color="black"),
                                          bgcolor="rgba(255,255,255,0.85)"), annotation_position="top left")
    if any(tr["t"] == "hist" for tr in d["traces"]):
        fig.update_layout(bargap=0.0, barmode="overlay" if sum(tr["t"] == "hist" for tr in d["traces"]) > 1 else "group")
        for tr_ in fig.data:
            if isinstance(tr_, go.Bar) and sum(t_["t"] == "hist" for t_ in d["traces"]) > 1:
                tr_.opacity = 0.65
    else:
        fig.update_layout(barmode=d.get("barmode", "group"))
    style_plotly(fig, d)
    if has_bar_labels and ymax_bar is not None and not d["logy"] and not d.get("yrange"):
        horiz = any(tr.get("horizontal") for tr in d["traces"] if tr["t"] == "bar")
        rng = [min(0, ymin_bar * 1.2), ymax_bar * 1.22 if ymax_bar > 0 else 1]
        if horiz:
            fig.update_xaxes(range=rng)
        else:
            fig.update_yaxes(range=rng)
    return fig


# =============================================================================
# Matplotlib (publication export)
# =============================================================================
MPL_DASH = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-."}


def _mpl_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 17, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12, "axes.edgecolor": "black",
        "axes.linewidth": 1.3, "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black",
        "ytick.color": "black", "savefig.facecolor": "white", "figure.facecolor": "white", "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def to_matplotlib(f: Fig, width_in: float = EXPORT_WIDTH_IN, height_in: float = EXPORT_HEIGHT_IN, dpi: int = 100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _mpl_style()
    d = f.d
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    if d["kind"] == "3d":
        ax = fig.add_subplot(111, projection="3d")
        for tr in d["traces"]:
            ax.scatter(tr["x"], tr["y"], tr["z"], s=max(tr["size"], 1) * 4, c=tr["color"], alpha=tr["opacity"],
                       label=tr["name"], depthshade=True, edgecolors="none")
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)"); ax.set_zlabel("z (Å)")
        try:
            xs = np.concatenate([np.asarray(tr["x"], float) for tr in d["traces"]])
            ys = np.concatenate([np.asarray(tr["y"], float) for tr in d["traces"]])
            zs = np.concatenate([np.asarray(tr["z"], float) for tr in d["traces"]])
            ax.set_box_aspect((np.ptp(xs) or 1, np.ptp(ys) or 1, max(np.ptp(zs), 0.15 * max(np.ptp(xs), 1))))
        except Exception:
            pass
        if any(tr["name"] for tr in d["traces"]):
            ax.legend(frameon=True, edgecolor="black", loc="upper left")
        ax.set_title("\n".join(textwrap.wrap(d["title"], 70)), fontweight="bold", loc="left")
        fig.tight_layout()
        return fig
    ax = fig.add_subplot(111)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    cats = None
    bar_traces = [tr for tr in d["traces"] if tr["t"] == "bar"]
    if bar_traces and any(isinstance(v, str) for v in bar_traces[0]["x"]):
        cats = []
        for tr in bar_traces:
            for v in tr["x"]:
                if v not in cats:
                    cats.append(v)
    nbar = max(len(bar_traces), 1)
    bi = 0
    for tr in d["traces"]:
        t = tr["t"]
        if t == "line":
            ax.plot(tr["x"], [np.nan if v is None else v for v in tr["y"]], MPL_DASH.get(tr["dash"], "-"),
                    color=tr["color"], lw=tr["width"], marker="o" if tr["markers"] else None, ms=6,
                    mec="black", mew=0.8, label=tr["name"])
            if tr["yerr"]:
                ax.errorbar(tr["x"], tr["y"], yerr=tr["yerr"], fmt="none", ecolor="black", capsize=3)
        elif t == "scatter":
            ax.scatter(tr["x"], [np.nan if v is None else v for v in tr["y"]], s=tr["size"] ** 2 * 0.9,
                       c=tr["color"], alpha=tr["opacity"], edgecolors="black", linewidths=0.7, label=tr["name"],
                       zorder=3)
            if tr["yerr"]:
                ax.errorbar(tr["x"], tr["y"], yerr=tr["yerr"], fmt="none", ecolor="black", capsize=3)
            if tr["text"]:
                for xx, yy, tt in zip(tr["x"], tr["y"], tr["text"]):
                    ax.annotate(str(tt), (xx, yy), textcoords="offset points", xytext=(0, 9), ha="center",
                                fontsize=10.5, color="black")
        elif t == "bar":
            yv = np.array([np.nan if v is None else v for v in tr["y"]], float)
            if cats is not None:
                pos = np.array([cats.index(v) for v in tr["x"]], float)
                w = 0.8 / nbar if d.get("barmode", "group") == "group" else 0.8
                off = (bi - (nbar - 1) / 2) * w if d.get("barmode", "group") == "group" else 0
            else:
                pos = np.asarray(tr["x"], float)
                gaps = np.diff(np.unique(pos))
                w = (gaps.min() if len(gaps) else 1.0) * 0.8 / (nbar if d.get("barmode", "group") == "group" else 1)
                off = (bi - (nbar - 1) / 2) * w if d.get("barmode", "group") == "group" else 0
            if tr.get("horizontal"):
                cont = ax.barh(pos + off, yv, height=w, color=tr["color"], edgecolor="black", linewidth=0.8,
                               label=tr["name"])
            else:
                cont = ax.bar(pos + off, yv, width=w, color=tr["color"], edgecolor="black", linewidth=0.8,
                              label=tr["name"], yerr=tr.get("yerr"), capsize=3 if tr.get("yerr") else 0,
                              error_kw=dict(ecolor="black"))
            if tr.get("labels"):
                fmt = tr.get("fmt", "{:.2f}")
                ax.bar_label(cont, labels=[fmt.format(v) if np.isfinite(v) else "" for v in yv], padding=3,
                             fontsize=11, color="black")
            bi += 1
        elif t == "hist":
            multi = sum(tr_["t"] == "hist" for tr_ in d["traces"]) > 1
            ax.bar(tr["x"], tr["y"], width=tr["w"], color=tr["color"], edgecolor="black", linewidth=0.6,
                   alpha=0.65 if multi else 0.95, label=tr["name"])
        elif t == "band":
            ax.fill_between(tr["x"], tr["ylo"], tr["yhi"], color=tr["color"], alpha=tr["alpha"], lw=0, label=tr["name"])
        elif t == "heatmap":
            sc = LIGHT_SCALES.get(tr["scale"], LIGHT_SCALES["Blues"])
            cmap = LinearSegmentedColormap.from_list("light", sc)
            Z = np.array([[np.nan if v is None else v for v in row] for row in tr["z"]], float)
            im = ax.imshow(Z, cmap=cmap, aspect="auto", origin="lower")
            ax.set_xticks(range(len(tr["x"])), [str(v) for v in tr["x"]])
            ax.set_yticks(range(len(tr["y"])), [str(v) for v in tr["y"]])
            if tr["annotate"]:
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        if np.isfinite(Z[i, j]):
                            ax.text(j, i, tr["fmt"].format(Z[i, j]), ha="center", va="center", color="black",
                                    fontsize=12)
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(tr["zlabel"])
            ax.grid(False)
    if cats is not None:
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=30 if max(len(str(c)) for c in cats) > 8 else 0,
                           ha="right" if max(len(str(c)) for c in cats) > 8 else "center")
    kv = 0
    for ln in d["lines"]:
        if ln["o"] == "v":
            ax.axvline(ln["v"], color=ln["color"], ls=MPL_DASH.get(ln["dash"], "--"), lw=2)
            if ln["label"]:
                kv += 1
                ax.annotate(ln["label"], (ln["v"], 1), xycoords=("data", "axes fraction"), xytext=(4, -14 - 16 * (kv - 1)),
                            textcoords="offset points", fontsize=11, color="black",
                            bbox=dict(fc="white", ec="none", alpha=0.85))
        else:
            ax.axhline(ln["v"], color=ln["color"], ls=MPL_DASH.get(ln["dash"], "--"), lw=2)
            if ln["label"]:
                ax.annotate(ln["label"], (0, ln["v"]), xycoords=("axes fraction", "data"), xytext=(6, 4),
                            textcoords="offset points", fontsize=11, color="black",
                            bbox=dict(fc="white", ec="none", alpha=0.85))
    if d["logx"]:
        ax.set_xscale("log")
    if d["logy"]:
        ax.set_yscale("log")
    if d.get("xrange"):
        ax.set_xlim(*d["xrange"])
    if d.get("yrange"):
        ax.set_ylim(*d["yrange"])
    elif any(tr.get("labels") for tr in bar_traces) and not d["logy"]:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0), hi * 1.12)
    if d.get("xreverse"):
        ax.invert_xaxis()
    if d.get("hide_yticks"):
        ax.set_yticks([])
    ax.set_xlabel(d["xlabel"])
    ax.set_ylabel(d["ylabel"])
    ax.tick_params(direction="out", length=6, width=1.2)
    ax.set_title("\n".join(textwrap.wrap(d["title"], 70)), fontweight="bold", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    if d.get("legend", True) and any(labels):
        ax.legend(frameon=True, edgecolor="black", framealpha=0.9, loc="best")
    fig.tight_layout()
    return fig


def export_bytes(f: Fig, fmt: str = "png", dpi: int = EXPORT_DPI) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = to_matplotlib(f, dpi=100)
    buf = io.BytesIO()
    fmt = fmt.lower()
    try:
        if fmt in ("png", "tiff", "tif"):
            fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
            if fmt != "png":
                from PIL import Image
                buf.seek(0)
                im = Image.open(buf).convert("RGB")
                out = io.BytesIO()
                try:
                    im.save(out, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
                except Exception:
                    out = io.BytesIO()
                    im.save(out, format="TIFF", dpi=(dpi, dpi))
                return out.getvalue()
            # embed dpi metadata in PNG
            from PIL import Image
            buf.seek(0)
            im = Image.open(buf)
            out = io.BytesIO()
            im.save(out, format="PNG", dpi=(dpi, dpi))
            return out.getvalue()
        fig.savefig(buf, format=fmt, facecolor="white")
        return buf.getvalue()
    finally:
        plt.close(fig)


MIME = {"png": "image/png", "tiff": "image/tiff", "svg": "image/svg+xml", "pdf": "application/pdf"}
