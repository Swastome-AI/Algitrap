"""Minimal stand-ins for `streamlit` and `plotly` so the whole app script can be
executed headlessly (smoke test). Widgets return their defaults unless a value
is supplied in CONTROL['values'] or CONTROL['buttons']."""
from __future__ import annotations

import sys
import types

CONTROL = {"values": {}, "buttons": set(), "errors": [], "calls": [], "uploads": {}}


class SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(ST, name)


class _Progress:
    def progress(self, *a, **k):
        v = a[0] if a else k.get("value", 0)
        assert 0 <= v <= 1.0001, f"progress out of range {v}"
        return self

    def empty(self):
        pass


def _widget(kind):
    def f(label="", *args, key=None, **kw):
        CONTROL["calls"].append((kind, key, label))
        ss = ST.session_state
        if key is not None and key in CONTROL["values"]:
            ss[key] = CONTROL["values"][key]
        if key is not None and key in ss:
            return ss[key]
        if kind == "selectbox" or kind == "radio":
            opts = list(args[0]) if args else list(kw.get("options", []))
            idx = kw.get("index", 0)
            val = opts[idx] if opts else None
        elif kind == "multiselect":
            val = list(kw.get("default", []) or [])
        elif kind in ("slider", "number_input"):
            if len(args) >= 3:
                val = args[2]
            elif "value" in kw:
                val = kw["value"]
            else:
                val = args[0] if args else 0
        elif kind in ("checkbox", "toggle"):
            val = args[0] if args else kw.get("value", False)
        elif kind in ("text_input", "text_area"):
            val = args[0] if args else kw.get("value", "")
        else:
            val = None
        if key is not None:
            ss[key] = val
        return val
    return f


def _button(label="", *a, key=None, **kw):
    CONTROL["calls"].append(("button", key, label))
    return key in CONTROL["buttons"] and not kw.get("disabled", False)


def _download(label, data, *a, key=None, **kw):
    assert data is not None and (isinstance(data, (bytes, str)) and len(data) > 0), f"empty download {label}"
    CONTROL["calls"].append(("download", key, label, len(data)))
    return False


def _uploader(label, *a, key=None, **kw):
    return CONTROL["uploads"].get(key)


def _error(msg, *a, **k):
    m = str(msg)
    if m.startswith("Something went wrong") or m.startswith("Could not"):
        CONTROL["errors"].append(m)
    else:
        CONTROL.setdefault("ui_errors", []).append(m)


def _exception(e, *a, **k):
    import traceback
    CONTROL["errors"].append("".join(traceback.format_exception(type(e), e, e.__traceback__)))


def _noop(*a, **k):
    return None


def _cache_data(*dargs, **dkw):
    if dargs and callable(dargs[0]):
        return dargs[0]

    def deco(fn):
        return fn
    return deco


def _columns(spec, *a, **k):
    n = spec if isinstance(spec, int) else len(spec)
    return [_Ctx() for _ in range(n)]


def _tabs(labels):
    return [_Ctx() for _ in labels]


def _plotly_chart(fig, *a, **k):
    assert isinstance(fig, Figure), "plotly_chart expects a Figure"
    CONTROL["calls"].append(("plotly", k.get("key")))


ST = types.ModuleType("streamlit")
ST.__version__ = "1.51.0"
ST.session_state = SessionState()
for name in ("selectbox", "radio", "multiselect", "slider", "number_input", "checkbox", "toggle", "text_input",
             "text_area"):
    setattr(ST, name, _widget(name))
ST.button = _button
ST.download_button = _download
ST.file_uploader = _uploader
ST.error = _error
ST.exception = _exception
for name in ("set_page_config", "markdown", "info", "success", "warning", "caption", "code", "write", "subheader",
             "divider", "image", "graphviz_chart", "toast", "metric", "dataframe", "header", "title"):
    setattr(ST, name, _noop)
ST.columns = _columns
ST.tabs = _tabs
ST.expander = lambda *a, **k: _Ctx()
ST.popover = lambda *a, **k: _Ctx()
ST.spinner = lambda *a, **k: _Ctx()
ST.container = lambda *a, **k: _Ctx()
ST.progress = lambda *a, **k: _Progress()
ST.cache_data = _cache_data
ST.cache_resource = _cache_data
ST.plotly_chart = _plotly_chart
ST.rerun = _noop
comp = types.ModuleType("streamlit.components")
compv1 = types.ModuleType("streamlit.components.v1")


def _html(html, height=None, **k):
    assert "$3Dmol" in html or "<" in html
    CONTROL["calls"].append(("html", len(html)))


compv1.html = _html
comp.v1 = compv1
ST.components = comp


# ------------------------------------------------------------------ plotly
class _Obj:
    def __init__(self, *a, **k):
        self.__dict__.update(k)
        self._args = a


class Figure(_Obj):
    def __init__(self, data=None, frames=None, **k):
        super().__init__(**k)
        self.data = list(data or [])
        self.frames = list(frames or [])

    def add_trace(self, t, **k):
        self.data.append(t)
        return self

    def update_layout(self, *a, **k):
        return self

    def update_xaxes(self, *a, **k):
        return self

    def update_yaxes(self, *a, **k):
        return self

    def add_vline(self, *a, **k):
        return self

    def add_hline(self, *a, **k):
        return self

    def to_html(self, *a, **k):
        return "<div>plotly</div>"


go = types.ModuleType("plotly.graph_objects")
go.Figure = Figure
for nm in ("Scatter", "Bar", "Heatmap", "Scatter3d", "Frame"):
    setattr(go, nm, type(nm, (_Obj,), {}))
plotly = types.ModuleType("plotly")
plotly.graph_objects = go
offline = types.ModuleType("plotly.offline")
offline.get_plotlyjs = lambda: "/* plotly.js */"
plotly.offline = offline


def install():
    sys.modules["streamlit"] = ST
    sys.modules["streamlit.components"] = comp
    sys.modules["streamlit.components.v1"] = compv1
    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = go
    sys.modules["plotly.offline"] = offline
