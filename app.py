"""AlgiTrap – Alginate Hydrogel Microplastic Capture & Photodegradation Studio.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="AlgiTrap – Alginate Microplastic Capture Studio", page_icon="🌊", layout="wide",
                   initial_sidebar_state="collapsed")

from algitrap import state as S  # noqa: E402
from algitrap import (tab_blueprint, tab_capture, tab_input, tab_matrix, tab_report, tab_science,  # noqa: E402
                      tab_thermo, tab_uv, tab_wetlab)
from algitrap.alginate import build_chain  # noqa: E402
from algitrap.constants import APP_NAME, APP_TAGLINE, APP_VERSION  # noqa: E402
from algitrap.network import network_stats  # noqa: E402
from algitrap.pipeline import DEFAULT_SEQUENCE, demo_plastic_file  # noqa: E402
from algitrap.ui import chips, footer, inject_css  # noqa: E402

S.init()
inject_css()


def run_full_pipeline():
    """One click: inputs → chain → mesh → Ca²⁺ → minimisation → pores → capture."""
    bar = st.progress(0.0, text="Preparing inputs…")
    ss = st.session_state
    if not ss.get("mp"):
        n, b = demo_plastic_file("PS")
        tab_input.set_plastic(n, b, "Demo")
    if ss.get("chain") is None:
        if ss.get("alg") and ss["alg"].get("ok"):
            ss["chain"] = ss["alg"]["chain"]
            ss["chain_src"] = f"Uploaded: {ss['alg']['name']}"
        else:
            ss["chain"] = build_chain(DEFAULT_SEQUENCE, include_h=False)
            ss["chain_src"] = f"Builder: {DEFAULT_SEQUENCE}"
    bar.progress(0.1, text="Building crosslinked network…")
    tab_matrix.run_build(ss["chain"])
    bar.progress(0.2, text="Energy minimisation…")
    tab_thermo.run_relax(lambda f, t: bar.progress(0.2 + 0.4 * min(f, 1.0), text=t))
    bar.progress(0.6, text="Pore analysis…")
    tab_capture.run_pores(lambda f, t: bar.progress(0.6 + 0.3 * min(f, 1.0), text=t))
    rel = ss.get("relax")
    if rel is not None:
        from algitrap.pipeline import pore_dynamics
        bar.progress(0.92, text="Pore dynamics…")
        ss["dyn"] = pore_dynamics(ss["net"], rel.frames, rel.frame_labels, max_frames=S.P()["dyn_frames"])
    bar.progress(1.0, text="Done – explore the tabs!")
    st.toast("Pipeline finished – results are ready in Tabs 3–9.", icon="✅")


def status_strip():
    ss = st.session_state
    alg = ss.get("alg")
    mp = ss.get("mp")
    ch = ss.get("chain")
    net = ss.get("net")
    rel = ss.get("relax")
    pr = ss.get("pores")
    items = [
        ("Alginate file", f"{alg['name']} · {alg['mol'].n} atoms" if alg else "– (demo or builder)"),
        ("Microplastic file", f"{mp['ident']['label']} · {mp['mol'].n} atoms · ⌀min {mp['size']['Min cross-section diameter (Å)']} Å"
         if mp else "–"),
        ("Chain", f"DP {len(ch.sequence)} · F_G {ch.sequence.count('G') / len(ch.sequence):.2f}" if ch else "not built"),
    ]
    if net:
        s = network_stats(net)
        items.append(("Network", f"{s['Chains']} chains · {s['Ca²⁺ ions']} Ca²⁺"))
    else:
        items.append(("Network", "not built"))
    items.append(("Relaxation", f"{rel.n_iter} it · {'converged' if rel.converged else 'max-iter'}" if rel is not None
                  else "not run"))
    if pr:
        b = pr.get("relaxed") or pr["rigid"]
        d = ss.get("particle_A")
        verdict = ("✅ trapped" if d and d > b["pld"] else "⚠️ passes") if d else ""
        items.append(("Pores / capture", f"PLD {b['pld']:.2f} Å {verdict}"))
    else:
        items.append(("Pores / capture", "not analysed"))
    chips(items)


# --------------------------------------------------------------------- header
st.markdown(f"<div class='at-hero'><h1>🌊 {APP_NAME}</h1><p><b>{APP_TAGLINE}</b> · from seaweed polymer to trapped and "
            f"UV-degraded microplastics · v{APP_VERSION}</p></div>", unsafe_allow_html=True)
h1, h2, h3 = st.columns([1.3, 1.6, 1.1])
with h1:
    st.text_input("✍️ Your name (used on the PDF/HTML report and ZIP)", key="user_name", placeholder="e.g. Anjali Sharma")
with h2:
    st.radio("Simulation preset", list(S.PRESETS), horizontal=True, key="preset",
             help="Quick = small mesh and short runs (seconds). Research = larger systems, tighter grids, more sampling.")
with h3:
    if st.session_state.get("user_name"):
        st.markdown(f"<br>👋 Report will be prepared for **{st.session_state['user_name']}**", unsafe_allow_html=True)

tabs = st.tabs(["1️⃣ Upload & Check", "2️⃣ The Science", "3️⃣ Molecular Blueprint", "4️⃣ Hydrogel Matrix",
                "5️⃣ Thermodynamics", "6️⃣ Capture Prediction", "7️⃣ Wet-Lab Synthesis", "8️⃣ UV Photodegradation",
                "9️⃣ Report & Downloads"])
renderers = [lambda: tab_input.render(run_full_pipeline), tab_science.render, tab_blueprint.render, tab_matrix.render,
             tab_thermo.render, tab_capture.render, tab_wetlab.render, tab_uv.render, tab_report.render]
for t, fn in zip(tabs, renderers):
    with t:
        status_strip()
        try:
            fn()
        except Exception as e:  # noqa: BLE001 – keep the rest of the app usable
            st.error(f"Something went wrong in this tab: {type(e).__name__}: {e}")
            st.exception(e)
footer()
