"""Tab 9 – Report, overall summary and downloads."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from . import state as S
from .constants import APP_NAME, APP_VERSION
from .figs import Fig
from .forcefield import ca_coordination
from .molio import write_pdb, write_sdf
from .network import network_stats
from .pipeline import egg_box_score
from .report import build_html, build_pdf, build_zip, safe_name
from .ui import show_df, show_plotly, box, cached_export, interpretation, stretch, summary_box


def _exp(f, fmt, dpi=300):
    return cached_export(f.to_json(), fmt, int(dpi))

TAB = "Tab 9 – Report"

TAB_TEXT = {
    "Tab 1 – Upload & format check": (
        "Input structures were parsed and checked for format, geometry and chemistry. Alginate residues were recognised "
        "from their rings and M/G units assigned from the C5 configuration; the microplastic was identified and sized.",
        "PASS rows mean the inputs are usable; the minimum cross-section of the microplastic is the size used for sieving."),
    "Tab 2 – The science": (
        "Background: alginate chemistry, the egg-box model and the microplastic problem, with the relevant size scales.",
        "Molecular pores are nanometres wide while microplastics are micrometres to millimetres."),
    "Tab 3 – Molecular blueprint": (
        "A 3-D alginate chain was generated and characterised (sequence statistics, dimensions, glycosidic torsions).",
        "F_G and the length of G-blocks control how many egg-box junctions can form."),
    "Tab 4 – Hydrogel matrix": (
        "The chain was replicated into a network and Ca²⁺ ions were placed between facing carboxylates of different chains.",
        "R = Ca²⁺/G and the share of G–G junctions show how egg-box-like the crosslinking is."),
    "Tab 5 – Thermodynamics": (
        "The network was relaxed by energy minimisation (and optionally Langevin dynamics) so chains wrap around Ca²⁺.",
        "Falling energy, higher Ca–O coordination and a higher egg-box score show junction formation."),
    "Tab 6 – Capture prediction": (
        "A distance field gave the pore-limiting diameter (PLD), pore-size distribution and capture efficiency of the gel.",
        "A particle larger than the PLD cannot cross the network (size-exclusion capture)."),
    "Tab 7 – Wet-lab synthesis": (
        "Bench recipe, bead size, cure time and theoretical mesh size for making the hydrogel in the laboratory.",
        "Bead gaps (~15 % of bead size) – not the gel mesh – limit capture in loose bead beds."),
    "Tab 8 – UV photodegradation": (
        "UV-C dose and kinetics were modelled and laboratory data (kinetics, FTIR, microscopy) analysed.",
        "Rising carbonyl index, fragmentation and falling molar mass together demonstrate photodegradation."),
}

LIMITATIONS = [
    "The hydrogel network is an idealised, regular model (woven mesh or bundle) built from one chain type; real gels are "
    "random, polydisperse and contain junction zones of varying length.",
    "AlgiTrap-ENM is an implicit-solvent elastic-network model with a Morse Ca–O term; it captures egg-box geometry "
    "qualitatively but is not a substitute for explicit-solvent molecular dynamics (e.g. GLYCAM/CHARMM36 with water).",
    "Pore sizes are geometric (vdW surfaces, ionic radius for Ca²⁺) at the chosen grid resolution; hydration shells and "
    "chain mobility make real effective pores somewhat smaller and dynamic.",
    "Microplastic molecules in .sdf files are short oligomers (nanoplastic fragments). Real microplastic particles are "
    "10³–10⁶ times larger; the multiscale table bridges this with mesh-size theory, which has its own uncertainty "
    "(model estimates span ~10×).",
    "UV kinetic defaults are illustrative planning values. Conclusions about degradation must come from measured data "
    "(FTIR, mass/molar-mass loss, microscopy) with replicates and controls (dark control, gel without UV).",
    "Simulated demo data and images are clearly labelled and must not be presented as experimental results.",
]


def cover_metrics() -> list[tuple]:
    ss = st.session_state
    out = [("Prepared by", ss.get("user_name") or "–"), ("Preset", ss.get("preset"))]
    if ss.get("alg"):
        out.append(("Alginate input", ss["alg"]["name"]))
    if ss.get("mp"):
        mp = ss["mp"]
        out.append(("Microplastic input", f"{mp['name']} ({mp['ident']['label']})"))
        out.append(("Microplastic min. cross-section (Å)", mp["size"]["Min cross-section diameter (Å)"]))
    ch = ss.get("chain")
    if ch:
        out += [("Chain sequence", ch.sequence if len(ch.sequence) <= 40 else ch.sequence[:37] + "…"),
                ("G fraction F_G", round(ch.sequence.count("G") / len(ch.sequence), 3))]
    net = ss.get("net")
    if net:
        st_ = network_stats(net)
        out += [("Network", f"{st_['Topology']}, {st_['Chains']} chains, {st_['Atoms (incl. Ca²⁺)']} atoms"),
                ("Ca²⁺ ions / R = Ca²⁺:G", f"{st_['Ca²⁺ ions']} / {st_['R = Ca²⁺ / G (molar)']}")]
    rel = ss.get("relax")
    if rel is not None and net is not None:
        h = pd.DataFrame(rel.history)
        out.append(("Energy change on relaxation", f"{h['Total energy'].iloc[-1] - h['Total energy'].iloc[0]:,.1f} {rel.units}"))
        out.append(("Egg-box score (relaxed)",
                    f"{100 * egg_box_score(ca_coordination(net.mol, rel.final, chain_of_atom=net.atom_chain)):.0f} %"))
    pr = ss.get("pores")
    if pr:
        b = pr.get("relaxed") or pr["rigid"]
        out.append(("Pore-limiting diameter PLD (Å)", round(b["pld"], 2)))
        out.append(("Median pore diameter d50 (Å)", b["summary"]["Median pore diameter d50 (Å)"]))
        if ss.get("particle_A"):
            d = ss["particle_A"]
            out.append(("Particle diameter (Å)", round(d, 2)))
            out.append(("Capture verdict", "TRAPPED (size exclusion)" if d > b["pld"] else "CAN MIGRATE"))
    uvm = ss.get("uv_model")
    if uvm:
        out.append(("UV model k (1/h)", round(uvm["k"], 4)))
    return out


def findings() -> list[str]:
    ss = st.session_state
    f = []
    ch = ss.get("chain")
    if ch:
        nG = ch.sequence.count("G")
        gg = sum(1 for i in range(len(ch.sequence) - 1) if ch.sequence[i:i + 2] == "GG")
        f.append(f"The alginate chain has {len(ch.sequence)} residues ({nG} G, {len(ch.sequence) - nG} M; F_G = "
                 f"{nG / len(ch.sequence):.2f}) with {gg} G–G dyads available for egg-box junctions.")
    net = ss.get("net")
    if net:
        s = network_stats(net)
        f.append(f"The {s['Topology'].lower()} contains {s['Chains']} chains crosslinked by {s['Ca²⁺ ions']} Ca²⁺ ions "
                 f"(R = {s['R = Ca²⁺ / G (molar)']}; {s['G–G junctions']} G–G, {s['G–M junctions']} G–M, "
                 f"{s['M–M junctions']} M–M junctions).")
    rel = ss.get("relax")
    if rel is not None and net is not None:
        c0 = ca_coordination(net.mol, net.mol.coords, chain_of_atom=net.atom_chain)
        c1 = ca_coordination(net.mol, rel.final, chain_of_atom=net.atom_chain)
        if c1:
            f.append(f"Energy minimisation ({rel.engine}, {rel.n_iter} iterations) raised the mean Ca–O coordination from "
                     f"{np.mean([c['O within cutoff'] for c in c0]):.2f} to {np.mean([c['O within cutoff'] for c in c1]):.2f} "
                     f"and the egg-box score from {100 * egg_box_score(c0):.0f} % to {100 * egg_box_score(c1):.0f} %.")
    pr = ss.get("pores")
    if pr:
        r0 = pr["rigid"]["pld"]
        if pr.get("relaxed"):
            r1 = pr["relaxed"]["pld"]
            f.append(f"The pore-limiting diameter changed from {r0:.2f} Å (rigid) to {r1:.2f} Å (relaxed), a "
                     f"{100 * (r1 - r0) / r0:+.1f} % change caused by egg-box formation.")
        else:
            f.append(f"The pore-limiting diameter of the rigid network is {r0:.2f} Å.")
        d = ss.get("particle_A")
        b = pr.get("relaxed") or pr["rigid"]
        if d:
            f.append(f"A particle of {d:,.2f} Å is predicted to be "
                     f"{'TRAPPED by size exclusion' if d > b['pld'] else 'able to MIGRATE through the network'} "
                     f"(d/PLD = {d / b['pld']:.2f}).")
    sw = ss.get("sweep")
    if sw is not None and ss.get("particle_A"):
        from .pipeline import critical_spacing
        cs = critical_spacing(sw, ss["particle_A"])
        if cs:
            f.append(f"Design rule: chain spacings below {cs:.1f} Å trap this particle.")
    fl_ = ss.get("uv_demo_flags", {})
    tag = lambda key: " [SIMULATED DEMO DATA – not an experimental result]" if fl_.get(key) else ""  # noqa: E731
    fits = ss.get("uv_fits")
    if fits is not None and len(fits) and "k (1/h)" in fits:
        best = fits.sort_values("k (1/h)", ascending=False).iloc[0]
        f.append(f"Fastest fitted degradation: {best['Group']} with k = {best['k (1/h)']:.4f} 1/h"
                 + (f" (half-life {best['Half-life (h)']:.0f} h)." if "Half-life (h)" in best and pd.notna(best['Half-life (h)']) else ".")
                 + tag("kinetics"))
    ft = ss.get("uv_ftir")
    if ft is not None and len(ft) >= 2:
        a, b2 = ft.iloc[0]["Carbonyl index (1850–1650 / ref)"], ft.iloc[-1]["Carbonyl index (1850–1650 / ref)"]
        f.append(f"FTIR carbonyl index changed from {a:.3f} to {b2:.3f} after UV-C exposure ({ft.iloc[-1]['Spectrum']})."
                 + tag("ftir"))
    im = ss.get("uv_images")
    if im is not None and "Fragmentation index (N/N_before)" in im:
        f.append(f"Microscopy fragmentation index = {im['Fragmentation index (N/N_before)'].iloc[-1]:.2f} "
                 f"(particle count after / before)." + tag("images"))
    if not f:
        f.append("Run the pipeline (Tab 1 → ⚡) to generate findings.")
    return f


def _sections() -> list[dict]:
    figs = st.session_state.get("figs", {})
    tabs = st.session_state.get("tables", {})
    out = []
    for tab, (summ, interp) in TAB_TEXT.items():
        fl = [Fig.from_json(v["json"]) for v in figs.values() if v["tab"] == tab]
        tl = [(v["caption"], v["df"]) for v in tabs.values() if v["tab"] == tab]
        if fl or tl:
            out.append({"title": tab, "summary": summ, "figures": fl, "tables": tl, "interpretation": interp})
    return out


def _structures() -> dict:
    ss = st.session_state
    out = {}
    if ss.get("alg"):
        out["input_" + ss["alg"]["name"]] = write_pdb(ss["alg"]["mol"])
    if ss.get("mp"):
        out["input_" + ss["mp"]["name"].rsplit(".", 1)[0] + ".sdf"] = write_sdf(ss["mp"]["mol"])
    ch = ss.get("chain")
    if ch:
        out["single_chain.pdb"] = write_pdb(ch.mol)
        out["single_chain.sdf"] = write_sdf(ch.mol)
    net = ss.get("net")
    if net:
        N = net.params.grid
        out[f"Crosslinked_Hydrogel_Net_{N}x{N}.pdb"] = S.network_pdb(net)
        rel = ss.get("relax")
        if rel is not None:
            out[f"Perfect_Hydrogel_{N}x{N}.pdb"] = S.network_pdb(net, rel.final)
            out[f"relaxation_trajectory_{N}x{N}.pdb"] = S.network_traj_pdb(net, rel.frames)[0]
        md = ss.get("md")
        if md is not None:
            out[f"langevin_trajectory_{N}x{N}.pdb"] = S.network_traj_pdb(net, md.frames)[0]
    if ss.get("gif"):
        out["egg_box_relaxation.gif"] = ss["gif"]
    return out


def _params() -> dict:
    ss = st.session_state
    p = {"app": f"{APP_NAME} {APP_VERSION}", "created": dt.datetime.now().isoformat(timespec="seconds"),
         "user": ss.get("user_name"), "preset": ss.get("preset")}
    if ss.get("net"):
        p["network"] = asdict(ss["net"].params)
    if ss.get("relax") is not None:
        r = ss["relax"]
        p["minimisation"] = {"engine": r.engine, "iterations": r.n_iter, "converged": r.converged, "message": r.message}
    try:
        from .tab_thermo import ff_from_widgets
        p["forcefield"] = asdict(ff_from_widgets())
    except Exception:
        pass
    if ss.get("pores"):
        p["pore_grid_spacing_A"] = ss["pores"]["h"]
    if ss.get("particle_A"):
        p["particle_diameter_A"] = ss["particle_A"]
    if ss.get("uv_model"):
        p["uv_model"] = ss["uv_model"]
    p["widgets"] = {k: v for k, v in ss.items() if isinstance(v, (int, float, str, bool)) and not k.startswith(("FormSubmitter", "dl_", "btn_"))}
    return p


def render():
    summary_box("What happens in this tab",
                "Everything you produced is collected here: an overall summary, auto-written key findings, and one-click "
                "exports – a landscape PDF report with your name, an interactive HTML report, a ZIP of all plots in "
                "publication formats, and a full ZIP with structures, trajectories, tables, figures and parameters.")
    name = (st.session_state.get("user_name") or "").strip()
    cm = cover_metrics()
    st.markdown("### 📊 Overall results")
    show = [(k, v) for k, v in cm if k not in ("Prepared by", "Preset", "Alginate input", "Microplastic input",
                                                 "Chain sequence")]
    for i in range(0, len(show), 4):
        cols = st.columns(4)
        for c, (k, v) in zip(cols, show[i:i + 4]):
            c.metric(k, v if isinstance(v, str) else (f"{v:,.4g}" if isinstance(v, float) else v))
    pr = st.session_state.get("pores")
    d = st.session_state.get("particle_A")
    if pr and d:
        b = pr.get("relaxed") or pr["rigid"]
        if d > b["pld"]:
            st.success(f"✅ Final prediction: a {d:,.2f} Å particle is **trapped** by the {('relaxed' if pr.get('relaxed') else 'rigid')} "
                       f"alginate network (PLD {b['pld']:.2f} Å).")
        else:
            st.error(f"⚠️ Final prediction: a {d:,.2f} Å particle **can migrate** through the network (PLD {b['pld']:.2f} Å).")
    st.markdown("### 📝 Key findings")
    fl = findings()
    for x in fl:
        st.markdown(f"- {x}")
    st.markdown("### 🧾 What the whole study shows")
    box("<b>Story of the project.</b> Alginate from brown seaweed carries a negative charge on every sugar unit. "
        "Calcium ions zip neighbouring chains together where buckled G-blocks face each other – the egg-box. AlgiTrap "
        "rebuilt this process atom by atom: it generated the chain with correct stereochemistry, wove it into a mesh, "
        "placed Ca²⁺ at facing carboxylates and let physics relax the structure. Measuring the empty space inside the "
        "relaxed gel gives a pore-limiting diameter that decides which particles can pass. Because real microplastics are "
        "far larger than molecular pores, an intact gel layer acts as an absolute barrier, and the weak points of a real "
        "filter are the spaces between beads – a design insight that guides the wet-lab prototype. Once trapped, the "
        "plastics sit in a water-rich, UV-transparent matrix where 254 nm light (boosted by H₂O₂) can oxidise and cut "
        "their chains; FTIR carbonyl growth and fragment counts are the evidence to collect.", "interp")

    figs = st.session_state.get("figs", {})
    tabs = st.session_state.get("tables", {})
    st.markdown(f"### 🗂️ Everything available for export ({len(figs)} figures, {len(tabs)} tables)")
    if figs:
        show_df(pd.DataFrame([{"Figure ID": k, "Tab": v["tab"], "Title": v["title"]} for k, v in figs.items()]),
                hide_index=True, height=240)
    st.markdown("### ⬇️ Exports")
    c1, c2 = st.columns(2)
    with c1:
        fmts = st.multiselect("Figure formats", ["png", "tiff", "svg", "pdf"], default=["png", "svg"], key="rep_fmts")
    with c2:
        st.selectbox("Raster resolution (dpi)", [300, 600], key="export_dpi")
    dpi = int(st.session_state.get("export_dpi", 300))
    if not name:
        st.warning("✍️ Enter **your name** at the top of the page – the PDF report button stays locked until you do.")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("📄 Create PDF report", type="primary", disabled=not name, key="btn_pdf"):
            bar = st.progress(0.0, text="Rendering figures…")
            st.session_state["pdf"] = build_pdf(name, _sections(), cm, fl, LIMITATIONS, dpi=200,
                                                progress=lambda f, t: bar.progress(f, text=t), exporter=_exp)
            bar.empty()
        if st.session_state.get("pdf") and name:
            st.download_button(f"⬇️ {APP_NAME}_Report_{safe_name(name)}.pdf", st.session_state["pdf"],
                               file_name=f"{APP_NAME}_Report_{safe_name(name)}.pdf", mime="application/pdf",
                               key="dl_pdf")
    with b2:
        if st.button("🌐 Create interactive HTML report", key="btn_html"):
            with st.spinner("Building HTML…"):
                st.session_state["html"] = build_html(name or "User", _sections(), cm, fl, LIMITATIONS)
        if st.session_state.get("html"):
            st.download_button("⬇️ HTML report", st.session_state["html"].encode("utf-8"),
                               file_name=f"{APP_NAME}_Report_{safe_name(name)}.html", mime="text/html", key="dl_html")
    with b3:
        if st.button("🖼️ ZIP of all plots", key="btn_zipfigs", disabled=not fmts or not figs):
            bar = st.progress(0.0, text="Exporting…")
            st.session_state["zipfigs"] = build_zip(name or "User", [Fig.from_json(v["json"]) for v in figs.values()],
                                                    fmts, {}, {}, {}, {"formats": fmts, "dpi": dpi}, dpi=dpi,
                                                    progress=lambda f, t: bar.progress(f, text=t), exporter=_exp)
            bar.empty()
        if st.session_state.get("zipfigs"):
            st.download_button("⬇️ Plots ZIP", st.session_state["zipfigs"],
                               file_name=f"{APP_NAME}_plots_{safe_name(name)}.zip", mime="application/zip", key="dl_zipf")
    with b4:
        if st.button("📦 Full analysis ZIP", key="btn_zipall", disabled=not figs):
            bar = st.progress(0.0, text="Collecting…")
            reports = {}
            if name:
                reports[f"{APP_NAME}_Report_{safe_name(name)}.pdf"] = st.session_state.get("pdf") or build_pdf(
                    name, _sections(), cm, fl, LIMITATIONS, dpi=200, exporter=_exp)
            reports[f"{APP_NAME}_Report_{safe_name(name)}.html"] = st.session_state.get("html") or build_html(
                name or "User", _sections(), cm, fl, LIMITATIONS)
            readme = "Key findings:\n" + "\n".join("- " + x for x in fl)
            st.session_state["zipall"] = build_zip(name or "User", [Fig.from_json(v["json"]) for v in figs.values()],
                                                   fmts or ["png"], {k: v["df"] for k, v in tabs.items()}, _structures(),
                                                   reports, _params(), readme_extra=readme, dpi=dpi,
                                                   progress=lambda f, t: bar.progress(f, text=t), exporter=_exp)
            bar.empty()
        if st.session_state.get("zipall"):
            st.download_button("⬇️ Full ZIP", st.session_state["zipall"],
                               file_name=f"{APP_NAME}_{safe_name(name)}_analysis.zip", mime="application/zip",
                               key="dl_zipall")
    st.markdown("### ⚠️ Limitations and good practice")
    for l_ in LIMITATIONS:
        st.markdown(f"- {l_}")
    interpretation(
        "Use the PDF for judges and teachers (it carries your name on every page), the HTML report to show interactive "
        "plots on a laptop, and the full ZIP as your lab notebook: structures open in PyMOL, ChimeraX or VMD, tables "
        "open in Excel, and <code>parameters.json</code> lets anyone reproduce the run exactly.",
        "How to use these outputs")
    st.caption(f"{APP_NAME} v{APP_VERSION} · Suggested citation: {APP_NAME} – Alginate Hydrogel Microplastic Capture & "
               "Photodegradation Studio, Swastome-AI (2026).")
