"""Tab 1 – Upload & format check."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import state as S
from .figs import Fig
from .molio import write_pdb
from .pipeline import demo_alginate_file, demo_plastic_file, load_alginate, load_plastic
from .ui import dict_table, fig_block, interpretation, mol_viewer, style_sets, summary_box, table_block

TAB = "Tab 1 – Upload & format check"
ICON = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL", "INFO": "ℹ️ INFO"}


def _checks_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Status"] = d["Status"].map(lambda s: ICON.get(s, s))
    return d


def set_alginate(name: str, data: bytes, origin: str):
    res = load_alginate(name, data)
    st.session_state["alg"] = res
    st.session_state["alg_origin"] = origin
    if res["ok"]:
        S.invalidate("chain")
        st.session_state["chain"] = res["chain"]
        st.session_state["chain_src"] = f"{origin}: {name}"


def set_plastic(name: str, data: bytes, origin: str):
    res = load_plastic(name, data)
    st.session_state["mp"] = res
    st.session_state["mp_origin"] = origin
    S.invalidate("capture_curve")


def render(run_pipeline):
    summary_box("What happens in this tab",
                "Upload the two input structures. <b>(1)</b> an alginate chain as <code>.pdb</code> (or .sdf/.mol/.xyz) – "
                "it becomes the building block for the chain → mesh → Ca²⁺ pipeline; <b>(2)</b> a microplastic oligomer "
                "(polystyrene PS or polyethylene PE) as <code>.sdf</code> (or .mol/.pdb) – it is the particle we try to "
                "trap. Every file is parsed, chemically checked (rings, stereochemistry, bonds, clashes) and summarised. "
                "No files? Use the built-in demo structures – they are generated from first principles by AlgiTrap.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("① Alginate chain")
        up = st.file_uploader("Upload alginate structure (.pdb preferred)", type=["pdb", "ent", "sdf", "mol", "xyz"],
                              key="up_alg")
        if up is not None and st.session_state.get("alg_file_id") != (up.name, up.size):
            st.session_state["alg_file_id"] = (up.name, up.size)
            with st.spinner("Parsing and checking alginate structure…"):
                set_alginate(up.name, up.getvalue(), "Uploaded")
        if st.button("Use built-in demo alginate (12-mer G₄M₂G₄M₂)", key="btn_demo_alg"):
            n, b = demo_alginate_file()
            set_alginate(n, b, "Demo")
    with c2:
        st.subheader("② Microplastic oligomer")
        up2 = st.file_uploader("Upload microplastic structure (.sdf preferred)", type=["sdf", "mol", "pdb", "xyz"],
                               key="up_mp")
        if up2 is not None and st.session_state.get("mp_file_id") != (up2.name, up2.size):
            st.session_state["mp_file_id"] = (up2.name, up2.size)
            with st.spinner("Parsing and identifying microplastic…"):
                set_plastic(up2.name, up2.getvalue(), "Uploaded")
        b1, b2 = st.columns(2)
        if b1.button("Demo polystyrene (PS 6-mer)", key="btn_demo_ps"):
            n, b = demo_plastic_file("PS")
            set_plastic(n, b, "Demo")
        if b2.button("Demo polyethylene (PE 12-mer)", key="btn_demo_pe"):
            n, b = demo_plastic_file("PE", 6)
            set_plastic(n, b, "Demo")
        st.selectbox("Polymer type used downstream", ["Auto-detect", "PS", "PE", "PP", "Other"], key="mp_type_override",
                     help="Overrides the automatic identification (affects FTIR reference band and UV model).")

    st.divider()
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        if st.button("⚡ Run the full pipeline with default settings", type="primary", key="btn_full"):
            run_pipeline()
    with cc2:
        st.caption("Runs every computational step (chain → mesh → Ca²⁺ → energy minimisation → pores → capture) with the "
                   "current preset, using your uploads or the demo files. Each tab also has its own controls.")

    alg, mp = st.session_state.get("alg"), st.session_state.get("mp")
    if not alg and not mp:
        st.info("Upload files or click a demo button to see the format checks.")
    cols = st.columns(2)
    if alg:
        with cols[0]:
            st.markdown(f"#### Alginate · `{alg['name']}`")
            table_block(_checks_df(alg["checks"]), "t1_alginate_checks", TAB, "Format & chemistry checks – alginate")
            table_block(dict_table(alg["stats"]), "t1_alginate_stats", TAB, "File statistics – alginate")
            pdb = write_pdb(alg["chain"].mol, conect=True) if alg.get("chain") else write_pdb(alg["mol"])
            mol_viewer(pdb, style_sets("mg"), height=330, legend="blue = M · orange = G")
    if mp:
        with cols[1]:
            st.markdown(f"#### Microplastic · `{mp['name']}`")
            table_block(_checks_df(mp["checks"]), "t1_plastic_checks", TAB, "Format & chemistry checks – microplastic")
            stats = dict(mp["stats"])
            stats.update({k: v for k, v in mp["size"].items() if not k.startswith("_")})
            stats["Identified polymer"] = f"{mp['ident']['name']} ({mp['ident']['confidence']})"
            table_block(dict_table(stats), "t1_plastic_stats", TAB, "File statistics & size descriptors – microplastic")
            mol_viewer(write_pdb(mp["mol"]), style_sets("element"), height=330, legend="microplastic (C green, H white)")

    if alg or mp:
        st.markdown("### Structure quality plots")
        f = Fig("t1_element_composition", "Element composition of the uploaded structures", "Element",
                "Atom count", categorical_x=True)
        for lab, rec, col in (("Alginate", alg, "#0072B2"), ("Microplastic", mp, "#E69F00")):
            if rec:
                els, cnt = np.unique([e.title() for e in rec["mol"].elements], return_counts=True)
                f.bar(list(els), cnt.tolist(), name=lab, color=col, fmt="{:.0f}")
        fig_block(f, TAB, "Counts of each element. Alginate = C, H, O (+Na/Ca if present); PE/PS = C and H only.")
        f2 = Fig("t1_bond_lengths", "Heavy-atom bond-length distribution (geometry quality check)", "Bond length (Å)",
                 "Number of bonds")
        for lab, rec, col in (("Alginate", alg, "#0072B2"), ("Microplastic", mp, "#E69F00")):
            if rec and len(rec["mol"].bonds):
                m = rec["mol"]
                hb = (m.elements[m.bonds[:, 0]] != "H") & (m.elements[m.bonds[:, 1]] != "H")
                d = np.linalg.norm(m.coords[m.bonds[hb, 0]] - m.coords[m.bonds[hb, 1]], axis=1)
                f2.hist(d, bins=30, name=lab, color=col, rng=(1.1, 1.8))
        f2.vline(1.39, "aromatic C–C 1.39", color="#999999", dash="dot").vline(1.53, "C–C 1.53", color="#000000", dash="dot")
        fig_block(f2, TAB, "Peaks should sit near 1.25 (C=O), 1.39 (aromatic), 1.43 (C–O) and 1.53 Å (C–C).")

    with st.expander("📥 Download sample input files (to see the expected formats)"):
        n, b = demo_alginate_file()
        st.download_button("Sample alginate .pdb (12-mer)", b, file_name=n, mime="chemical/x-pdb", key="smp_alg")
        n, b = demo_plastic_file("PS")
        st.download_button("Sample polystyrene .sdf (6-mer)", b, file_name=n, mime="chemical/x-mdl-sdfile", key="smp_ps")
        n, b = demo_plastic_file("PE", 6)
        st.download_button("Sample polyethylene .sdf (12-mer)", b, file_name=n, mime="chemical/x-mdl-sdfile",
                           key="smp_pe")

    interpretation(
        "<b>✅ PASS</b> rows mean the file is ready. <b>⚠️ WARN</b> rows are usable but deserve a look (e.g. residue labels "
        "that disagree with the 3-D stereochemistry – AlgiTrap trusts the geometry). <b>❌ FAIL</b> rows stop the "
        "pipeline for that file. M and G units are assigned from the configuration at C5 (M = D-manno, C5-S; "
        "G = L-gulo, C5-R), so the sequence is correct even if residue names are missing. For the microplastic, the "
        "<i>minimum cross-section diameter</i> is the narrowest 'doorway' the molecule could squeeze through – it is the "
        "size used for the sieving prediction in Tab 6.")
