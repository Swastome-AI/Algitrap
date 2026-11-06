"""Tab 3 – Molecular blueprint (chain generator)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import state as S
from .alginate import (LINKAGE_DEFAULTS, analyze_alginate, build_chain, chain_metrics, generate_sequence,
                       linkage_table, ring_centroids, sequence_stats, standardize)
from .chem_rdkit import EXAMPLE_SMILES, molecule_to_smiles, rdkit_available, rdkit_version, smiles_to_molecule
from .constants import COLOR_G, COLOR_M, RISE_POLY_G, RISE_POLY_M
from .figs import Fig
from .molio import deprotonate_carboxylic_acids, write_pdb, write_sdf
from .ui import dict_table, fig_block, interpretation, mol_viewer, style_sets, summary_box, table_block

TAB = "Tab 3 – Molecular blueprint"


def _set_chain(chain, src):
    S.invalidate("chain")
    st.session_state["chain"] = chain
    st.session_state["chain_src"] = src


def render():
    summary_box("What happens in this tab",
                "A single alginate chain is generated in 3-D. Choose its <b>M/G sequence</b> (built-in fibre-model builder "
                "with correct stereochemistry), paste a <b>SMILES</b> string (RDKit ETKDG embedding + force-field "
                "clean-up), or keep the chain from your uploaded file. The tab then reports NMR-style sequence statistics "
                "(F<sub>G</sub>, F<sub>GG</sub>, N<sub>G&gt;1</sub>), chain dimensions and glycosidic torsion angles.")
    has_upload = bool(st.session_state.get("alg", {}) and st.session_state["alg"].get("ok"))
    sources = ["Uploaded file (Tab 1)", "M/G sequence builder (built-in)", "SMILES → 3-D (RDKit)"]
    src = st.radio("Chain source", sources, index=0 if has_upload else 1, horizontal=True, key="chain_source")
    P = S.P()
    if src.startswith("Uploaded") and not has_upload:
        st.info("No valid alginate file uploaded yet – go to Tab 1, or choose the sequence builder.")
    elif src.startswith("Uploaded"):
        a = st.session_state["alg"]
        st.success(f"Using `{a['name']}` – sequence **{a['chain'].sequence}**.")
        if st.button("Use this uploaded chain", key="btn_use_upload"):
            _set_chain(a["chain"], f"Uploaded: {a['name']}")
    elif src.startswith("M/G"):
        c1, c2, c3 = st.columns(3)
        with c1:
            pat = st.selectbox("Sequence pattern", ["G-block / M-block", "random (Markov)", "poly-G", "poly-M",
                                                    "alternating MG", "custom"], key="seq_pattern")
            dp = st.slider("Degree of polymerisation (residues)", 2, 60 if P["short"] == "q" else 120, P["dp"],
                           key=S.k("dp"))
        with c2:
            fg = st.slider("Target G fraction F_G", 0.0, 1.0, 0.67, 0.01, key="seq_fg")
            blk = st.slider("Mean G-block length", 1, 20, 4, key="seq_blk")
        with c3:
            seed = st.number_input("Random seed", 0, 99999, 7, key="seq_seed")
            custom = st.text_input("Custom sequence (M/G letters)", "GGGGMMGGGGMM", key="seq_custom")
            inc_h = st.checkbox("Include hydrogens (all-atom)", False, key="seq_h",
                                help="Hydrogens make the model ~2× larger; heavy-atom models are faster for meshes.")
        seq = generate_sequence(pat, dp, fg, blk, int(seed), custom)
        st.code(seq, language=None)
        with st.expander("⚙️ Advanced: glycosidic torsions φ/ψ (IUPAC, heavy-atom definition)"):
            st.caption("Defaults come from a clash-free search that reproduces fibre-diffraction 2-fold helices "
                       "(poly-M 5.17 Å, poly-G 4.35 Å per residue).")
            tors = {}
            cols = st.columns(4)
            for j, key in enumerate(("MM", "GG", "MG", "GM")):
                with cols[j]:
                    phi = st.number_input(f"{key} φ (°)", -180.0, 180.0, float(LINKAGE_DEFAULTS[key][0]), key=f"phi_{key}")
                    psi = st.number_input(f"{key} ψ (°)", -180.0, 180.0, float(LINKAGE_DEFAULTS[key][1]), key=f"psi_{key}")
                    tors[key] = (phi, psi)
            if st.button("Re-run the φ/ψ optimisation (≈3 s)", key="btn_reopt"):
                with st.spinner("Scanning 4 × 1296 linkage conformations…"):
                    st.session_state["reopt"] = linkage_table()
            if st.session_state.get("reopt"):
                st.write({k: tuple(round(x, 1) for x in v) for k, v in st.session_state["reopt"].items()})
        if st.button("🧬 Generate 3-D chain", type="primary", key="btn_build_chain"):
            with st.spinner("Building chain from residue templates…"):
                ch = build_chain(seq, include_h=inc_h, torsions=tors)
            _set_chain(ch, f"Builder: {seq}")
    else:
        if not rdkit_available():
            st.warning("RDKit is not installed in this environment, so SMILES embedding is disabled. "
                       "Install it with `pip install rdkit` (it is listed in requirements.txt) – the built-in builder "
                       "works without it.")
        default = EXAMPLE_SMILES["Alginate trimer G-G-M (stereo-free example)"]
        if rdkit_available() and st.session_state.get("chain") is not None:
            try:
                default = molecule_to_smiles(st.session_state["chain"].mol)
            except Exception:
                pass
        smi = st.text_area("Paste a SMILES string", default, height=110, key="smiles_in")
        c1, c2 = st.columns(2)
        seed = c1.number_input("ETKDG random seed", 0, 99999, 42, key="smi_seed")
        opt = c2.selectbox("Clean-up force field", ["MMFF94", "UFF", "none"], key="smi_opt")
        st.caption(f"RDKit: {rdkit_version()}. SMILES without stereo marks (@/@@) get random configurations – M/G are then "
                   "assigned from whatever 3-D geometry results.")
        if st.button("🧬 Fold SMILES into 3-D", type="primary", key="btn_smiles", disabled=not rdkit_available()):
            try:
                with st.spinner("Embedding with ETKDGv3…"):
                    mol, info = smiles_to_molecule(smi, int(seed), opt)
                    mol, _ = deprotonate_carboxylic_acids(mol)
                    ch = standardize(analyze_alginate(mol))
                ch.source = f"RDKit {info['engine']} + {info['optimiser']}"
                _set_chain(ch, f"SMILES ({info['formula']})")
                st.success(f"Embedded {info['formula']} · optimiser {info['optimiser']} · converged: {info['converged']}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not build an alginate chain from this SMILES: {e}")

    ch = st.session_state.get("chain")
    if ch is None:
        st.info("No chain yet – generate one above (or run the full pipeline in Tab 1).")
        return
    st.markdown(f"### Active chain · {st.session_state.get('chain_src', '')}")
    met = chain_metrics(ch)
    stats = sequence_stats(ch.sequence)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Residues", met["Residues (DP)"])
    m2.metric("G fraction F_G", f"{stats['F_G']:.2f}")
    m3.metric("GG dyads (egg-box sites)", stats["Egg-box capable GG dyads"])
    m4.metric("Contour length", f"{met['Contour length (Å)']:.1f} Å")
    m5.metric("Mw (Na salt)", f"{met['Mw, Na-salt (g/mol)']:,.0f} g/mol")
    v1, v2 = st.columns([1.4, 1])
    with v1:
        mol_viewer(write_pdb(ch.mol), style_sets("mg"), height=420, legend="blue = M · orange = G")
    with v2:
        table_block(dict_table(met), "t3_chain_metrics", TAB, "Chain dimensions and composition")
        st.download_button("⬇️ single_chain.pdb", write_pdb(ch.mol), file_name="single_chain.pdb",
                           mime="chemical/x-pdb", key="dl_chain_pdb")
        st.download_button("⬇️ single_chain.sdf", write_sdf(ch.mol), file_name="single_chain.sdf",
                           mime="chemical/x-mdl-sdfile", key="dl_chain_sdf")
        if rdkit_available():
            try:
                st.caption("Isomeric SMILES of this chain (stereochemistry read from 3-D):")
                st.code(molecule_to_smiles(ch.mol), language=None)
            except Exception:
                pass

    # ---------------------------------------------------------- figures
    n = len(ch.sequence)
    pos = np.arange(1, n + 1)
    f = Fig("t3_sequence_track", "Monomer sequence along the chain (non-reducing → reducing end)", "Residue position",
            "", barmode="overlay", height=300)
    isg = np.array([c == "G" for c in ch.sequence])
    f.bar(pos[~isg].tolist(), [1] * int((~isg).sum()), name="M (β-D-mannuronate)", color=COLOR_M, labels=False)
    f.bar(pos[isg].tolist(), [1] * int(isg.sum()), name="G (α-L-guluronate)", color=COLOR_G, labels=False)
    f.d["hide_yticks"] = True
    fig_block(f, TAB, "Orange stretches are G-blocks – the regions able to form egg-box junctions with Ca²⁺.")
    c1, c2 = st.columns(2)
    with c1:
        pairs = [ch.sequence[i:i + 2] for i in range(n - 1)]
        z = [[pairs.count("MM"), pairs.count("MG")], [pairs.count("GM"), pairs.count("GG")]]
        tot = max(sum(map(sum, z)), 1)
        f2 = Fig("t3_dyad_matrix", "Nearest-neighbour (dyad) frequencies", "Next residue (i + 1)", "Residue i")
        f2.heatmap([[v / tot for v in row] for row in z], ["M", "G"], ["M", "G"], zlabel="Fraction of dyads")
        fig_block(f2, TAB, "Diagonal = homo-blocks (MM, GG); off-diagonal = alternating junctions.")
    with c2:
        runs = stats["runs"]
        lens = sorted({l for _, l in runs})
        f3 = Fig("t3_block_lengths", "Block-length distribution", "Block length (residues)", "Number of blocks",
                 categorical_x=True)
        f3.bar([str(l) for l in lens], [sum(1 for c, l2 in runs if c == "G" and l2 == l) for l in lens], name="G-blocks",
               color=COLOR_G, fmt="{:.0f}")
        f3.bar([str(l) for l in lens], [sum(1 for c, l2 in runs if c == "M" and l2 == l) for l in lens], name="M-blocks",
               color=COLOR_M, fmt="{:.0f}")
        fig_block(f3, TAB, "Long G-blocks (≥ 4–8) form the strongest junction zones.")
    if ch.linkages:
        lk = pd.DataFrame(ch.linkages)
        c3, c4 = st.columns(2)
        with c3:
            f4 = Fig("t3_phi_psi", "Glycosidic torsions (φ, ψ) of every linkage", "φ = O5–C1–O4′–C4′ (°)",
                     "ψ = C1–O4′–C4′–C3′ (°)", xrange=[-180, 180], yrange=[-180, 180])
            for typ, col in (("MM", COLOR_M), ("GG", COLOR_G), ("MG", "#009E73"), ("GM", "#CC79A7")):
                sub = lk[lk["Type"] == typ]
                if len(sub):
                    f4.scatter(sub["φ (°)"], sub["ψ (°)"], name=typ, color=col, size=12)
            fig_block(f4, TAB, "Points of the same linkage type should cluster; scattered points indicate strain.")
        with c4:
            cen = ring_centroids(ch)
            step = np.linalg.norm(np.diff(cen, axis=0), axis=1)
            f5 = Fig("t3_residue_spacing", "Distance between consecutive ring centres", "Linkage number",
                     "Centre-to-centre distance (Å)")
            f5.line(list(range(1, len(step) + 1)), step.tolist(), name="This chain", color="#000000", markers=True)
            f5.hline(RISE_POLY_M, "poly-M fibre rise 5.17 Å", color=COLOR_M).hline(RISE_POLY_G, "poly-G fibre rise 4.35 Å",
                                                                                    color=COLOR_G)
            fig_block(f5, TAB, "Shorter steps in G-blocks = buckled egg-box geometry.")
        table_block(lk, "t3_linkages", TAB, "Glycosidic linkage geometry")
    srows = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items() if k != "runs"}
    table_block(dict_table(srows, "Descriptor", "Value"), "t3_sequence_statistics", TAB,
                "Sequence statistics (Grasdalen NMR notation)")
    interpretation(
        "<b>F<sub>G</sub></b> is the fraction of G-units; commercial alginates range from about 0.3 to 0.7. "
        "<b>F<sub>GG</sub></b> counts G–G neighbours, the dyads that can host Ca²⁺ in the egg-box. "
        "<b>N<sub>G&gt;1</sub></b> is the average length of G-blocks longer than one unit – gels made from alginates with "
        "N<sub>G&gt;1</sub> above ~8 are strong and porous; low values give soft, elastic gels. In the torsion plot, a "
        "tight cluster per linkage type means a regular helix; the spacing plot shows the shorter, buckled steps that "
        "make G-blocks good Ca²⁺ cradles.")
