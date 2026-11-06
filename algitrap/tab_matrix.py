"""Tab 4 – Hydrogel matrix (mesh + Ca²⁺ crosslinking)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import state as S
from .anim import residue_centroids
from .constants import COLOR_CA, EGG_BOX_REPEAT, PALETTE
from .figs import Fig
from .network import NetworkParams, build_network, ca_nn_distances, network_stats, place_particle
from .ui import dict_table, fig_block, interpretation, mol_viewer, style_sets, summary_box, table_block
from .molio import Molecule

TAB = "Tab 4 – Hydrogel matrix"


def params_from_widgets() -> NetworkParams:
    P = S.P()
    ss = st.session_state
    return NetworkParams(
        topology=ss.get("net_topology", "Woven mesh"), grid=int(ss.get(S.k("net_grid"), P["grid"])),
        spacing=float(ss.get(S.k("net_spacing"), P["spacing"])), layers=int(ss.get(S.k("net_layers"), P["layers"])),
        gap=float(ss.get("net_gap", 4.5)), ca_spacing=float(ss.get("net_caspacing", EGG_BOX_REPEAT)),
        max_bridge=float(ss.get("net_maxbridge", 8.0)), g_only=bool(ss.get("net_gonly", False)),
        g_weight=float(ss.get("net_gweight", 2.0)), facing_deg=float(ss.get("net_facing", 80.0)))


def run_build(chain) -> None:
    p = params_from_widgets()
    S.invalidate("net")
    st.session_state["net"] = build_network(chain, p)


def render():
    summary_box("What happens in this tab",
                "The single chain is copied into a <b>grid</b>. In the <b>woven mesh</b>, layers of parallel chains "
                "alternate in x and y like a fishing net and the gaps between them are the pores. In the <b>egg-box "
                "bundle</b>, N × N parallel chains face each other in pairs (the classic junction zone). Ca²⁺ ions are "
                "then injected wherever two carboxylates from <i>different</i> chains face each other within bridging "
                "distance, preferring G-units and keeping a minimum Ca–Ca spacing. This is the rigid, un-relaxed "
                "starting structure for Tab 5.")
    chain = st.session_state.get("chain")
    if chain is None:
        st.info("Build or upload a chain first (Tab 1 or Tab 3).")
        return
    P = S.P()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.selectbox("Network topology", ["Woven mesh", "Egg-box bundle"], key="net_topology")
        st.slider("Grid size N (chains per direction)", 2, 10 if P["short"] == "q" else 16, P["grid"], key=S.k("net_grid"))
        st.slider("Chain spacing (Å, centre-to-centre)", 8.0, 40.0, P["spacing"], 0.5, key=S.k("net_spacing"),
                  help="Controls the pore size of the woven mesh (sheet spacing for the bundle).")
    with c2:
        st.slider("Layers (woven mesh)", 2, 4 if P["short"] == "q" else 8, P["layers"], key=S.k("net_layers"))
        st.slider("Calcium spacing – minimum Ca–Ca distance (Å)", 4.0, 20.0, EGG_BOX_REPEAT, 0.1, key="net_caspacing",
                  help="8.7 Å = one Ca²⁺ per egg-box cavity along a G-block (fibre repeat).")
        st.slider("Junction gap (Å, carboxylate to carboxylate)", 3.0, 9.0, 4.5, 0.1, key="net_gap")
    with c3:
        st.slider("Max bridging distance (Å)", 5.0, 12.0, 8.0, 0.5, key="net_maxbridge")
        st.slider("G preference factor", 1.0, 5.0, 2.0, 0.5, key="net_gweight",
                  help="How strongly Ca²⁺ prefers G-carboxylates when choosing sites.")
        st.checkbox("Strict egg-box: Ca²⁺ only between two G-units", False, key="net_gonly")
    p = params_from_widgets()
    if p.topology.startswith("Woven"):
        st.caption(f"Total chains ≈ {p.grid} lines × {p.layers} layers (each line may be tiled with several chain copies).")
    else:
        st.caption(f"Total chains arrayed = {p.grid} × {p.grid} = {p.grid ** 2}.")
    if st.button("🕸️ Build crosslinked network", type="primary", key="btn_build_net"):
        with st.spinner("Placing chains and injecting Ca²⁺…"):
            run_build(chain)
    net = st.session_state.get("net")
    if net is None:
        st.info("Press **Build crosslinked network**.")
        return
    stats = network_stats(net)
    for n_ in net.notes:
        st.caption("ℹ️ " + n_)
    m = st.columns(6)
    m[0].metric("Chains", stats["Chains"])
    m[1].metric("Atoms", f"{stats['Atoms (incl. Ca²⁺)']:,}")
    m[2].metric("Ca²⁺ ions", stats["Ca²⁺ ions"])
    m[3].metric("R = Ca²⁺/G", stats["R = Ca²⁺ / G (molar)"])
    m[4].metric("G–G junctions", stats["G–G junctions"])
    m[5].metric("Mean Ca–Ca NN", f"{stats['Mean Ca–Ca nearest-neighbour (Å)']} Å")
    show_mp = st.checkbox("Show the microplastic hovering above the mesh", True, key="net_show_mp")
    pdb = S.network_pdb(net)
    static = None
    if show_mp and st.session_state.get("mp"):
        mp = st.session_state["mp"]
        pm = place_particle(net, mp["mol"], mp["size"].get("_best_dir"))
        from .molio import write_pdb
        static = write_pdb(pm, conect=False)
    big = net.mol.n > 15000
    mol_viewer(pdb, style_sets("rainbow", net.n_chains, stick_r=0.14, ca_r=1.0, lines=big), height=520,
               static_pdb=static, legend="rainbow = chains · sky-blue = Ca²⁺ · pink = microplastic")
    st.download_button(f"⬇️ {net.mol.title}.pdb", pdb, file_name=f"{net.mol.title}.pdb", mime="chemical/x-pdb",
                       key="dl_net_pdb")

    # ------------------------------------------------------------ figures
    X = net.mol.coords
    cen, _ = residue_centroids(X, net.atom_res, net.mol.elements != "H")
    res_chain = np.array([r["chain"] for r in net.residues])
    layer = np.round((cen[:, 2] - cen[:, 2].min()) / max(net.layer_sep, 1)).astype(int)
    f = Fig("t4_top_view", "Top view of the network (x–y projection)", "x (Å)", "y (Å)", height=560)
    for L in np.unique(layer):
        sel = layer == L
        f.scatter(cen[sel, 0], cen[sel, 1], name=f"Layer {L + 1} residues" if p.topology.startswith("Woven")
                  else "Residue centres", color=PALETTE[int(L) % 5], size=6, opacity=0.8)
    cax = X[net.mol.elements == "CA"]
    if len(cax):
        f.scatter(cax[:, 0], cax[:, 1], name="Ca²⁺ crosslinks", color=COLOR_CA, size=12, symbol="diamond")
    fig_block(f, TAB, "Openings between the chains are the pores seen by a particle travelling along z.")
    c1, c2 = st.columns(2)
    with c1:
        nn = ca_nn_distances(X, net.mol.elements)
        f2 = Fig("t4_ca_nn_hist", "Ca–Ca nearest-neighbour distances (node spacing)", "Distance (Å)", "Number of Ca²⁺")
        if len(nn):
            f2.hist(nn, bins=20, name=None, color=COLOR_CA)
            f2.vline(float(np.mean(nn)), f"mean {np.mean(nn):.1f} Å")
        f2.vline(net.params.ca_spacing, "minimum allowed", color="#D55E00", dash="dot")
        fig_block(f2, TAB, "The node-based (original) pore estimate: distances between crosslink nodes.")
    with c2:
        jt = pd.Series([r["Junction type"] for r in net.ca]).value_counts() if net.ca else pd.Series(dtype=int)
        f3 = Fig("t4_junction_types", "Which units are bridged by Ca²⁺?", "Junction type", "Number of Ca²⁺",
                 categorical_x=True)
        cats = ["GG", "GM", "MM"]
        f3.bar(["G–G", "G–M", "M–M"], [int(jt.get(c, 0)) for c in cats], color=["#E69F00", "#009E73", "#0072B2"],
               fmt="{:.0f}")
        fig_block(f3, TAB, "Egg-box theory expects G–G junctions to dominate.")
    per_chain = np.zeros(net.n_chains, int)
    for r in net.ca:
        per_chain[r["Chain A"] - 1] += 1
        per_chain[r["Chain B"] - 1] += 1
    vals, cnts = np.unique(per_chain, return_counts=True)
    f4 = Fig("t4_crosslinks_per_chain", "Crosslinks per chain (network connectivity)", "Ca²⁺ crosslinks on the chain",
             "Number of chains", categorical_x=True)
    f4.bar([str(v) for v in vals], cnts.tolist(), color="#0072B2", fmt="{:.0f}")
    fig_block(f4, TAB, "Chains with 0 crosslinks are dangling; ≥ 2 crosslinks make a chain load-bearing.")
    table_block(dict_table(stats), "t4_network_statistics", TAB, "Network statistics")
    if net.ca:
        jdf = pd.DataFrame([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")}
                            for r in net.ca])
        table_block(jdf, "t4_ca_junctions", TAB, "Every Ca²⁺ crosslink (coordinates in Å)", height=260)
    table_block(pd.DataFrame(net.params.as_table()).astype(str), "t4_parameters", TAB, "Parameters used")
    interpretation(
        "<b>R = Ca²⁺/G</b> is the classic stoichiometric ratio: below ~0.1 Ca²⁺ only binds single chains, 0.1–0.25 "
        "forms egg-box <i>dimers</i>, above 0.25 dimers associate laterally. <b>Charge neutralisation</b> tells how many "
        "carboxylates are paired with Ca²⁺ (100 % = one Ca²⁺ per two residues). A high share of <b>G–G junctions</b> "
        "means the model follows egg-box chemistry. The structure is still <i>rigid</i>: chains have not yet bent "
        "around the ions – that happens in Tab 5.")
