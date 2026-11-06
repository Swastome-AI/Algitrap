"""Tab 5 – Thermodynamics simulation (egg-box formation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import state as S
from .anim import relaxation_animation, trajectory_gif
from .constants import COLOR_CA, PALETTE
from .figs import Fig
from .forcefield import (FFParams, available_engines, ca_coordination, minimize, radial_distribution, run_external,
                         run_langevin)
from .network import network_with_coords
from .pipeline import egg_box_score
from .ui import show_df, show_plotly, fig_block, interpretation, mol_viewer, stretch, style_sets, summary_box, table_block

TAB = "Tab 5 – Thermodynamics"


def ff_from_widgets() -> FFParams:
    ss = st.session_state
    d = FFParams()
    for f_ in d.__dataclass_fields__:
        key = f"ff_{f_}"
        if key in ss:
            setattr(d, f_, type(getattr(d, f_))(ss[key]))
    return d


def run_relax(progress=None):
    net = st.session_state["net"]
    P = S.P()
    engine = st.session_state.get("ff_engine", "AlgiTrap-ENM (built-in)")
    iters = int(st.session_state.get(S.k("ff_iters"), P["iters"]))
    nfr = int(st.session_state.get(S.k("ff_frames"), P["frames"]))
    S.invalidate("relax")
    if engine.startswith("AlgiTrap"):
        res = minimize(net.mol, ff_from_widgets(), max_iter=iters, n_frames=nfr,
                       gtol=float(st.session_state.get("ff_gtol", 0.05)), progress=progress)
    else:
        try:
            res = run_external(engine, net.mol, max_iter=iters, n_frames=nfr, progress=progress)
            if res.mol is not None and res.mol.n != net.mol.n:
                # keep analysis on the heavy-atom network: drop added H from frames
                keep = np.arange(net.mol.n)
                res.frames = [f[keep] for f in res.frames]
                res.final = res.final[keep]
        except Exception as e:  # noqa: BLE001
            st.warning(f"{engine} failed ({e}). Falling back to the built-in AlgiTrap-ENM engine.")
            res = minimize(net.mol, ff_from_widgets(), max_iter=iters, n_frames=nfr, progress=progress)
    st.session_state["relax"] = res
    st.session_state["relaxed_net"] = network_with_coords(net, res.final)
    if st.session_state.get("md_on", False):
        res_md = run_langevin(net.mol, res.final, ff_from_widgets(), n_steps=int(st.session_state.get(S.k("md_steps"), P["md_steps"])),
                              dt_fs=float(st.session_state.get("md_dt", 1.0)),
                              temperature=float(st.session_state.get("md_T", 298.15)),
                              gamma_ps=float(st.session_state.get("md_gamma", 20.0)), progress=progress)
        st.session_state["md"] = res_md


def render():
    summary_box("What happens in this tab",
                "The rigid network is relaxed by <b>energy minimisation</b>. Chains are allowed to bend at their "
                "glycosidic links while bonds, angles and rings keep their shape; Ca²⁺ is attracted to carboxylate, "
                "hydroxyl and ring oxygens and screened electrostatics act between charges. As the energy falls, "
                "chains wrap around the ions and the <b>egg-box junctions</b> form. Optional Langevin dynamics then "
                "lets the gel 'breathe' at room temperature so pore sizes fluctuate.")
    net = st.session_state.get("net")
    if net is None:
        st.info("Build the network in Tab 4 first.")
        return
    P = S.P()
    eng = available_engines()
    opts = [k for k, v in eng.items() if v]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.selectbox("Force-field engine", opts, key="ff_engine",
                     help="AlgiTrap-ENM is fast and includes Ca²⁺–oxygen binding. OpenBabel/RDKit appear when installed.")
        missing = [k for k, v in eng.items() if not v]
        if missing:
            st.caption("Not installed: " + ", ".join(missing))
    with c2:
        st.number_input("Max iterations", 20, 20000, P["iters"], 50, key=S.k("ff_iters"))
        st.number_input("Animation frames", 4, 60, P["frames"], 1, key=S.k("ff_frames"))
    with c3:
        st.number_input("Convergence: RMS-gradient tolerance (kcal/mol/Å)", 0.001, 5.0, 0.05, 0.01, key="ff_gtol",
                        format="%.3f")
        st.checkbox("Also run Langevin dynamics (thermal breathing)", False, key="md_on")
    with st.expander("⚙️ Force-field parameters (research settings)"):
        d = FFParams()
        cols = st.columns(4)
        fields = [("k_bond", "Bond k", 10.0, 2000.0), ("k_angle", "1-3 k", 1.0, 500.0), ("k_ring14", "Ring 1-4 k", 0.0, 300.0),
                  ("k_torsion14", "Torsion 1-4 k", 0.0, 50.0), ("k_rep", "Repulsion k", 1.0, 200.0),
                  ("rep_scale", "Contact scale", 0.5, 1.0), ("ca_o_r0", "Ca–O r₀ (Å)", 2.0, 3.0),
                  ("depth_carboxylate", "Ca–O(COO⁻) depth", 0.0, 30.0), ("depth_other_o", "Ca–O(OH) depth", 0.0, 15.0),
                  ("morse_a", "Morse a (1/Å)", 0.5, 4.0), ("eps_r", "ε_r", 1.0, 100.0),
                  ("ionic_strength", "Ionic strength (M)", 0.0, 2.0)]
        for j, (f_, lab, lo, hi) in enumerate(fields):
            with cols[j % 4]:
                st.number_input(lab, lo, hi, float(getattr(d, f_)), key=f"ff_{f_}")
        st.checkbox("Screened electrostatics", True, key="ff_coulomb")
        st.caption(f"Debye length at the chosen ionic strength: {ff_from_widgets().debye_length():.1f} Å")
    with st.expander("🌡️ Langevin dynamics settings"):
        cols = st.columns(4)
        cols[0].number_input("Steps", 50, 50000, P["md_steps"], 50, key=S.k("md_steps"))
        cols[1].number_input("Time step (fs)", 0.2, 2.0, 1.0, 0.1, key="md_dt")
        cols[2].number_input("Temperature (K)", 250.0, 400.0, 298.15, 1.0, key="md_T")
        cols[3].number_input("Friction γ (1/ps)", 0.5, 100.0, 20.0, 0.5, key="md_gamma")
    if st.button("▶ Run Physics Simulation", type="primary", key="btn_relax"):
        bar = st.progress(0.0, text="Starting…")
        run_relax(lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
        bar.empty()
    res = st.session_state.get("relax")
    if res is None:
        st.info("Press **Run Physics Simulation** to relax the network.")
        return
    hist = pd.DataFrame(res.history)
    E0, E1 = float(hist["Total energy"].iloc[0]), float(hist["Total energy"].iloc[-1])
    co0 = ca_coordination(net.mol, net.mol.coords, chain_of_atom=net.atom_chain)
    co1 = ca_coordination(net.mol, res.final, chain_of_atom=net.atom_chain)
    sc0, sc1 = egg_box_score(co0), egg_box_score(co1)
    m = st.columns(6)
    m[0].metric("Engine", res.engine.split(" (")[0])
    m[1].metric("Iterations", res.n_iter)
    m[2].metric("ΔE", f"{E1 - E0:,.1f} {res.units}")
    m[3].metric("Converged", "yes" if res.converged else "no (max iter)")
    m[4].metric("Mean Ca–O coordination", f"{np.mean([c['O within cutoff'] for c in co1]):.2f}" if co1 else "–",
                delta=f"{np.mean([c['O within cutoff'] for c in co1]) - np.mean([c['O within cutoff'] for c in co0]):+.2f}"
                if co1 else None)
    m[5].metric("Egg-box score", f"{100 * sc1:.0f} %" if co1 else "–",
                delta=f"{100 * (sc1 - sc0):+.0f} pts" if co1 else None)
    rn = st.session_state["relaxed_net"]
    tab_a, tab_b, tab_c = st.tabs(["🎞️ Full-atom animation (3Dmol)", "🎞️ Coarse animation (Plotly)", "🧊 Final structure"])
    with tab_a:
        traj, nfr = S.network_traj_pdb(net, res.frames)
        mol_viewer(traj, style_sets("rainbow", net.n_chains, stick_r=0.13, lines=net.mol.n > 15000), height=540,
                   frames=True, legend=f"{nfr} frames · looping relaxation", interval=260)
    with tab_b:
        fig = relaxation_animation(res.frames, res.frame_labels, net.mol, net.atom_res, net.atom_chain)
        show_plotly(fig, "pl_relax_anim")
    with tab_c:
        pdb_final = S.network_pdb(rn)
        mol_viewer(pdb_final, style_sets("rainbow", net.n_chains, stick_r=0.14, lines=net.mol.n > 15000), height=520,
                   legend="relaxed hydrogel · rainbow chains · sky-blue Ca²⁺")
    N = net.params.grid
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button(f"⬇️ Perfect_Hydrogel_{N}x{N}.pdb", S.network_pdb(rn), file_name=f"Perfect_Hydrogel_{N}x{N}.pdb",
                       mime="chemical/x-pdb", key="dl_relaxed")
    d2.download_button("⬇️ Trajectory (multi-model PDB)", S.network_traj_pdb(net, res.frames)[0],
                       file_name=f"relaxation_trajectory_{N}x{N}.pdb", mime="chemical/x-pdb", key="dl_traj")
    if d3.button("🎬 Render animated GIF", key="btn_gif"):
        with st.spinner("Rendering frames…"):
            st.session_state["gif"] = trajectory_gif(res.frames, net.mol, net.atom_res, net.atom_chain, res.frame_labels)
    if st.session_state.get("gif"):
        d4.download_button("⬇️ relaxation.gif", st.session_state["gif"], file_name="egg_box_relaxation.gif",
                           mime="image/gif", key="dl_gif")
        st.image(st.session_state["gif"], caption="Egg-box relaxation (residue centres, rainbow chains, sky-blue Ca²⁺)")

    # ---------------------------------------------------------------- plots
    c1, c2 = st.columns(2)
    with c1:
        f = Fig("t5_energy_convergence", "Energy minimisation: total potential energy", "Iteration",
                f"Total energy ({res.units})")
        f.line(hist["Iteration"], hist["Total energy"], name=None, color="#000000", markers=len(hist) < 40)
        fig_block(f, TAB, "A curve that flattens = the structure has reached a (local) energy minimum.")
    with c2:
        if "RMS gradient" in hist:
            f = Fig("t5_rms_gradient", "Convergence: RMS force on the atoms", "Iteration",
                    "RMS gradient (kcal/mol/Å, log scale)", logy=True)
            f.line(hist["Iteration"], hist["RMS gradient"].clip(lower=1e-6), name=None, color="#0072B2")
            fig_block(f, TAB, "Each decade down means 10× smaller residual forces.")
    comp_cols = [c for c in hist.columns if c not in ("Iteration", "Total energy", "RMS gradient")]
    if comp_cols:
        f = Fig("t5_energy_components", "Energy components during minimisation", "Iteration", f"Energy ({res.units})")
        for j, c in enumerate(comp_cols):
            if np.abs(hist[c]).max() > 1e-6:
                f.line(hist["Iteration"], hist[c], name=c, color=PALETTE[j % len(PALETTE)])
        fig_block(f, TAB, "Ca–O binding becomes more negative as egg-boxes form; strain terms rise slightly in return.")
    c3, c4 = st.columns(2)
    with c3:
        if co1:
            cn0 = np.array([c["O within cutoff"] for c in co0])
            cn1 = np.array([c["O within cutoff"] for c in co1])
            vals = list(range(0, int(max(cn0.max(), cn1.max())) + 1))
            f = Fig("t5_ca_coordination", "Ca²⁺ coordination number before vs after relaxation",
                    "Oxygen atoms within 3.0 Å of Ca²⁺", "Number of Ca²⁺", categorical_x=True)
            f.bar([str(v) for v in vals], [int((cn0 == v).sum()) for v in vals], name="Rigid (start)", color="#999999",
                  fmt="{:.0f}")
            f.bar([str(v) for v in vals], [int((cn1 == v).sum()) for v in vals], name="Relaxed", color=COLOR_CA,
                  fmt="{:.0f}")
            fig_block(f, TAB, "Real egg-box Ca²⁺ is 7–9-coordinate (incl. water); the model counts alginate O only.")
    with c4:
        ca = np.flatnonzero(net.mol.elements == "CA")
        ox = np.flatnonzero(net.mol.elements == "O")
        if len(ca):
            r, g0 = radial_distribution(net.mol.coords, ca, ox)
            _, g1 = radial_distribution(res.final, ca, ox)
            f = Fig("t5_ca_o_pair_distribution", "Ca–O pair distribution function", "Ca–O distance (Å)",
                    "Pair density (per Ca²⁺, Å⁻³)")
            f.line(r, g0, name="Rigid (start)", color="#999999", dash="dash")
            f.line(r, g1, name="Relaxed", color=COLOR_CA)
            f.vline(2.40, "ideal Ca–O 2.40 Å", color="#009E73", dash="dot")
            fig_block(f, TAB, "A sharp first peak near 2.4 Å = oxygens have closed in around Ca²⁺.")
    # per-chain RMSD
    disp = np.linalg.norm(res.final - net.mol.coords, axis=1)
    rmsd = [float(np.sqrt((disp[net.atom_chain == c] ** 2).mean())) for c in range(net.n_chains)]
    f = Fig("t5_chain_rmsd", "How far each chain moved during relaxation", "RMSD from rigid start (Å)",
            "Number of chains")
    f.hist(rmsd, bins=min(20, max(5, net.n_chains // 2)), color="#0072B2")
    fig_block(f, TAB, "Chains that move more are the ones pulled into junctions.")
    emp = pd.DataFrame({"Component": ["Total energy"] + comp_cols,
                        "Start": [hist["Total energy"].iloc[0]] + [hist[c].iloc[0] for c in comp_cols],
                        "Final": [hist["Total energy"].iloc[-1]] + [hist[c].iloc[-1] for c in comp_cols]})
    emp["Change"] = emp["Final"] - emp["Start"]
    table_block(emp.round(3), "t5_energy_components", TAB, f"Energy terms ({res.units})")
    table_block(hist.round(4), "t5_minimisation_log", TAB, "Full minimisation log", height=240)
    if co1:
        cdf = pd.DataFrame(co1)
        cdf.insert(1, "O within cutoff (start)", [c["O within cutoff"] for c in co0])
        table_block(cdf, "t5_ca_coordination", TAB, "Coordination of every Ca²⁺ after relaxation", height=240)
    table_block(pd.DataFrame(ff_from_widgets().as_table()).astype(str), "t5_forcefield_parameters", TAB,
                "Force-field parameters used")
    md = st.session_state.get("md")
    if md is not None:
        st.markdown("### Langevin dynamics – thermal breathing")
        mh = pd.DataFrame(md.history)
        c5, c6 = st.columns(2)
        with c5:
            f = Fig("t5_md_temperature", "Langevin dynamics: instantaneous temperature", "Time (ps)", "Temperature (K)")
            f.line(mh["Time (ps)"], mh["Temperature (K)"], color="#D55E00", markers=True)
            f.hline(md.extra["temperature"], "thermostat target")
            fig_block(f, TAB, "Fluctuations around the target show the thermostat is working.")
        with c6:
            f = Fig("t5_md_potential", "Langevin dynamics: potential energy", "Time (ps)", "Potential energy (kcal/mol)")
            f.line(mh["Time (ps)"], mh["Potential energy"], color="#0072B2", markers=True)
            fig_block(f, TAB, "Thermal energy lifts the system above the minimum; a stable plateau = equilibrium.")
        traj, nfr = S.network_traj_pdb(net, md.frames)
        mol_viewer(traj, style_sets("rainbow", net.n_chains, stick_r=0.13, lines=net.mol.n > 15000), height=460,
                   frames=True, legend=f"thermal motion · {nfr} frames", interval=200)
        table_block(mh.round(4), "t5_md_log", TAB, "Langevin dynamics log")
    interpretation(
        "Energy should fall steeply and then level off. The most important number is the <b>egg-box score</b>: the share "
        "of Ca²⁺ ions that hold ≥ 4 oxygens coming from ≥ 2 different chains – a true crosslink. A higher Ca–O "
        "coordination and a sharp Ca–O peak near 2.4 Å confirm that chains wrapped around the ions. The model is an "
        "implicit-solvent elastic network, so absolute energies are model-specific – compare runs with each other, not "
        "with calorimetry.")
