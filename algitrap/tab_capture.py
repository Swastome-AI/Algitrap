"""Tab 6 – Microplastic capture prediction (pore analysis)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import state as S
from .anim import capture_animation
from .constants import COLOR_BAD, COLOR_CA, COLOR_GOOD
from .figs import Fig
from .multiscale import (bead_diameter_tate, capture_curve_mesh, g_prime_estimate, mesh_size_canal_peppas,
                         mesh_size_rubber, multiscale_verdicts)
from .network import ca_nn_distances
from .pipeline import capture_curve, critical_spacing, design_sweep, pore_analysis, pore_dynamics
from .pores import capture_verdict, renkin, transport_mc
from .ui import show_df, show_plotly, fig_block, interpretation, stretch, summary_box, table_block, verdict

TAB = "Tab 6 – Capture prediction"
UNITS = {"Å": 1.0, "nm": 10.0, "µm": 1e4, "mm": 1e7}


def particle_size() -> tuple[float, str]:
    ss = st.session_state
    src = ss.get("cap_src", "Enter manually")
    mp = ss.get("mp")
    if mp and not src.startswith("Enter"):
        key = {"Uploaded microplastic – minimum cross-section": "Min cross-section diameter (Å)",
               "Uploaded microplastic – equivalent sphere": "Equivalent-sphere diameter (Å)",
               "Uploaded microplastic – maximum dimension": "Max dimension (Å)"}.get(src, "Min cross-section diameter (Å)")
        return float(mp["size"][key]), f"{mp['ident']['label']} oligomer, {key.split(' (')[0].lower()}"
    val = float(ss.get("cap_val", 15.0))
    unit = ss.get("cap_unit", "Å")
    return val * UNITS[unit], f"manual entry ({val:g} {unit})"


def run_pores(progress=None):
    net = st.session_state["net"]
    P = S.P()
    h = float(st.session_state.get(S.k("cap_h"), P["pore_h"]))
    S.invalidate("pores")
    out = {"h": h}
    if progress:
        progress(0.05, "Distance field – rigid network")
    r = pore_analysis(net, h=h)
    r["grid"].field = r["grid"].field.astype(np.float32)
    out["rigid"] = r
    rel = st.session_state.get("relax")
    if rel is not None:
        if progress:
            progress(0.45, "Distance field – relaxed network")
        r2 = pore_analysis(net, X=rel.final, h=h)
        r2["grid"].field = r2["grid"].field.astype(np.float32)
        out["relaxed"] = r2
    st.session_state["pores"] = out
    if progress:
        progress(0.75, "Capture-efficiency curve")
    run_curve()
    if progress:
        progress(1.0, "Done")


def run_curve():
    pr = st.session_state["pores"]
    P = S.P()
    d, _ = particle_size()
    best = pr.get("relaxed") or pr["rigid"]
    hi = max(2.0 * best["pld"], 1.3 * d if d < 200 else 2.0 * best["pld"], 10.0)
    sizes = np.round(np.linspace(2.0, hi, 12), 2)
    rows = []
    for state_name in ("rigid", "relaxed"):
        if pr.get(state_name):
            r = pr[state_name]
            df = capture_curve(r["grid"], r["pld"], r["psd"], sizes, walkers=int(P["walkers"] * 0.6),
                               steps=P["mc_steps"])
            df.insert(0, "Network", "Rigid" if state_name == "rigid" else "Relaxed")
            rows.append(df)
    st.session_state["capture_curve"] = pd.concat(rows, ignore_index=True)


def render():
    summary_box("What happens in this tab",
                "A 3-D grid is laid through the network and, for every point, the distance to the nearest atom surface "
                "is computed. From this <b>distance field</b> AlgiTrap finds the <b>pore-limiting diameter (PLD)</b> – "
                "the widest sphere that can travel all the way through the gel – plus the full pore-size distribution. "
                "Your microplastic size is compared with the PLD (size-exclusion capture), with hindered-diffusion "
                "theory, and with a Monte-Carlo particle-transport simulation that you can watch as an animation.")
    net = st.session_state.get("net")
    if net is None:
        st.info("Build the network in Tab 4 first (and ideally relax it in Tab 5).")
        return
    P = S.P()
    mp = st.session_state.get("mp")
    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        opts = (["Uploaded microplastic – minimum cross-section", "Uploaded microplastic – equivalent sphere",
                 "Uploaded microplastic – maximum dimension"] if mp else []) + ["Enter manually"]
        st.radio("Particle size from", opts, key="cap_src")
    with c2:
        st.number_input("Enter your microplastic size", 0.1, 1e9, 15.0, key="cap_val",
                        disabled=not st.session_state.get("cap_src", "Enter manually").startswith("Enter"))
        st.selectbox("Unit", list(UNITS), key="cap_unit",
                     disabled=not st.session_state.get("cap_src", "Enter manually").startswith("Enter"))
    with c3:
        st.number_input("Grid spacing (Å) – smaller = more precise", 0.25, 1.5, P["pore_h"], 0.05, key=S.k("cap_h"))
        st.checkbox("Track pore dynamics during relaxation", True, key="cap_dyn")
    d_A, d_src = particle_size()
    st.session_state["particle_A"] = d_A
    st.caption(f"Particle diameter used: **{d_A:,.2f} Å** = {d_A / 10:,.3f} nm = {d_A / 1e4:,.5f} µm ({d_src}).")
    if st.button("🔬 Run pore analysis", type="primary", key="btn_pores"):
        bar = st.progress(0.0, text="Starting…")
        run_pores(lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
        rel = st.session_state.get("relax")
        if rel is not None and st.session_state.get("cap_dyn", True):
            bar.progress(0.9, text="Pore dynamics along the relaxation")
            frames, labels = list(rel.frames), list(rel.frame_labels)
            md = st.session_state.get("md")
            if md is not None:
                frames += md.frames[1:]
                labels += ["MD " + l for l in md.frame_labels[1:]]
            st.session_state["dyn"] = pore_dynamics(net, frames, labels, max_frames=P["dyn_frames"])
        bar.empty()
    pr = st.session_state.get("pores")
    if pr is None:
        st.info("Press **Run pore analysis**.")
        return
    best_name = "relaxed" if pr.get("relaxed") else "rigid"
    best = pr[best_name]
    v = capture_verdict(d_A, best["pld"], best["psd"])
    blocked = v["Size-exclusion capture"] == "YES"
    if blocked:
        verdict(True, "SUCCESS – the microplastic is trapped",
                f"Particle diameter <b>{d_A:,.2f} Å</b> is larger than the pore-limiting diameter "
                f"<b>{best['pld']:.2f} Å</b> of the {best_name} network (margin {d_A - best['pld']:,.2f} Å). It cannot "
                f"percolate through the gel and is <b>BLOCKED by size exclusion</b>.")
    else:
        verdict(False, "WARNING – the microplastic can migrate through",
                f"Particle diameter <b>{d_A:,.2f} Å</b> is smaller than the pore-limiting diameter "
                f"<b>{best['pld']:.2f} Å</b> of the {best_name} network. A continuous channel exists, so it can "
                f"<b>MIGRATE</b> through (hindered: D<sub>eff</sub>/D<sub>0</sub> ≈ {v['D_eff / D0']:.3f}). Tighten the mesh "
                f"(smaller chain spacing, more layers or more Ca²⁺) – see the design curve below.")
    s = best["summary"]
    m = st.columns(6)
    m[0].metric("PLD", f"{s['PLD – pore-limiting diameter (Å)']:.2f} Å")
    m[1].metric("LCD", f"{s['LCD – largest cavity diameter (Å)']:.2f} Å")
    m[2].metric("Median pore d50", f"{s['Median pore diameter d50 (Å)']:.2f} Å")
    m[3].metric("Porosity", f"{100 * s['Porosity (void fraction in slab)']:.1f} %")
    m[4].metric("Particle / PLD (λ)", f"{v['λ = d / PLD']:.2f}")
    m[5].metric("Accessible pore volume", f"{100 * v['Accessible pore-volume fraction']:.1f} %")

    # ------------------------------------------------------------- PSD plots
    rmax = max(pr["rigid"]["psd"].max() if len(pr["rigid"]["psd"]) else 1,
               pr.get("relaxed", {}).get("psd", np.zeros(1)).max() if pr.get("relaxed") else 0)
    f = Fig("t6_pore_size_distribution", "Pore-size distribution (volume-weighted, geometric PSD)",
            "Pore diameter (Å)", "Fraction of void volume")
    f.hist(pr["rigid"]["psd"], bins=30, name="Rigid network", color="#999999", density=True, rng=(0, rmax))
    if pr.get("relaxed"):
        f.hist(pr["relaxed"]["psd"], bins=30, name="Relaxed network", color=COLOR_CA, density=True, rng=(0, rmax))
    f.vline(best["pld"], f"PLD {best['pld']:.1f} Å", color="#000000")
    if d_A < 3 * rmax:
        f.vline(d_A, f"particle {d_A:.1f} Å", color=COLOR_BAD if not blocked else COLOR_GOOD)
    fig_block(f, TAB, "Each void point is assigned the diameter of the largest empty sphere that covers it.")
    c1, c2 = st.columns(2)
    with c1:
        cen = best["centres"]
        f2 = Fig("t6_pore_centre_histogram", "Distribution of hydrogel pore sizes (pore centres)", "Pore diameter (Å)",
                 "Frequency")
        f2.hist(cen, bins=18, color=COLOR_CA)
        if len(cen):
            f2.vline(float(cen.mean()), f"average {cen.mean():.1f} Å", color="#D55E00")
        fig_block(f2, TAB, "Count-based view: one entry per local pore centre (largest inscribed sphere).")
    with c2:
        nn = ca_nn_distances(net.mol.coords if best_name == "rigid" else st.session_state["relax"].final,
                             net.mol.elements)
        labels = ["Ca–Ca node spacing", "Pore-centre mean", "PSD median d50", "PLD (limiting)", "LCD (largest)"]
        vals = [float(nn.mean()) if len(nn) else np.nan, float(best["centres"].mean()) if len(best["centres"]) else np.nan,
                s["Median pore diameter d50 (Å)"], best["pld"], s["LCD – largest cavity diameter (Å)"]]
        f3 = Fig("t6_pore_size_methods", "Pore size by method", "Method", "Size (Å)", categorical_x=True)
        f3.bar(labels, vals, color=["#999999", COLOR_CA, "#0072B2", "#000000", "#009E73"], fmt="{:.1f}")
        if d_A < 3 * max(v_ for v_ in vals if np.isfinite(v_)):
            f3.hline(d_A, f"particle {d_A:.1f} Å", color=COLOR_BAD)
        fig_block(f3, TAB, "The PLD is the method that decides sieving; node spacing over-estimates open space.")

    # ------------------------------------------------------- capture curve
    cc = st.session_state.get("capture_curve")
    if cc is not None and len(cc):
        if st.button("↻ Recompute capture curve for the current particle size", key="btn_curve"):
            with st.spinner("Running Monte-Carlo transport for 12 sizes…"):
                run_curve()
            cc = st.session_state["capture_curve"]
        f4 = Fig("t6_capture_efficiency_curve", "Capture efficiency versus particle size", "Particle diameter (Å)",
                 "Capture efficiency (%)", yrange=[-3, 105])
        for name, col in (("Rigid", "#999999"), ("Relaxed", COLOR_CA)):
            sub = cc[cc["Network"] == name]
            if len(sub):
                f4.line(sub["Particle diameter (Å)"], sub["Capture efficiency (%)"], name=f"{name} – Monte-Carlo",
                        color=col, markers=True)
                f4.line(sub["Particle diameter (Å)"], 100 * (1 - sub["Accessible pore-volume fraction"]),
                        name=f"{name} – excluded pore volume", color=col, dash="dash")
        f4.vline(best["pld"], f"PLD {best['pld']:.1f} Å")
        if d_A <= cc["Particle diameter (Å)"].max():
            f4.vline(d_A, "your particle", color=COLOR_BAD)
        fig_block(f4, TAB, "Solid: share of random-walk particles that never got through. Dashed: share of pore volume "
                           "a particle of that size cannot enter.")
        c3, c4 = st.columns(2)
        with c3:
            dd = np.linspace(0.1, best["pld"] * 1.05, 60)
            f5 = Fig("t6_hindered_diffusion", "Hindered transport inside pores (Renkin model)", "Particle diameter (Å)",
                     "Relative diffusivity D_eff / D₀")
            f5.line(dd, [renkin(x / best["pld"])["D_eff / D0"] for x in dd], name="Renkin", color="#0072B2")
            f5.line(dd, [renkin(x / best["pld"])["Partition coefficient Φ"] for x in dd], name="Partition Φ",
                    color="#E69F00", dash="dash")
            if d_A < best["pld"] * 1.05:
                f5.vline(d_A, "your particle", color=COLOR_BAD)
            fig_block(f5, TAB, "Even particles smaller than the PLD are slowed down strongly as they approach it.")
        with c4:
            dyn = st.session_state.get("dyn")
            if dyn is not None and len(dyn):
                f6 = Fig("t6_pore_dynamics", "Pore-size dynamics during relaxation", "Frame",
                         "Diameter (Å)")
                xs = list(range(len(dyn)))
                f6.line(xs, dyn["PLD (Å)"], name="PLD", color="#000000", markers=True)
                f6.line(xs, dyn["LCD (Å)"], name="LCD", color="#009E73", markers=True)
                f6.hline(d_A, "particle", color=COLOR_BAD) if d_A < 3 * dyn["LCD (Å)"].max() else None
                fig_block(f6, TAB, "Frame 0 = rigid mesh; later frames = egg-box formation (and MD breathing).")
            else:
                st.info("Relax the network in Tab 5 and re-run to see how pores change during egg-box formation.")
        table_block(cc.round(4), "t6_capture_curve", TAB, "Capture curve data (Monte-Carlo + geometry)")
        if st.session_state.get("dyn") is not None:
            table_block(st.session_state["dyn"], "t6_pore_dynamics", TAB, "Pore metrics for each trajectory frame")

    # ------------------------------------------------------- MC animation
    st.markdown("### 🎞️ Watch particles try to cross the gel")
    grid = best["grid"]
    mc = st.session_state.get("mc")
    if mc is None or mc.get("d") != round(d_A, 3) or mc.get("net") != best_name:
        res = transport_mc(grid, d_A / 2, n_walkers=P["walkers"], n_steps=P["mc_steps"], n_record=24, record_every=40)
        mc = {"d": round(d_A, 3), "net": best_name, "res": res}
        st.session_state["mc"] = mc
    res = mc["res"]
    if res["walkers"] == 0:
        st.warning("The particle is too large to even enter the region above the mesh – it is trivially blocked.")
    else:
        X = st.session_state["relax"].final if best_name == "relaxed" else net.mol.coords
        fig = capture_animation(X, net.mol, net.atom_res, net.atom_chain, res, d_A,
                                f"Monte-Carlo transport: {100 * res['transmission']:.0f} % of {res['walkers']} particles "
                                f"crossed ({d_A:.1f} Å)")
        show_plotly(fig, "pl_capture_anim")
        st.caption("Particles start above the gel and random-walk with a downward drift (flow). Green = crossed, "
                   "orange = still trapped. Positions are particle centres; the walk only visits space where a sphere of "
                   "this diameter fits.")

    # ------------------------------------------------------- design sweep
    st.markdown("### 📐 Design rule: which chain spacing traps this particle?")
    if st.button("Run spacing sweep", key="btn_sweep"):
        bar = st.progress(0.0, text="Sweeping")
        base = net.params
        sp = np.linspace(max(8.0, base.spacing * 0.5), base.spacing * 1.6, P["sweep_n"]).round(1)
        st.session_state["sweep"] = design_sweep(st.session_state["chain"], base, sp.tolist(),
                                                 progress=lambda f, t: bar.progress(f, text=t))
        bar.empty()
    sw = st.session_state.get("sweep")
    if sw is not None:
        crit = critical_spacing(sw, d_A)
        f7 = Fig("t6_design_curve", "Design curve: pore-limiting diameter versus chain spacing (rigid mesh)",
                 "Chain spacing (Å)", "PLD (Å)")
        f7.line(sw["Chain spacing (Å)"], sw["PLD (Å)"], name="PLD", color="#0072B2", markers=True)
        f7.hline(d_A, f"particle {d_A:.1f} Å", color=COLOR_BAD)
        if crit:
            f7.vline(crit, f"critical spacing {crit:.1f} Å", color="#009E73")
        fig_block(f7, TAB, "Keep the spacing to the left of the critical value to trap the particle.")
        if crit:
            st.success(f"Spacings below **{crit:.1f} Å** give PLD < {d_A:.1f} Å → the particle is trapped.")
        table_block(sw, "t6_design_sweep", TAB, "Spacing sweep results")

    # ------------------------------------------------------- multiscale
    st.markdown("### 🔭 From molecules to real microplastics (multiscale check)")
    conc = float(st.session_state.get("wl_conc", 2.0))
    gp = float(st.session_state.get("wl_gprime", g_prime_estimate(conc)))
    xi_lo = mesh_size_rubber(gp)
    xi_hi = mesh_size_canal_peppas(conc, gp)["xi_nm"]
    bead = bead_diameter_tate(st.session_state.get("wl_needle", "22 G"))["Gel bead diameter (mm)"]
    ms = pd.DataFrame(multiscale_verdicts(d_A, best["pld"], (min(xi_lo, xi_hi), max(xi_lo, xi_hi)), bead))
    table_block(ms, "t6_multiscale_verdicts", TAB, f"Capture at three length scales (alginate {conc:g} % w/v, G′ ≈ {gp:,.0f} Pa, "
                                                   f"beads ≈ {bead:.1f} mm)")
    dn = np.logspace(-1, 3.5, 80)
    og = capture_curve_mesh(dn, xi_hi, phi=conc / 100 / 1.6)
    f8 = Fig("t6_gel_partition", "Bulk gel: share of gel volume open to a particle (Ogston model)",
             "Particle diameter (nm, log scale)", "Accessible fraction of gel volume", logx=True, yrange=[-0.03, 1.05])
    f8.line(dn, og["ogston"], name=f"Ogston, {conc:g} % gel", color="#0072B2")
    f8.vline(min(xi_lo, xi_hi), f"ξ low {min(xi_lo, xi_hi):.0f} nm", color="#999999")
    f8.vline(max(xi_lo, xi_hi), f"ξ high {max(xi_lo, xi_hi):.0f} nm", color="#000000")
    if 0.1 <= d_A / 10 <= 10 ** 3.5:
        f8.vline(d_A / 10, "your particle", color=COLOR_BAD)
    fig_block(f8, TAB, "Particles far bigger than the mesh size ξ cannot enter the gel at all – they are retained at its "
                       "surface or entrapped when the gel forms around them.")
    srows = pd.DataFrame([{"Metric": k, "Rigid": pr["rigid"]["summary"][k],
                           "Relaxed": pr["relaxed"]["summary"][k] if pr.get("relaxed") else "–"}
                          for k in pr["rigid"]["summary"]]).astype(str)
    table_block(srows, "t6_pore_summary", TAB, "Pore metrics – rigid vs relaxed network")
    table_block(pd.DataFrame([v]).T.reset_index().rename(columns={"index": "Quantity", 0: "Value"}).astype(str),
                "t6_capture_verdict", TAB, "Capture verdict details")
    interpretation(
        "<b>PLD is the decisive number</b>: if your particle's smallest cross-section is larger than the PLD it cannot cross "
        "the gel (✅). If it is smaller (⚠️), it can migrate, but the Renkin curve shows how strongly it is slowed down. "
        "The capture curve is S-shaped – the steep part sits just below the PLD. Compare <i>rigid</i> and <i>relaxed</i>: "
        "egg-box formation usually tightens junctions and opens or closes particular pores, so the pore-dynamics plot "
        "shows whether relaxation helps capture. Real microplastics (µm–mm) are thousands of times larger than these "
        "molecular pores, so inside an intact gel they are always size-excluded; the multiscale table shows that the "
        "weak point of a bead filter is the <i>gaps between beads</i>, not the gel itself.")
