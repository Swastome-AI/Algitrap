"""Tab 8 – Active photodegradation with 254 nm UV-C."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from . import state as S
from .constants import PALETTE, PLASTIC_INFO
from .figs import Fig
from .uv import (UV_DEFAULTS, analyse_particles, arrhenius, demo_kinetics_csv, demo_microscope, demo_spectrum,
                 dose_table, fit_first_order, ftir_indices, rate_constant, read_spectrum_table, simulate, to_absorbance)
from .ui import fig_block, interpretation, summary_box, table_block

TAB = "Tab 8 – UV photodegradation"
TIMES = {"24 Hours": 24, "48 Hours": 48, "72 Hours": 72, "96 Hours": 96, "168 Hours (1 week)": 168}
OXID = {"None": 0.0, "Low-dose H₂O₂ (5 mM)": 5.0, "High-dose H₂O₂ (20 mM)": 20.0}


def _read_csv(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text), comment="#", keep_default_na=False, na_values=["", "NA", "NaN", "nan"])


def render():
    summary_box("What happens in this tab",
                "Captured plastics are held inside the water-rich, UV-transparent gel and irradiated with <b>254 nm UV-C</b>. "
                "Photons (and hydroxyl radicals from H₂O₂) break C–C and C–H bonds, oxygen adds to form carbonyl groups, and "
                "chains get shorter (chain scission). This tab (A) predicts dose and kinetics for your settings, (B) fits "
                "your own time-series data, (C) measures oxidation in FTIR spectra and (D) counts fragments in "
                "before/after microscope photos.")
    poly = S.plastic_type()
    st.markdown(f"**Polymer:** {PLASTIC_INFO[poly]['name']} ({poly}) · UV-254 absorption: {PLASTIC_INFO[poly]['uv254']}")

    # ============================================================ A – model
    st.markdown("## A · Experimental variables and model prediction")
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Wavelength", ["254 nm (UV-C)"], key="uv_wl")
    tsel = c2.selectbox("Exposure time", list(TIMES), index=2, key="uv_time")
    temp = c3.selectbox("Temperature", ["20 °C", "25 °C", "35 °C", "45 °C"], index=1, key="uv_temp")
    ox = c4.selectbox("Oxidising catalyst", list(OXID), key="uv_ox")
    c5, c6, c7, c8 = st.columns(4)
    irr = c5.number_input("Lamp irradiance at sample (mW/cm²)", 0.1, 50.0, 2.0, 0.1, key="uv_irr")
    absb = c6.number_input("Gel absorbance at 254 nm (per mm)", 0.0, 2.0, 0.10, 0.01, key="uv_abs")
    thick = c7.number_input("Gel thickness above particles (mm)", 0.0, 20.0, 2.0, 0.5, key="uv_thick")
    with c8.popover("⚙️ Kinetic parameters") if hasattr(c8, "popover") else c8.expander("⚙️ Kinetic parameters"):
        kref = st.number_input("k_ref at 2 mW/cm², 25 °C (1/h)", 0.0001, 1.0,
                               float(UV_DEFAULTS.get(poly, UV_DEFAULTS["Other"])["k_ref_per_h"]), 0.001,
                               format="%.4f", key=f"uv_kref_{poly}")
        ea = st.number_input("Activation energy Ea (kJ/mol)", 0.0, 120.0, 25.0, 1.0, key="uv_ea")
        gain = st.number_input("Max H₂O₂ enhancement (×)", 0.0, 10.0, 1.5, 0.1, key="uv_gain")
    T_h = TIMES[tsel]
    T_c = float(temp.split()[0])
    trans = 10 ** (-absb * thick)
    k = rate_constant(poly, irr, T_c, OXID[ox], trans, ea_kj=ea, ox_gain=gain, custom_k_ref=kref)
    hours = np.linspace(0, T_h, 49)
    sim = simulate(poly, k, hours)
    end = sim.iloc[-1]
    m = st.columns(5)
    m[0].metric("Rate constant k", f"{k:.4f} 1/h")
    m[1].metric("Half-life of oxidation", f"{np.log(2) / k:,.0f} h" if k > 0 else "∞")
    m[2].metric("UV reaching particles", f"{100 * trans:.0f} %")
    m[3].metric(f"Carbonyl index at {T_h} h", f"{end['Carbonyl index']:.3f}")
    m[4].metric("Scissions per chain", f"{end['Scissions per chain']:.1f}")
    st.caption("⚠️ Model curves use illustrative literature-style defaults – replace k_ref and Ea with values fitted from your "
               "own data in section B. They are for planning, not for proving degradation.")
    st.session_state["uv_model"] = {"k": k, "T_h": T_h, "CI_end": float(end["Carbonyl index"]),
                                    "scissions": float(end["Scissions per chain"]), "polymer": poly,
                                    "conditions": f"{tsel}, {temp}, {ox}, {irr:g} mW/cm²"}
    c1, c2 = st.columns(2)
    with c1:
        f = Fig("t8_model_carbonyl", f"Predicted oxidation of {poly}: carbonyl index vs time", "Time (h)",
                "Carbonyl index (–)")
        for j, (lab, mm) in enumerate(OXID.items()):
            kk = rate_constant(poly, irr, T_c, mm, trans, ea_kj=ea, ox_gain=gain, custom_k_ref=kref)
            f.line(hours, simulate(poly, kk, hours)["Carbonyl index"], name=lab, color=PALETTE[j],
                   width=3.5 if lab == ox else 2.0, dash="solid" if lab == ox else "dash")
        fig_block(f, TAB, f"At {temp}; thick line = selected catalyst.")
    with c2:
        f2 = Fig("t8_model_chain_scission", "Predicted chain scission: number-average molar mass", "Time (h)",
                 "Mn / Mn₀")
        for j, tt in enumerate(["20 °C", "25 °C", "35 °C", "45 °C"]):
            kk = rate_constant(poly, irr, float(tt.split()[0]), OXID[ox], trans, ea_kj=ea, ox_gain=gain, custom_k_ref=kref)
            f2.line(hours, simulate(poly, kk, hours)["Mn / Mn0"], name=tt, color=PALETTE[j],
                    width=3.5 if tt == temp else 2.0)
        fig_block(f2, TAB, "Falling Mn = chains are being cut ('shattered') into shorter pieces.")
    temps = [15, 20, 25, 30, 35, 40, 45]
    h2o2 = [0, 2, 5, 10, 20, 40]
    Z = [[simulate(poly, rate_constant(poly, irr, t, o, trans, ea_kj=ea, ox_gain=gain, custom_k_ref=kref),
                   np.array([T_h]))["Carbonyl index"].iloc[0] for t in temps] for o in h2o2]
    f3 = Fig("t8_model_heatmap", f"Predicted carbonyl index after {T_h} h: temperature × H₂O₂", "Temperature (°C)",
             "H₂O₂ (mM)")
    f3.heatmap(Z, [str(t) for t in temps], [str(o) for o in h2o2], scale="Oranges", zlabel="Carbonyl index")
    fig_block(f3, TAB, "Use this map to choose the most informative conditions for real experiments.")
    dt = dose_table(irr, [6, 12, 24, 48, 72, 96, 168], gel_absorbance_per_mm=absb, gel_thickness_mm=thick)
    table_block(dt, "t8_uv_dose", TAB, "UV-C dose delivered (J/cm² and einstein/cm²)")
    table_block(sim.round(4), "t8_model_timecourse", TAB, "Model time course for the selected conditions", height=220)

    # ============================================================ B – lab data
    st.markdown("## B · Your lab data: degradation kinetics")
    up = st.file_uploader("Upload a CSV (columns: time_h, one or more responses, optional condition columns)",
                          type=["csv", "txt"], key="uv_csv")
    use_demo = st.checkbox("Use the simulated demo data set (for practice)", value=up is None, key="uv_demo")
    df = None
    if up is not None:
        try:
            df = _read_csv(up.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read CSV: {e}")
    flags = st.session_state.setdefault("uv_demo_flags", {})
    flags["kinetics"] = up is None
    if up is None and use_demo:
        df = _read_csv(demo_kinetics_csv(poly if poly in ("PS", "PE") else "PS").encode())
        st.caption("Showing SIMULATED DEMO DATA – replace with your measurements.")
        st.download_button("⬇️ Demo CSV (template)", demo_kinetics_csv(poly if poly in ("PS", "PE") else "PS"),
                           file_name="uv_kinetics_template.csv", mime="text/csv", key="dl_uv_demo")
    if df is not None and len(df):
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        c1, c2, c3 = st.columns(3)
        tidx = next((i for i, c in enumerate(num) if "time" in c.lower() or c.lower() in ("t", "h", "hours")), 0)
        tcol = c1.selectbox("Time column", num, index=tidx, key="uv_tcol")
        resp_opts = [c for c in num if c != tcol]
        if not resp_opts:
            st.error("Need at least one numeric response column.")
        else:
            pref = [c for c in resp_opts if any(w in c.lower() for w in ("index", "mass", "remain", "mn", "loss",
                                                                          "weight", "count", "carbonyl", "ci"))]
            ridx = resp_opts.index(pref[0]) if pref else int(np.argmax([df[c].nunique() for c in resp_opts]))
            ycol = c2.selectbox("Response", resp_opts, index=ridx, key="uv_ycol")
            mode = c3.selectbox("Kinetic model", ["growth", "decay", "decay to plateau", "zero-order"],
                                index=0 if "carbonyl" in ycol.lower() else 1, key="uv_mode")
            groups = [c for c in df.columns if c not in (tcol, ycol) and df[c].nunique() <= 12]
            groups = [g for g in groups if g not in resp_opts or df[g].nunique() <= 6]
            gcols = st.multiselect("Group by (drop-down variables)", groups, default=groups[:3], key="uv_groups")
            filt = df.copy()
            fcols = st.columns(max(len(gcols), 1))
            for j, g in enumerate(gcols):
                vals = sorted(filt[g].astype(str).unique())
                sel = fcols[j].multiselect(f"Show {g}", vals, default=vals, key=f"uv_f_{g}")
                filt = filt[filt[g].astype(str).isin(sel)]
            res_rows = []
            f4 = Fig("t8_lab_kinetics", f"Measured {ycol} with fitted {mode} kinetics", tcol, ycol)
            grouped = filt.groupby(gcols) if gcols else [("all data", filt)]
            for j, (key, sub) in enumerate(grouped):
                lab = key if isinstance(key, str) else " · ".join(map(str, key if isinstance(key, tuple) else (key,)))
                col = PALETTE[j % len(PALETTE)]
                f4.scatter(sub[tcol], sub[ycol], name=f"{lab} (data)", color=col, size=9)
                try:
                    fit = fit_first_order(sub[tcol].to_numpy(), sub[ycol].to_numpy(), mode)
                    tt = np.linspace(sub[tcol].min(), sub[tcol].max(), 80)
                    f4.line(tt, fit["_f"](tt, *fit["params"]), name=f"{lab} (fit)", color=col)
                    row = {"Group": lab, **{k_: v for k_, v in fit.items() if not k_.startswith("_") and k_ != "params"}}
                    if gcols:
                        kv = key if isinstance(key, tuple) else (key,)
                        row.update({g: v for g, v in zip(gcols, kv)})
                    res_rows.append(row)
                except Exception as e:  # noqa: BLE001
                    res_rows.append({"Group": lab, "k (1/h)": np.nan, "R²": np.nan, "model": f"fit failed: {e}"})
            fig_block(f4, TAB, "Points = measurements, lines = least-squares fits.")
            rdf = pd.DataFrame(res_rows)
            st.session_state["uv_fits"] = rdf
            if len(rdf) and "k (1/h)" in rdf:
                c1, c2 = st.columns(2)
                with c1:
                    f5 = Fig("t8_rate_constants", "Fitted rate constants by condition", "Condition", "k (1/h)",
                             categorical_x=True)
                    err = (1.96 * rdf["k std. error"]).fillna(0).tolist() if "k std. error" in rdf else None
                    f5.bar(rdf["Group"].astype(str).tolist(), rdf["k (1/h)"].tolist(), yerr=err, color="#0072B2",
                           fmt="{:.4f}")
                    fig_block(f5, TAB, "Error bars = 95 % confidence intervals of the fitted k.")
                with c2:
                    tcols = [g for g in gcols if "temp" in g.lower()]
                    if tcols and len(rdf) >= 2:
                        tc = tcols[0]
                        other = [g for g in gcols if g != tc]
                        f6 = Fig("t8_arrhenius", "Arrhenius plot of fitted rate constants", "1000 / T (1/K)", "ln k")
                        ar_rows = []
                        for j, (key, sub) in enumerate(rdf.groupby(other) if other else [("all", rdf)]):
                            T = pd.to_numeric(sub[tc], errors="coerce")
                            ok = (sub["k (1/h)"] > 0) & T.notna()
                            if ok.sum() >= 2:
                                ar = arrhenius(T[ok].tolist(), sub["k (1/h)"][ok].tolist())
                                lab = key if isinstance(key, str) else " · ".join(map(str, key if isinstance(key, tuple) else (key,)))
                                f6.scatter(1000 / (T[ok] + 273.15), np.log(sub["k (1/h)"][ok]),
                                           name=f"{lab}: Ea = {ar['Ea (kJ/mol)']:.1f} kJ/mol",
                                           color=PALETTE[j % len(PALETTE)], size=11)
                                ar_rows.append({"Series": lab, **ar})
                        fig_block(f6, TAB, "Slope = −Ea/R. Steeper slope = more temperature-sensitive degradation.")
                        if ar_rows:
                            table_block(pd.DataFrame(ar_rows).round(3), "t8_arrhenius", TAB, "Activation energies")
                    else:
                        st.info("Add a temperature column as a group to get an Arrhenius analysis.")
                table_block(rdf.round(5).astype(str), "t8_kinetic_fits", TAB, "Kinetic fits per condition")
    # ============================================================ C – FTIR
    st.markdown("## C · FTIR spectroscopy: proof of oxidation")
    c1, c2 = st.columns(2)
    fb = c1.file_uploader("FTIR before UV (CSV: wavenumber, signal)", type=["csv", "txt"], key="ftir_before")
    fa = c2.file_uploader("FTIR after UV (CSV: wavenumber, signal)", type=["csv", "txt"], key="ftir_after")
    ykind = st.selectbox("Signal type", ["auto", "absorbance", "transmittance %", "transmittance (0–1)"], key="ftir_kind")
    spectra = {}
    try:
        for lab, upf in (("Before UV", fb), ("After UV", fa)):
            if upf is not None:
                t = read_spectrum_table(upf.getvalue())
                spectra[lab] = (t.iloc[:, 0].to_numpy(float), to_absorbance(t.iloc[:, 1].to_numpy(float), ykind))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read FTIR file: {e}")
    st.session_state.setdefault("uv_demo_flags", {})["ftir"] = not spectra
    if not spectra:
        st.caption("No spectra uploaded – showing SIMULATED demo spectra for the selected conditions.")
        spectra["Before UV (simulated)"] = demo_spectrum("PS" if poly == "PS" else "PE", 0, k)
        spectra[f"After {T_h} h UV (simulated)"] = demo_spectrum("PS" if poly == "PS" else "PE", T_h, k, seed=1)
    f7 = Fig("t8_ftir_overlay", "FTIR spectra before and after UV-C", "Wavenumber (cm⁻¹)", "Absorbance (a.u.)",
             xreverse=True)
    rows = []
    for j, (lab, (wn, a)) in enumerate(spectra.items()):
        f7.line(wn, a, name=lab, color=["#000000", "#D55E00", "#0072B2"][j % 3], width=1.8)
        rows.append({"Spectrum": lab, **ftir_indices(wn, a, poly)})
    f7.vline(1715, "C=O 1715", color="#999999", dash="dot").vline(3400, "O–H 3400", color="#999999", dash="dot")
    fig_block(f7, TAB, "New bands at ~1715 cm⁻¹ (carbonyl) and ~3400 cm⁻¹ (hydroxyl) are the fingerprints of photo-oxidation.")
    idx = pd.DataFrame(rows)
    st.session_state["uv_ftir"] = idx
    f8 = Fig("t8_ftir_indices", "Oxidation indices from FTIR (band area / reference band)", "Index", "Value",
             categorical_x=True)
    for j, r in idx.iterrows():
        f8.bar(["Carbonyl", "Hydroxyl", "Vinyl"], [r["Carbonyl index (1850–1650 / ref)"],
                                                  r["Hydroxyl index (3600–3100 / ref)"], r["Vinyl index (925–895 / ref)"]],
               name=r["Spectrum"], color=["#999999", "#D55E00", "#0072B2"][j % 3], fmt="{:.3f}")
    fig_block(f8, TAB, f"Reference band: {PLASTIC_INFO[poly]['ftir_ref_label']} (SAUB method, linear baselines).")
    table_block(idx.round(4), "t8_ftir_indices", TAB, "FTIR oxidation indices")

    # ============================================================ D – microscope
    st.markdown("## D · Microscope images: before vs after")
    c1, c2, c3 = st.columns(3)
    ib = c1.file_uploader("Image before UV", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="img_before")
    ia = c2.file_uploader("Image after UV", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="img_after")
    um = c3.number_input("Scale (µm per pixel)", 0.01, 100.0, 0.5, 0.01, key="img_scale")
    dark = c3.checkbox("Particles are darker than background", True, key="img_dark")
    from PIL import Image
    imgs = {}
    for lab, upf in (("Before UV", ib), ("After UV", ia)):
        if upf is not None:
            try:
                imgs[lab] = np.array(Image.open(io.BytesIO(upf.getvalue())).convert("RGB"))
            except Exception as e:  # noqa: BLE001
                st.error(f"{lab}: {e}")
    st.session_state.setdefault("uv_demo_flags", {})["images"] = not imgs
    if not imgs:
        st.caption("No images uploaded – showing SIMULATED demo micrographs.")
        imgs = {"Before UV (simulated)": demo_microscope(False), "After UV (simulated)": demo_microscope(True)}
    cols = st.columns(len(imgs))
    res = {}
    for col, (lab, im) in zip(cols, imgs.items()):
        r = analyse_particles(im, um, dark)
        res[lab] = r
        with col:
            st.image(im, caption=f"{lab}: {r['count']} particles")
    f9 = Fig("t8_particle_sizes", "Particle size distribution from microscopy", "Equivalent diameter (µm)",
             "Fraction of particles")
    allmax = max([r["eq_diameters_um"].max() for r in res.values() if len(r["eq_diameters_um"])] + [1])
    for j, (lab, r) in enumerate(res.items()):
        if len(r["eq_diameters_um"]):
            f9.hist(r["eq_diameters_um"], bins=20, name=lab, color=["#0072B2", "#D55E00"][j % 2], density=True,
                    rng=(0, allmax))
    fig_block(f9, TAB, "A shift to smaller sizes and more particles = fragmentation.")
    prow = pd.DataFrame([{"Image": lab, "Particles": r["count"],
                          "Mean diameter (µm)": round(float(r["eq_diameters_um"].mean()), 3) if r["count"] else np.nan,
                          "Median diameter (µm)": round(float(np.median(r["eq_diameters_um"])), 3) if r["count"] else np.nan,
                          "Area fraction": round(r["area_fraction"], 4)} for lab, r in res.items()])
    if len(prow) == 2 and prow["Particles"].iloc[0] > 0:
        prow["Fragmentation index (N/N_before)"] = [1.0, round(prow["Particles"].iloc[1] / prow["Particles"].iloc[0], 3)]
    st.session_state["uv_images"] = prow
    table_block(prow, "t8_microscopy", TAB, "Image analysis summary (Otsu threshold + connected components)")
    interpretation(
        "Look for three independent signs of degradation: (1) the <b>carbonyl index rises</b> (new C=O groups from "
        "photo-oxidation); (2) particles become <b>smaller and more numerous</b> (fragmentation index > 1); (3) mass or "
        "molar mass falls (chain scission). The fitted <b>k</b> lets you compare conditions fairly: a larger k with H₂O₂ "
        "shows radical-assisted degradation, and the Arrhenius Ea tells how much warming helps. PS absorbs 254 nm strongly "
        "through its phenyl rings; PE barely absorbs, so it usually needs H₂O₂ or longer exposure. Note that "
        "fragmentation alone can <i>increase</i> the number of microplastics – the goal is oxidation all the way to small, "
        "soluble products, so report both indices and counts.")
