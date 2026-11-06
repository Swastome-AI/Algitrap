"""Tab 7 – Wet-lab synthesis (physical prototyping)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import state as S
from .figs import Fig
from .multiscale import (CACL2_MW, NEEDLE_OD_MM, bead_diameter_tate, g_prime_estimate, gelation_time_min,
                         interstitial_throat_um, mesh_size_canal_peppas, mesh_size_rubber, mesh_size_table,
                         solution_recipe)
from .ui import dict_table, fig_block, interpretation, summary_box, table_block

TAB = "Tab 7 – Wet-lab synthesis"

PROTOCOL = """
### Materials
* Sodium alginate powder (food or lab grade; note the supplier's M/G ratio if given)
* Calcium chloride (anhydrous or dihydrate)
* Distilled or deionised water
* Polystyrene microbeads (e.g. 1–10 µm) or polyethylene powder as the microplastic proxy
* Syringe (10–20 mL) with a blunt needle, magnetic stirrer, beakers, sieve or strainer, balance
* Optional: microscope, haemocytometer or particle counter, UV-C lamp in a closed box (Tab 8)

### Safety
Wear goggles and gloves. CaCl₂ dissolution is exothermic – add the solid slowly. **UV-C (254 nm) damages eyes and skin:**
only run it inside a closed, opaque box and never look at the lamp. Collect microplastic waste – do not pour it down the drain.

### Extrusion protocol
1. **Polymer solution.** Sprinkle the alginate slowly onto stirred water (vortex) to avoid lumps; stir 1–2 h until clear,
   then let it stand 30 min to remove bubbles. The Na⁺ ions dissociate, leaving negatively charged chains.
2. **Load the microplastics (entrapment mode, optional).** Disperse a known mass of microbeads in the alginate solution.
3. **Crosslinking bath.** Dissolve CaCl₂ in water (see calculator) in a beaker on a stirrer at low speed.
4. **Gelation.** Drip the alginate through the needle from ~10 cm height into the bath. Ca²⁺ locks G-blocks together
   instantly – beads form on contact. Leave them to cure for the time given below (the Ca²⁺ front must reach the centre).
5. **Wash.** Sieve the beads and rinse 2–3× with distilled water to remove free Ca²⁺.
6. **Filtration test (flow-through mode).** Pack beads into a column (or cast a gel sheet on a mesh), pass a known
   suspension of microplastics, and count particles before and after (microscope counts, turbidity or filtration and weighing).
7. **Capture efficiency** = (C_in − C_out) / C_in × 100 %. Repeat at least three times and report mean ± SD.
"""


def render():
    summary_box("What happens in this tab",
                "The computer model is turned into a bench protocol. Calculators give exact masses to weigh, the "
                "Ca²⁺/G stoichiometry of your bath, the bead size you will get from a given needle (Tate's law), how "
                "long beads need to cure, and the theoretical mesh size of the resulting gel – which links your "
                "experiment back to the pore analysis of Tab 6.")
    st.markdown(PROTOCOL)
    st.markdown("### 🧮 Recipe calculator")
    chain = st.session_state.get("chain")
    fg_default = chain.sequence.count("G") / len(chain.sequence) if chain else 0.6
    c1, c2, c3, c4 = st.columns(4)
    conc = c1.number_input("Alginate concentration (% w/v)", 0.25, 6.0, 2.0, 0.25, key="wl_conc")
    vol = c1.number_input("Alginate solution volume (mL)", 5.0, 2000.0, 50.0, 5.0, key="wl_vol")
    ca_mm = c2.number_input("CaCl₂ bath concentration (mM)", 10.0, 1000.0, 100.0, 10.0, key="wl_ca")
    ca_vol = c2.number_input("Bath volume (mL)", 10.0, 5000.0, 200.0, 10.0, key="wl_cavol")
    salt = c3.selectbox("Calcium salt", list(CACL2_MW), index=1, key="wl_salt")
    fg = c3.number_input("G fraction of your alginate", 0.05, 0.95, float(round(fg_default, 2)), 0.05, key="wl_fg")
    needle = c4.selectbox("Needle gauge", list(NEEDLE_OD_MM), index=4, key="wl_needle")
    gp_est = g_prime_estimate(conc)
    gp = c4.number_input("Gel modulus G′ (Pa) – measured or estimated", 100.0, 1e6, float(round(gp_est, -1)), 100.0,
                         key="wl_gprime", help=f"Estimate for {conc:g} % w/v: {gp_est:,.0f} Pa (G′ ∝ c², 10 kPa at 2 %).")
    rec = solution_recipe(conc, vol, ca_mm, ca_vol, salt, fg)
    bead = bead_diameter_tate(needle)
    cure = gelation_time_min(bead["Gel bead diameter (mm)"])
    m = st.columns(5)
    m[0].metric("Weigh alginate", f"{rec['Sodium alginate to weigh (g)']:.3f} g")
    m[1].metric(f"Weigh {salt}", f"{rec[f'{salt} to weigh (g)']:.3f} g")
    m[2].metric("Bath R = Ca²⁺/G", f"{rec['Bath excess R = Ca²⁺ / G']:.1f}")
    m[3].metric("Expected bead size", f"{bead['Gel bead diameter (mm)']:.2f} mm")
    m[4].metric("Cure time (≥)", f"{cure:.0f} min")
    rdf = dict_table({**rec, **bead, "Ca²⁺ diffusion cure time (min)": cure,
                      "Gap between packed beads (µm)": interstitial_throat_um(bead["Gel bead diameter (mm)"])},
                     "Quantity", "Value")
    table_block(rdf, "t7_recipe", TAB, "Recipe and bead predictions")
    mt = pd.DataFrame(mesh_size_table(conc, gp))
    table_block(mt, "t7_mesh_size", TAB, "Theoretical mesh size of the gel")

    c1, c2 = st.columns(2)
    with c1:
        g = list(NEEDLE_OD_MM)
        ds = [bead_diameter_tate(n_)["Gel bead diameter (mm)"] for n_ in g]
        f = Fig("t7_bead_size_vs_needle", "Predicted bead diameter for each needle gauge (Tate's law)", "Needle gauge",
                "Bead diameter (mm)", categorical_x=True)
        f.bar(g, ds, color=["#D55E00" if n_ == needle else "#0072B2" for n_ in g], fmt="{:.2f}")
        fig_block(f, TAB, "Orange = your needle. Thinner needles → smaller beads → shorter cure time.")
    with c2:
        cs = np.linspace(0.5, 4.0, 30)
        f2 = Fig("t7_mesh_vs_concentration", "Gel mesh size versus alginate concentration", "Alginate (% w/v)",
                 "Mesh size ξ (nm, log scale)", logy=True)
        f2.line(cs, [mesh_size_rubber(g_prime_estimate(c)) for c in cs], name="Rubber elasticity", color="#0072B2")
        f2.line(cs, [mesh_size_canal_peppas(c, g_prime_estimate(c))["xi_nm"] for c in cs], name="Canal–Peppas",
                color="#E69F00", dash="dash")
        f2.vline(conc, f"your gel {conc:g} %")
        pa = st.session_state.get("particle_A")
        if pa and 0.5 < pa / 10 < 5000:
            f2.hline(pa / 10, "your particle", color="#D55E00", dash="dot")
        fig_block(f2, TAB, "More concentrated gels have tighter meshes; the two models bracket the real value.")
    cs2 = np.linspace(0.5, 4.0, 30)
    f3 = Fig("t7_calcium_demand", "Ca²⁺ needed per 100 mL of alginate solution", "Alginate (% w/v)",
             "Ca²⁺ (mmol per 100 mL)")
    f3.line(cs2, [solution_recipe(c, 100, 0, 100, salt, fg)["Ca²⁺ needed for egg-box saturation, R = 0.25 (mmol)"]
                  for c in cs2], name="Egg-box saturation (R = 0.25)", color="#E69F00")
    f3.line(cs2, [solution_recipe(c, 100, 0, 100, salt, fg)["Ca²⁺ needed for full charge neutralisation (mmol)"]
                  for c in cs2], name="Full charge neutralisation", color="#0072B2", dash="dash")
    f3.hline(ca_mm / 1000 * ca_vol / vol * 100, "your bath (scaled to 100 mL gel)", color="#D55E00", dash="dot")
    fig_block(f3, TAB, "A bath well above both lines guarantees fast, complete gelation.")
    checklist = "\n".join([
        f"AlgiTrap wet-lab checklist – prepared for {st.session_state.get('user_name') or 'user'}",
        f"Alginate: {rec['Sodium alginate to weigh (g)']:.3f} g in {vol:g} mL water ({conc:g} % w/v)",
        f"{salt}: {rec[f'{salt} to weigh (g)']:.3f} g in {ca_vol:g} mL water ({ca_mm:g} mM)",
        f"Needle {needle}: expected bead {bead['Gel bead diameter (mm)']:.2f} mm, cure ≥ {cure:.0f} min",
        "Safety: goggles, gloves; UV-C only in a closed box.", "", PROTOCOL])
    st.download_button("⬇️ Download printable protocol + recipe (.txt)", checklist.encode("utf-8"),
                       file_name="AlgiTrap_wetlab_protocol.txt", mime="text/plain", key="dl_protocol")
    interpretation(
        "The bath normally contains a large Ca²⁺ excess (R ≫ 0.25), which makes gelation fast but also leaves free Ca²⁺ "
        "to wash out. Bead size controls two things: cure time (∝ radius²) and the size of the gaps between packed beads "
        "(≈ 15 % of the bead diameter). Particles smaller than those gaps can pass a bead column unless they stick to or "
        "are entrapped in the gel – so for small microplastics a continuous gel sheet, or mixing the particles into the "
        "alginate before gelation (entrapment), works better than a loose bead bed.")
