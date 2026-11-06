"""Tab 2 – The science of alginate (introduction)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .figs import Fig
from .multiscale import g_prime_estimate, mesh_size_canal_peppas, mesh_size_rubber, scale_ladder
from .ui import fig_block, interpretation, summary_box, table_block

TAB = "Tab 2 – The science"

PIPELINE_DOT = """
digraph G {
  rankdir=LR; bgcolor="white";
  node [shape=box, style="rounded,filled", fillcolor="#eef6fb", color="#0072B2", fontname="Arial", fontsize=11];
  edge [color="#333333"];
  a [label="1 Upload\\n.pdb + .sdf"]; b [label="3 Chain\\n(M/G sequence)"]; c [label="4 Mesh + Ca²⁺\\n(crosslinks)"];
  d [label="5 Energy\\nminimisation"]; e [label="6 Pores &\\ncapture"]; f [label="7 Wet-lab\\nhydrogel"];
  g [label="8 UV-C 254 nm\\ndegradation"]; h [label="9 Report\\n& ZIP"];
  a -> b -> c -> d -> e -> f -> g -> h;
}
"""


def render():
    summary_box("What this tab explains",
                "Why seaweed-derived <b>sodium alginate</b> can form a molecular net, how its two sugar units behave, "
                "why microplastics are hard to remove, and how the rest of the app turns that idea into numbers.")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("""
### Sodium alginate – a polymer from brown seaweed
Alginate is a linear polysaccharide extracted from brown algae (*Laminaria*, *Macrocystis*, *Ascophyllum*). Its chain is
built from two **uronic-acid** sugars joined by 1→4 glycosidic bonds:

* **M-units – β-D-mannuronate.** Sit in the ⁴C₁ chair with equatorial–equatorial links, so poly-M forms a **flat,
  flexible ribbon** (about 5.2 Å per residue).
* **G-units – α-L-guluronate.** The C5 "mirror" of M. They flip into the ¹C₄ chair with **axial–axial** links, so poly-G
  **buckles into a zig-zag** (about 4.35 Å per residue) that leaves regular cavities.

In water the Na⁺ counter-ions dissociate and every residue carries a negative carboxylate (COO⁻).

### The egg-box model
When **Ca²⁺** is added, each ion sits in a cavity between two buckled G-blocks, held by carboxylate, ring and hydroxyl
oxygens – like eggs in an egg-box. These junctions crosslink chains into a 3-D **hydrogel** within seconds, which is why
dripping alginate into CaCl₂ makes solid beads instantly.
""")
    with c2:
        st.markdown("""
### The problem – microplastics
Microplastics are plastic particles **smaller than 5 mm**; nanoplastics are **below 1 µm**. They come from fragmented
packaging, textile fibres and tyre wear, and polystyrene (PS) and polyethylene (PE) are among the most common.

**Why ordinary filtering is not enough**
* Sand and screen filters remove by size, so the smallest particles slip through.
* Fine membranes clog (foul) quickly and need high pressure.
* Particles that *are* caught are only **moved** (into sludge) – the polymer is not destroyed.

### The idea tested here
1. Build an alginate–Ca²⁺ network on the computer and measure its pores.
2. Predict which particle sizes are **trapped** by size exclusion.
3. Make the gel in the lab and trap real particles.
4. Hold them in the transparent gel and break their bonds with **254 nm UV-C** light.
""")
        st.info("💡 **Project goal:** model a hydrogel mesh that physically captures microplastics and holds them in place "
                "for active UV-C photodegradation.")
    st.markdown("#### Workflow of this app")
    try:
        st.graphviz_chart(PIPELINE_DOT)
    except Exception:
        st.write("Upload → Chain → Mesh + Ca²⁺ → Minimisation → Pores → Wet lab → UV → Report")

    mg = pd.DataFrame([
        {"Property": "Full name", "M-unit": "β-D-mannuronate", "G-unit": "α-L-guluronate"},
        {"Property": "Ring conformation", "M-unit": "⁴C₁ chair", "G-unit": "¹C₄ chair"},
        {"Property": "Glycosidic link geometry", "M-unit": "diequatorial", "G-unit": "diaxial"},
        {"Property": "Chain shape", "M-unit": "flat, flexible ribbon", "G-unit": "buckled zig-zag"},
        {"Property": "Rise per residue (fibre data)", "M-unit": "5.17 Å", "G-unit": "4.35 Å"},
        {"Property": "C5 configuration (CIP)", "M-unit": "S", "G-unit": "R"},
        {"Property": "Ca²⁺ binding", "M-unit": "weak, non-specific", "G-unit": "strong, cooperative (egg-box)"},
        {"Property": "Gel character", "M-unit": "soft, elastic", "G-unit": "stiff, brittle, porous"},
    ])
    table_block(mg, "t2_M_vs_G", TAB, "M-units versus G-units at a glance")

    pld = None
    pr = st.session_state.get("pores")
    if pr and pr.get("relaxed"):
        pld = pr["relaxed"]["pld"]
    elif pr and pr.get("rigid"):
        pld = pr["rigid"]["pld"]
    gp = g_prime_estimate(2.0)
    xi = (mesh_size_rubber(gp), mesh_size_canal_peppas(2.0, gp)["xi_nm"])
    rows = scale_ladder(pld, xi, 2.2, st.session_state.get("particle_A"))
    f = Fig("t2_size_scale_ladder", "From ions to microplastics: the size scales that matter (log scale)",
            "Size (metres, log scale)", "", logx=True, legend=True, height=560)
    kinds = {"reference": ("Reference objects", "#999999"), "gel": ("Alginate gel", "#0072B2"),
             "bead": ("Alginate beads", "#009E73"), "plastic": ("Plastic size classes", "#E69F00"),
             "particle": ("Your particle", "#D55E00")}
    for kind, (lab, col) in kinds.items():
        pts = [(r["Size (m)"], i, r["Object"]) for i, r in enumerate(rows) if r["Kind"] == kind]
        if pts:
            f.scatter([p[0] for p in pts], [p[1] for p in pts], name=lab, color=col, size=13,
                      text=[p[2] for p in pts], textpos="middle right")
    f.d["yrange"] = [-1, len(rows) + 0.5]
    f.d["hide_yticks"] = True
    fig_block(f, TAB, "Each dot is one object; one step on the x-axis is ×10. Gel mesh estimates use the theory in Tab 7.")
    interpretation(
        "The ladder shows the central scale problem of the project: alginate pores are nanometres wide, while most "
        "microplastics are micrometres to millimetres. Particles larger than the gel mesh cannot pass <i>through</i> the "
        "gel – they are blocked at its surface or trapped if the gel forms around them. Nanoplastics are close to the "
        "mesh size, so they are the hardest case and the most interesting target for the molecular model.")
