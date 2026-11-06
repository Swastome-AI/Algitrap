# 🌊 AlgiTrap
**Alginate Hydrogel Microplastic Capture & Photodegradation Studio**

*AlgiTrap* = **Algi**nate + **Trap**: a seaweed-polymer net that traps microplastics so UV-C light can destroy them.
It takes an alginate chain (`.pdb`) and a microplastic oligomer (`.sdf`, polystyrene or polyethylene), builds the
chain → mesh → Ca²⁺ egg-box network, relaxes it with a force field, measures its pores, predicts whether the plastic
is trapped, plans the wet-lab gel, and analyses UV-C degradation data. Everything ends in a named PDF report and a ZIP.

Fast defaults suit a school science-fair project (results in seconds); the **Research (pro)** preset and the
advanced expanders expose the full parameters for professional use.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the browser tab, type **your name** at the top, then either upload files in Tab 1 or press
**⚡ Run the full pipeline with default settings** (demo structures are generated automatically).

**Streamlit Community Cloud:** push this folder to GitHub, create an app with `app.py` as the entry point. No system
packages are needed. RDKit installs from `requirements.txt`; OpenBabel is optional (commented out).

---

## The nine tabs

| Tab | What it does | Main outputs |
|---|---|---|
| 1 · Upload & Check | Parses `.pdb/.sdf/.mol/.xyz`, checks geometry/chemistry, recognises uronate rings, assigns **M/G from C5 chirality**, identifies PS/PE, computes particle size descriptors | Check tables (PASS/WARN/FAIL), file statistics, bond-length and composition plots, 3-D viewers |
| 2 · The Science | Alginate, M vs G, egg-box model, microplastic problem, workflow diagram | Size-scale ladder (log), M-vs-G table |
| 3 · Molecular Blueprint | Builds a 3-D chain from an M/G sequence (built-in fibre model, correct stereochemistry), from SMILES (RDKit ETKDGv3 + MMFF94/UFF) or uses the upload | NMR-style statistics (F_G, F_GG, N_G>1), φ/ψ torsions, spacing, `single_chain.pdb/.sdf` |
| 4 · Hydrogel Matrix | Woven mesh or N×N egg-box bundle; Ca²⁺ placed between facing carboxylates (G-preferred, min. Ca–Ca spacing) | R = Ca²⁺/G, junction types, node spacing, top view, `Crosslinked_Hydrogel_Net_NxN.pdb` |
| 5 · Thermodynamics | "Run Physics Simulation": L-BFGS energy minimisation (AlgiTrap-ENM, or OpenBabel UFF / RDKit UFF / MMFF94 when installed) + optional Langevin dynamics | Looping 3-D animation, energy & gradient convergence, Ca coordination, Ca–O pair distribution, egg-box score, `Perfect_Hydrogel_NxN.pdb`, trajectory PDB, GIF |
| 6 · Capture Prediction | Distance field → **PLD**, LCD, porosity, geometric PSD, pore centres; capture verdict (✅/⚠️), capture-efficiency curve, Renkin hindrance, Monte-Carlo transport **animation**, pore dynamics, spacing design curve, multiscale (molecule → gel → bead) check | Verdict banner, 8+ plots, tables |
| 7 · Wet-Lab Synthesis | Protocol, recipe calculator (masses, R ratio), bead size (Tate's law), cure time, mesh size theory | Printable protocol, plots, tables |
| 8 · UV Photodegradation | 254 nm dose maths, planning model with drop-downs (time, temperature, oxidant), fit **your** kinetics CSV (k, half-life, 95 % CI, Arrhenius Ea), FTIR carbonyl/hydroxyl/vinyl indices, microscope particle counting | Model and data plots, fit tables |
| 9 · Report & Downloads | Overall summary, auto-written key findings, story of the study, exports | PDF (landscape A4, your name, footer on every page), interactive HTML, plots ZIP, full analysis ZIP |

A **status strip** with the same file and result statistics appears at the top of every tab, and every tab starts with
a "What happens in this tab" block and ends with "How to interpret this".

---

## Plots and publication export
* Every plot is defined once (`algitrap/figs.py`) and drawn with **Plotly** on screen (toolbar, zoom, hover) and with
  **Matplotlib** for export, so downloads never depend on Kaleido/Chrome.
* White background, black text, grid lines on both axes, black frame, outside ticks, titles 21 px / axis 17 px /
  ticks 15 px, long titles wrap, value labels sit outside bars, light heat-map scales, **Okabe–Ito colour-blind-safe
  palette** (validated for colour-vision deficiency).
* Under each plot: **Download this plot (publication quality)** → PNG (300 dpi), TIFF (300 dpi), SVG or PDF.
  The Plotly camera icon saves a 3000 × 2100 px PNG. Tab 9 exports all plots at once (300 or 600 dpi).

---

## Methods in one page

**Chain builder.** M = β-D-mannopyranuronate (⁴C₁), G = α-L-gulopyranuronate (¹C₄); ideal chairs with axial/equatorial
substituents from the Haworth configuration; built-in CIP check (M: C5-S, C1-R; G: C5-R, C1-R). Residues are joined by
NeRF placement (C1–O4′–C4′ = 116.5°) with φ/ψ chosen by a clash-free search that reproduces the fibre-diffraction 2-fold
helices (poly-M 5.17 Å, poly-G 4.35 Å per residue).

**M/G recognition in any 3-D file.** Pyranose rings with a carboxyl-bearing C5 are located in the bond graph; the sign of
(O5 × C6)·C4 around C5 gives the configuration (S → M, R → G); 1→4 links order the sequence.

**Network.** Copies of the chain are placed as a woven mesh (alternating x/y layers) or an egg-box bundle. A Ca²⁺ is
inserted at the midpoint of two carboxylates from different chains when they lie within the bridging distance and face
each other, preferring G and enforcing the minimum Ca–Ca spacing (default 8.7 Å, the egg-box repeat).

**AlgiTrap-ENM force field (kcal/mol, Å).** Harmonic bonds, 1-3 and ring 1-4 distances (shape), weak 1-4 springs
(torsional stiffness), soft steric repulsion, Morse Ca–O binding (r₀ = 2.40 Å; deeper for carboxylate O) and
Debye–Hückel electrostatics (λ_D = 3.04/√I Å). Minimised with L-BFGS; Langevin dynamics with the BAOAB integrator.

**Pores.** Grid distance field to vdW surfaces (Ca²⁺ ionic radius); PLD by binary search on probe radius with
connected-component percolation along z; geometric PSD (largest covering sphere, Gelb–Gubbins); Renkin hindrance
Φ = (1 − λ)², K_d = 1 − 2.104λ + 2.09λ³ − 0.95λ⁵; Monte-Carlo biased random walk for breakthrough.

**Multiscale.** ξ = (k_BT/G′)^⅓ and Canal–Peppas ξ = v₂ₛ^(−⅓)(2C_nM_c/M_r)^½ l; Ogston partition; bead size by Tate's law
with Harkins–Brown correction; gap between packed beads (2/√3 − 1)·d; Ca²⁺ cure time R²/2D.

**UV.** Dose = irradiance × time; pseudo-first-order k with Arrhenius and saturating H₂O₂ enhancement (planning only);
curve_fit of first-order growth/decay/plateau/zero-order models with 95 % CIs; Arrhenius Ea; FTIR indices by the SAUB
approach (band area 1850–1650 cm⁻¹ over a polymer-specific reference band); Otsu segmentation for particle counts.

---

## Presets

| Setting | Quick (school demo) | Research (pro) |
|---|---|---|
| Chain length (residues) | 12 | 24 |
| Grid N × layers | 5 × 2 | 8 × 3 |
| Chain spacing | 16 Å | 18 Å |
| Minimisation iterations / frames | 250 / 16 | 1500 / 30 |
| Pore grid spacing | 0.6 Å | 0.4 Å |
| Monte-Carlo walkers / steps | 200 / 2500 | 600 / 6000 |
| Langevin steps | 800 | 3000 |

All values can be changed in the tabs; force-field and MD parameters are in the ⚙️ expanders of Tab 5.

---

## Limitations
The network is an idealised regular model; the ENM is implicit-solvent and qualitative; pores are geometric;
uploaded "microplastics" are oligomers (nanoplastic fragments) and the multiscale step bridges to real particle sizes
with theory; UV model defaults are illustrative and every conclusion about degradation must come from measured data.
Simulated demo data are labelled as such everywhere (including the report findings).

---

## Tests
```bash
python tests/test_core.py      # stereochemistry, helix rises, file round-trips, gradients, pores, UV fits, exports
python tests/smoke_app.py      # runs the whole Streamlit app headlessly with stubbed UI, pressing every button
```

## Project layout
```
app.py                    Streamlit entry point (header, presets, tabs, footer)
algitrap/
  molio.py  geom.py       file I/O, checks, geometry helpers
  alginate.py             residue templates, chain builder, M/G recognition, sequence statistics
  plastics.py             PE/PS builders, polymer identification, size descriptors
  network.py              mesh/bundle construction and Ca²⁺ placement
  forcefield.py           AlgiTrap-ENM, L-BFGS, Langevin, OpenBabel/RDKit bridges
  pores.py                distance field, PLD, PSD, capture physics, Monte-Carlo transport
  multiscale.py  uv.py    gel/bead theory, wet-lab calculators, UV kinetics, FTIR, image analysis
  figs.py  anim.py        figure system (Plotly + Matplotlib export), animations, GIF
  report.py               PDF / HTML / ZIP
  ui.py  state.py  pipeline.py  tab_*.py
sample_data/              example inputs (+ clearly labelled SIMULATED demo data)
tests/                    unit tests and headless app smoke test
```

---
AlgiTrap is actively maintained and developed by Swastome-AI under Swastome Private Limited.
© 2026 Swastome Private Limited. All rights reserved. Swastome and the Swastome logo are the property of Swastome
Private Limited. No part of this app may be copied, modified or distributed without written permission.
Email management@swastome.com for collaboration and licensing.
