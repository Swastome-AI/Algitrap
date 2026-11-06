"""Shared constants for AlgiTrap.

All lengths are in Angstrom (Å), energies in kcal/mol, masses in g/mol (amu)
unless a name says otherwise.
"""

APP_NAME = "AlgiTrap"
APP_TAGLINE = "Alginate Hydrogel Microplastic Capture & Photodegradation Studio"
APP_VERSION = "1.0.0"

FOOTER_LINES = [
    f"{APP_NAME} is actively maintained and developed by Swastome-AI under Swastome Private Limited.",
    "© 2026 Swastome Private Limited. All rights reserved. Swastome and the Swastome logo are the property of "
    "Swastome Private Limited. No part of this app may be copied, modified or distributed without written "
    "permission. Email management@swastome.com for collaboration and licensing.",
]

# ----------------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------------
K_B = 1.380649e-23          # J/K
N_A = 6.02214076e23         # 1/mol
R_GAS = 8.314462618         # J/(mol K)
R_KCAL = 0.0019872043       # kcal/(mol K)
COULOMB_KCAL = 332.0637     # kcal Å /(mol e^2)
H_PLANCK = 6.62607015e-34   # J s
C_LIGHT = 2.99792458e8      # m/s
G_ACC = 9.80665             # m/s^2

# ----------------------------------------------------------------------------
# Atomic data
# ----------------------------------------------------------------------------
MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "NA": 22.990,
    "MG": 24.305, "P": 30.974, "S": 32.06, "CL": 35.45, "K": 39.098, "CA": 40.078,
    "BR": 79.904, "ZN": 65.38, "FE": 55.845, "I": 126.90, "SI": 28.085, "B": 10.81,
}

# Bondi van der Waals radii (Å); ions use Shannon ionic radii (CN 6) because
# the bare ion is what occupies space inside the gel.
VDW_RADII = {
    "H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80, "S": 1.80,
    "CL": 1.75, "BR": 1.85, "I": 1.98, "SI": 2.10, "B": 1.92,
    "NA": 1.02, "K": 1.38, "MG": 0.72, "CA": 1.00, "ZN": 0.74, "FE": 0.78,
}

# Covalent radii (Å) for distance-based bond perception (Cordero 2008).
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07, "S": 1.05,
    "CL": 1.02, "BR": 1.20, "I": 1.39, "SI": 1.11, "B": 0.84,
}

IONS = {"NA", "K", "CA", "MG", "ZN", "FE", "CL"}  # CL treated as ion only when isolated
METAL_IONS = {"NA", "K", "CA", "MG", "ZN", "FE"}

ELEMENT_COLORS_3DMOL = {"C": "0x909090", "O": "0xff3030"}

# ----------------------------------------------------------------------------
# Alginate chemistry
# ----------------------------------------------------------------------------
# PDB chemical-component codes used for alginate residues.
#   BEM = beta-D-mannopyranuronic acid (M unit)
#   LGU = alpha-L-gulopyranuronic acid (G unit)
M_RESNAMES = {"BEM", "MAV", "MAN", "M", "MNA", "BMA_U", "MUA"}
G_RESNAMES = {"LGU", "GUL", "G", "GLA_U", "GUA"}
RES_M = "BEM"
RES_G = "LGU"
RES_CA = "CA"

# Residue masses (sodium salt repeat unit C6H7O6Na and acid form C6H8O6)
MW_NA_URONATE = 198.11
MW_H_URONATE = 176.12
MW_CA_HALF_URONATE = 195.15  # C6H7O6·½Ca

# Fibre-diffraction helix rises (Å per residue) for the 2-fold helices
RISE_POLY_M = 5.17   # poly-M: 10.35 Å repeat / 2 residues (flat ribbon)
RISE_POLY_G = 4.35   # poly-G: 8.70 Å repeat / 2 residues (buckled ribbon)
EGG_BOX_REPEAT = 8.70

# ----------------------------------------------------------------------------
# Microplastics
# ----------------------------------------------------------------------------
PLASTIC_INFO = {
    "PS": {
        "name": "Polystyrene", "monomer": "C8H8", "monomer_mw": 104.15,
        "density": 1.05, "uv254": "Strong (phenyl ring π→π*)",
        "ftir_ref_band": (1510.0, 1470.0), "ftir_ref_label": "aromatic C=C 1493 cm⁻¹",
    },
    "PE": {
        "name": "Polyethylene", "monomer": "C2H4", "monomer_mw": 28.05,
        "density": 0.94, "uv254": "Weak (only chromophoric defects absorb)",
        "ftir_ref_band": (1500.0, 1420.0), "ftir_ref_label": "CH₂ scissoring 1463 cm⁻¹",
    },
    "PP": {
        "name": "Polypropylene", "monomer": "C3H6", "monomer_mw": 42.08,
        "density": 0.90, "uv254": "Weak (tertiary C–H is photo-oxidation prone)",
        "ftir_ref_band": (1490.0, 1420.0), "ftir_ref_label": "CH₃/CH₂ bending 1456 cm⁻¹",
    },
    "Other": {
        "name": "Other / unknown", "monomer": "—", "monomer_mw": float("nan"),
        "density": 1.0, "uv254": "Unknown",
        "ftir_ref_band": (1500.0, 1420.0), "ftir_ref_label": "CH₂ bending 1460 cm⁻¹",
    },
}

# ----------------------------------------------------------------------------
# Plot style (colour-blind-safe Okabe–Ito order; validated in light mode)
# ----------------------------------------------------------------------------
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#999999"]
COLOR_M = "#0072B2"   # M units – blue
COLOR_G = "#E69F00"   # G units – orange
COLOR_CA = "#56B4E9"  # calcium – sky blue
COLOR_GOOD = "#009E73"
COLOR_BAD = "#D55E00"

FONT_TITLE = 21
FONT_AXIS = 17
FONT_TICK = 15
FONT_LEGEND = 14

EXPORT_WIDTH_IN = 10.0
EXPORT_HEIGHT_IN = 7.0
EXPORT_DPI = 300
