"""Multiscale bridge (molecule → gel → bead) and wet-lab calculators.

Mesh-size estimates
-------------------
1. Rubber elasticity (affine network):      ξ = (k_B T / G′)^(1/3)
2. Canal–Peppas (swollen network):          ξ = v₂ₛ^(−1/3) · (2 C_n M_c / M_r)^(1/2) · l
   with M_c from G′ = c R T / M_c  (c = polymer mass concentration).
Both are standard first-order estimates; reported alginate mesh sizes span
roughly 5–200 nm depending on concentration, G-content and Ca²⁺ dose.
"""
from __future__ import annotations

import math

import numpy as np

from .constants import G_ACC, K_B, MW_NA_URONATE, R_GAS

NEEDLE_OD_MM = {"16 G": 1.65, "18 G": 1.27, "20 G": 0.91, "21 G": 0.82, "22 G": 0.72, "23 G": 0.64,
                "25 G": 0.51, "27 G": 0.41, "30 G": 0.31}
CACL2_MW = {"anhydrous CaCl₂": 110.98, "CaCl₂·2H₂O": 147.01, "CaCl₂·6H₂O": 219.08}


def g_prime_estimate(conc_pct: float, g_ref_pa: float = 10_000.0, c_ref_pct: float = 2.0, exponent: float = 2.0) -> float:
    """Shear modulus scaling G′ ∝ c² (semi-dilute gels). Reference: ~10 kPa at 2 % w/v (high-G alginate)."""
    return g_ref_pa * (max(conc_pct, 1e-6) / c_ref_pct) ** exponent


def mesh_size_rubber(g_prime_pa: float, temp_k: float = 298.15) -> float:
    """ξ in nm."""
    return (K_B * temp_k / max(g_prime_pa, 1e-9)) ** (1 / 3) * 1e9


def mesh_size_canal_peppas(conc_pct: float, g_prime_pa: float, temp_k: float = 298.15, cn: float = 21.1,
                           l_nm: float = 0.515, mr: float = MW_NA_URONATE, rho_polymer: float = 1.6) -> dict:
    c = conc_pct * 10.0                          # % w/v → kg/m³
    mc = c * R_GAS * temp_k / max(g_prime_pa, 1e-9) * 1000  # g/mol
    v2s = (conc_pct / 100) / rho_polymer         # polymer volume fraction
    xi = v2s ** (-1 / 3) * math.sqrt(2 * cn * mc / mr) * l_nm
    return {"Mc (g/mol)": mc, "v2s": v2s, "xi_nm": xi}


def mesh_size_table(conc_pct: float, g_prime_pa: float, temp_k: float = 298.15) -> list[dict]:
    rub = mesh_size_rubber(g_prime_pa, temp_k)
    cp = mesh_size_canal_peppas(conc_pct, g_prime_pa, temp_k)
    return [
        {"Method": "Rubber elasticity ξ = (k_BT/G′)^⅓", "Mesh size (nm)": round(rub, 2),
         "Mesh size (Å)": round(rub * 10, 1), "Inputs": f"G′ = {g_prime_pa:,.0f} Pa, T = {temp_k:.1f} K"},
        {"Method": "Canal–Peppas (swollen network)", "Mesh size (nm)": round(cp["xi_nm"], 2),
         "Mesh size (Å)": round(cp["xi_nm"] * 10, 1),
         "Inputs": f"c = {conc_pct:.2f} % w/v, Mc = {cp['Mc (g/mol)']:,.0f} g/mol, v₂ₛ = {cp['v2s']:.4f}, Cn = 21.1, l = 0.515 nm"},
    ]


def bead_diameter_tate(needle: str, gamma: float = 0.068, rho: float = 1010.0, harkins_brown: float = 0.60,
                       shrink: float = 0.85) -> dict:
    """Drop size from Tate's law with Harkins–Brown correction; gel shrinkage on crosslinking."""
    r = NEEDLE_OD_MM[needle] / 2 / 1000
    m = 2 * math.pi * r * gamma / G_ACC * harkins_brown
    v = m / rho
    d_drop = (6 * v / math.pi) ** (1 / 3)
    return {"Drop volume (µL)": v * 1e9, "Drop diameter (mm)": d_drop * 1000,
            "Gel bead diameter (mm)": d_drop * 1000 * shrink}


def interstitial_throat_um(bead_mm: float) -> float:
    """Largest sphere passing through a triangular gap of three touching beads: (2/√3 − 1)·d."""
    return (2 / math.sqrt(3) - 1) * bead_mm * 1000


def gelation_time_min(bead_mm: float, d_ca: float = 1.0e-9) -> float:
    """Time for the Ca²⁺ front to reach the bead centre, t ≈ R²/(2D)."""
    rr = bead_mm / 2 / 1000
    return rr ** 2 / (2 * d_ca) / 60


def solution_recipe(alg_pct: float, alg_ml: float, ca_mm: float, ca_ml: float, salt: str, frac_g: float) -> dict:
    alg_g = alg_pct / 100 * alg_ml
    ca_g = ca_mm / 1000 * ca_ml / 1000 * CACL2_MW[salt]
    mol_res = alg_g / MW_NA_URONATE
    mol_g = mol_res * frac_g
    mol_ca = ca_mm / 1000 * ca_ml / 1000
    return {
        "Sodium alginate to weigh (g)": alg_g,
        f"{salt} to weigh (g)": ca_g,
        "Uronate residues (mmol)": mol_res * 1000,
        "G residues (mmol)": mol_g * 1000,
        "Ca²⁺ in bath (mmol)": mol_ca * 1000,
        "Bath excess R = Ca²⁺ / G": mol_ca / mol_g if mol_g > 0 else float("nan"),
        "Ca²⁺ needed for egg-box saturation, R = 0.25 (mmol)": 0.25 * mol_g * 1000,
        "Ca²⁺ needed for full charge neutralisation (mmol)": 0.5 * mol_res * 1000,
    }


def scale_ladder(pld_A: float | None, xi_nm: tuple[float, float], bead_mm: float, particle_A: float | None) -> list[dict]:
    rows = [
        {"Object": "Water molecule", "Size (m)": 2.8e-10, "Kind": "reference"},
        {"Object": "Ca²⁺ ion (diameter)", "Size (m)": 2.0e-10, "Kind": "reference"},
        {"Object": "Alginate ribbon width", "Size (m)": 1.0e-9, "Kind": "gel"},
    ]
    if pld_A:
        rows.append({"Object": "Model pore-limiting diameter (this run)", "Size (m)": pld_A * 1e-10, "Kind": "gel"})
    rows += [
        {"Object": "Gel mesh size ξ (low estimate)", "Size (m)": xi_nm[0] * 1e-9, "Kind": "gel"},
        {"Object": "Gel mesh size ξ (high estimate)", "Size (m)": xi_nm[1] * 1e-9, "Kind": "gel"},
        {"Object": "Nanoplastic upper limit", "Size (m)": 1e-6, "Kind": "plastic"},
        {"Object": "Bacterium (E. coli)", "Size (m)": 2e-6, "Kind": "reference"},
        {"Object": "Gap between packed beads", "Size (m)": interstitial_throat_um(bead_mm) * 1e-6, "Kind": "bead"},
        {"Object": "Alginate bead", "Size (m)": bead_mm * 1e-3, "Kind": "bead"},
        {"Object": "Microplastic upper limit", "Size (m)": 5e-3, "Kind": "plastic"},
    ]
    if particle_A:
        rows.append({"Object": "Your particle", "Size (m)": particle_A * 1e-10, "Kind": "particle"})
    rows.sort(key=lambda r: r["Size (m)"])
    return rows


def multiscale_verdicts(particle_A: float, pld_A: float | None, xi_nm: tuple[float, float], bead_mm: float) -> list[dict]:
    d_nm = particle_A / 10
    out = []
    if pld_A:
        out.append({"Scale": "Molecular network (model)", "Barrier size": f"{pld_A:.1f} Å",
                    "Particle": f"{particle_A:,.1f} Å",
                    "Verdict": "Blocked" if particle_A > pld_A else "Passes"})
    lo, hi = xi_nm
    v = "Blocked" if d_nm > hi else ("Hindered (partly enters)" if d_nm > lo else "Diffuses through")
    out.append({"Scale": "Bulk gel mesh (theory)", "Barrier size": f"{lo:.1f}–{hi:.1f} nm", "Particle": f"{d_nm:,.2f} nm",
                "Verdict": v})
    thr = interstitial_throat_um(bead_mm)
    d_um = particle_A / 1e4
    out.append({"Scale": "Packed bead bed (gaps)", "Barrier size": f"{thr:,.0f} µm", "Particle": f"{d_um:,.3f} µm",
                "Verdict": "Strained (blocked)" if d_um > thr else "Passes through gaps unless adsorbed/entrapped"})
    return out


def capture_curve_mesh(diam_nm: np.ndarray, xi_nm: float, phi: float = 0.0125, fiber_nm: float = 1.0) -> dict:
    """Ogston partition (fraction of gel volume accessible) vs particle size."""
    rs, rf = diam_nm / 2, fiber_nm / 2
    og = np.exp(-phi * (1 + rs / rf) ** 2)
    lam = np.clip(diam_nm / xi_nm, 0, 1)
    ren = (1 - lam) ** 2
    return {"ogston": og, "renkin": ren}
