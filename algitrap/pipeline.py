"""UI-independent workflow steps used by the Streamlit app (and tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .alginate import AlginateChain, analyze_alginate, build_chain, chain_metrics, standardize
from .constants import G_RESNAMES, M_RESNAMES
from .molio import (deprotonate_carboxylic_acids, generic_checks, load_structure, structure_stats, write_pdb,
                    write_sdf)
from .network import Network, NetworkParams, build_network, network_with_coords
from .plastics import build_pe, build_ps, identify_polymer, size_descriptors
from .pores import capture_verdict, grid_for, pore_limiting_diameter, pore_summary, transport_mc

DEFAULT_SEQUENCE = "GGGGMMGGGGMM"


# ---------------------------------------------------------------- inputs
def demo_alginate_file(seq: str = DEFAULT_SEQUENCE) -> tuple[str, bytes]:
    ch = build_chain(seq, include_h=True, title="demo_alginate_12mer")
    return "demo_alginate_12mer.pdb", write_pdb(ch.mol, remarks=[f"Alginate sequence {seq} (AlgiTrap demo)"]).encode()


def demo_plastic_file(kind: str = "PS", n: int = 6) -> tuple[str, bytes]:
    m = build_ps(n, "atactic") if kind == "PS" else build_pe(max(n, 2) * 2)
    return f"demo_{kind}_{m.title}.sdf", write_sdf(m).encode()


def load_alginate(name: str, data: bytes) -> dict:
    out = {"name": name, "ok": False}
    mol, info = load_structure(name, data)
    rows = generic_checks(mol, info)
    mol2, nfix = deprotonate_carboxylic_acids(mol)
    if nfix:
        rows.append({"Check": "Carboxyl ionisation (pH 7)", "Status": "INFO",
                     "Detail": f"{nfix} COOH groups converted to COO⁻ (alginate pKa ≈ 3.4–3.7)"})
    try:
        raw = analyze_alginate(mol2)
        chain = standardize(raw, title="single_chain")
        rows.append({"Check": "Uronate pyranose rings", "Status": "PASS",
                     "Detail": f"{len(chain.residues)} residues recognised"})
        rows.append({"Check": "M/G assignment from C5 chirality", "Status": "PASS",
                     "Detail": f"Sequence {chain.sequence} ({chain.sequence.count('G')} G, {chain.sequence.count('M')} M)"})
        nlink = len(chain.linkages)
        rows.append({"Check": "1→4 glycosidic linkages", "Status": "PASS" if nlink == len(chain.residues) - 1 else "WARN",
                     "Detail": f"{nlink} linkages for {len(chain.residues)} residues"})
        # residue-name consistency
        mism = 0
        named = 0
        for r in raw.residues:
            rn = str(mol2.resnames[r["atoms"]["C5"]]).upper()
            if rn in M_RESNAMES or rn in G_RESNAMES:
                named += 1
                if (rn in M_RESNAMES) != (r["type"] == "M"):
                    mism += 1
        if named:
            rows.append({"Check": "Residue names vs 3-D stereochemistry", "Status": "WARN" if mism else "PASS",
                         "Detail": f"{mism} of {named} residue labels disagree with the geometry" if mism
                         else f"All {named} labelled residues agree with their 3-D configuration"})
        for w in chain.warnings:
            rows.append({"Check": "Chain topology", "Status": "WARN", "Detail": w})
        out.update(chain=chain, ok=True)
    except Exception as e:  # noqa: BLE001
        rows.append({"Check": "Alginate recognition", "Status": "FAIL", "Detail": str(e)})
    out.update(mol=mol, info=info, checks=pd.DataFrame(rows), stats=structure_stats(mol))
    return out


def load_plastic(name: str, data: bytes) -> dict:
    mol, info = load_structure(name, data)
    rows = generic_checks(mol, info)
    ident = identify_polymer(mol)
    rows.append({"Check": "Polymer identification", "Status": "PASS" if ident["label"] in ("PS", "PE") else "WARN",
                 "Detail": f"{ident['name']} ({ident['confidence']} confidence) – " + "; ".join(ident["evidence"])})
    size = size_descriptors(mol)
    rows.append({"Check": "Size descriptors", "Status": "PASS",
                 "Detail": f"min cross-section {size['Min cross-section diameter (Å)']} Å, "
                           f"max dimension {size['Max dimension (Å)']} Å"})
    return {"name": name, "mol": mol, "info": info, "checks": pd.DataFrame(rows), "stats": structure_stats(mol),
            "ident": ident, "size": size, "ok": True}


# ---------------------------------------------------------------- pores
def pore_analysis(net: Network, X: np.ndarray | None = None, h: float = 0.6, pad: float = 10.0) -> dict:
    grid = grid_for(net.mol, net.region, X=X, h=h, pad=pad)
    res = pore_summary(grid)
    res["grid"] = grid
    return res


def capture_curve(grid, pld: float, psd: np.ndarray, sizes: np.ndarray, walkers: int = 200, steps: int = 2500,
                  seed: int = 0) -> pd.DataFrame:
    rows = []
    for d in sizes:
        v = capture_verdict(float(d), pld, psd)
        mc = transport_mc(grid, float(d) / 2, n_walkers=walkers, n_steps=steps, seed=seed, n_record=1)
        rows.append({"Particle diameter (Å)": float(d), "Accessible pore-volume fraction": v["Accessible pore-volume fraction"],
                     "MC transmission": mc["transmission"], "Capture efficiency (%)": 100 * (1 - mc["transmission"]),
                     "Size-exclusion capture": v["Size-exclusion capture"], "D_eff / D0 (Renkin)": v["D_eff / D0"]})
    return pd.DataFrame(rows)


def pore_dynamics(net: Network, frames: list, labels: list, h: float = 0.9, pad: float = 8.0, max_frames: int = 8) -> pd.DataFrame:
    idx = np.unique(np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round().astype(int))
    rows = []
    for k in idx:
        g = grid_for(net.mol, net.region, X=frames[k], h=h, pad=pad)
        s = g.field[:, :, g.slab_k[0]:g.slab_k[1] + 1]
        rows.append({"Frame": labels[k], "Frame index": int(k), "PLD (Å)": round(pore_limiting_diameter(g), 2),
                     "LCD (Å)": round(2 * float(s.max()), 2), "Porosity": round(float((s > 0).mean()), 4),
                     "Mean void-centre distance ×2 (Å)": round(2 * float(s[s > 0].mean()), 2)})
    return pd.DataFrame(rows)


def design_sweep(chain: AlginateChain, base: NetworkParams, spacings: list[float], h: float = 0.8,
                 progress=None) -> pd.DataFrame:
    from dataclasses import replace
    rows = []
    for k, s in enumerate(spacings):
        p = replace(base, spacing=float(s))
        net = build_network(chain, p)
        g = grid_for(net.mol, net.region, h=h, pad=8.0)
        rows.append({"Chain spacing (Å)": float(s), "PLD (Å)": round(pore_limiting_diameter(g), 2),
                     "Ca²⁺ ions": len(net.ca), "Chains": net.n_chains})
        if progress:
            progress((k + 1) / len(spacings), f"Spacing {s:.0f} Å")
    return pd.DataFrame(rows)


def critical_spacing(df: pd.DataFrame, d_particle: float) -> float | None:
    x, y = df["Chain spacing (Å)"].to_numpy(), df["PLD (Å)"].to_numpy()
    for i in range(len(x) - 1):
        if (y[i] - d_particle) * (y[i + 1] - d_particle) <= 0 and y[i + 1] != y[i]:
            return float(x[i] + (d_particle - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


def egg_box_score(coord_rows: list[dict], min_o: int = 4) -> float:
    if not coord_rows:
        return float("nan")
    return float(np.mean([(r["O within cutoff"] >= min_o) and (r["Chains bridged"] >= 2) for r in coord_rows]))


__all__ = ["DEFAULT_SEQUENCE", "demo_alginate_file", "demo_plastic_file", "load_alginate", "load_plastic",
           "pore_analysis", "capture_curve", "pore_dynamics", "design_sweep", "critical_spacing", "egg_box_score",
           "build_chain", "chain_metrics", "build_network", "network_with_coords", "NetworkParams"]
