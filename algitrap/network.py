"""Hydrogel network construction: replicate an alginate chain into a woven
mesh or an egg-box bundle and inject Ca²⁺ ions between facing carboxylate
groups of neighbouring chains."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .alginate import AlginateChain, carboxylate_sites
from .constants import EGG_BOX_REPEAT, N_A
from .geom import align_to_axes, rotation_matrix
from .molio import Molecule, make_molecule

CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWYabcdefghijklmnopqrstuvwxyz0123456789"
CA_CHAIN = "Z"


@dataclass
class NetworkParams:
    topology: str = "Woven mesh"        # or "Egg-box bundle"
    grid: int = 5                       # chains per direction (mesh) or N for N×N bundle
    spacing: float = 18.0               # centre-to-centre distance of parallel chains (Å)
    layers: int = 2                     # stacked layers (mesh only)
    gap: float = 4.5                    # carboxylate-to-carboxylate distance across a junction (Å)
    ca_spacing: float = EGG_BOX_REPEAT  # minimum Ca–Ca distance (Å)
    max_bridge: float = 8.0             # max site–site distance that one Ca²⁺ may bridge (Å)
    facing_deg: float = 80.0            # carboxylates must point towards each other within this angle
    g_only: bool = False                # strict egg-box: only G–G pairs get Ca
    g_weight: float = 2.0               # preference factor for each G in a pair

    def as_table(self) -> list[dict]:
        lab = {"topology": "Network topology", "grid": "Grid size N", "spacing": "Chain spacing (Å)",
               "layers": "Layers (mesh)", "gap": "Junction gap (Å)", "ca_spacing": "Minimum Ca–Ca spacing (Å)",
               "max_bridge": "Max bridging distance (Å)", "facing_deg": "Facing tolerance (°)",
               "g_only": "Ca²⁺ only at G–G pairs", "g_weight": "G preference factor"}
        return [{"Parameter": lab[k], "Value": v} for k, v in asdict(self).items()]


@dataclass
class Network:
    mol: Molecule
    params: NetworkParams
    atom_chain: np.ndarray          # chain index per atom (-1 for ions)
    atom_res: np.ndarray            # global residue index per atom (-1 for ions)
    residues: list[dict]            # global residue table
    sites: dict                     # arrays: pos, normal, chain, res, type
    ca: list[dict]                  # one row per Ca²⁺
    region: dict                    # box for pore analysis: lo, hi, slab_lo, slab_hi
    n_chains: int
    chain_len: float
    layer_sep: float
    notes: list[str] = field(default_factory=list)


def _orient_sites(chain: AlginateChain, X: np.ndarray, normal_axis: int) -> np.ndarray:
    """Rotate X about the x axis so carboxylate normals point along ±normal_axis."""
    sites = carboxylate_sites(chain, X)
    if not sites:
        return X
    N = np.array([s["normal"] for s in sites])
    best, best_a = -1, 0.0
    for a in np.arange(0, 180, 3.0):
        R = rotation_matrix([1, 0, 0], a)
        val = float(((N @ R.T)[:, normal_axis] ** 2).sum())
        if val > best:
            best, best_a = val, a
    R = rotation_matrix([1, 0, 0], best_a)
    return X @ R.T


def _chain_frame(chain: AlginateChain) -> np.ndarray:
    X, _, _ = align_to_axes(chain.mol.coords, 0)
    X = _orient_sites(chain, X, 2)
    return X - X.mean(0)


def build_network(chain: AlginateChain, p: NetworkParams, progress=None) -> Network:
    T = _chain_frame(chain)                       # long axis = x, carboxylates ± z
    heavy = chain.mol.heavy_mask()
    sites_t = carboxylate_sites(chain, T)
    if not sites_t:
        raise ValueError("The chain has no carboxylate groups – Ca²⁺ cannot be placed.")
    sp = np.array([s["pos"] for s in sites_t])
    sn = np.array([s["normal"] for s in sites_t])
    up = sn[:, 2] > 0.3
    dn = sn[:, 2] < -0.3
    r_up = float(np.median(sp[up, 2])) if up.any() else float(np.abs(sp[:, 2]).max())
    r_dn = float(np.median(-sp[dn, 2])) if dn.any() else float(np.abs(sp[:, 2]).max())
    length = float(np.ptp(T[heavy, 0]) + 3.4)
    notes = []
    placements = []  # (R, t, label)
    N = int(p.grid)
    if p.topology.startswith("Woven"):
        D = r_up + r_dn + p.gap
        offs = (np.arange(N) - (N - 1) / 2) * p.spacing
        span = N * p.spacing
        copies = max(1, int(math.ceil((span + 2.0) / (length + 2.0))))
        along = (np.arange(copies) - (copies - 1) / 2) * (length + 2.0)
        for k in range(int(p.layers)):
            R = np.eye(3) if k % 2 == 0 else rotation_matrix([0, 0, 1], 90)
            for j, u in enumerate(offs):
                for c, a in enumerate(along):
                    if k % 2 == 0:
                        t = np.array([a, u, k * D])
                    else:
                        t = np.array([u, a, k * D])
                    placements.append((R, t, f"L{k + 1}-line{j + 1}-seg{c + 1}"))
        if copies > 1:
            notes.append(f"Chain ({length:.0f} Å) is shorter than a mesh line ({span:.0f} Å): "
                         f"each line is tiled with {copies} chain copies.")
        layer_sep = D
        lo_xy = offs.min()
        hi_xy = offs.max()
        region_xy = (np.array([lo_xy, lo_xy]), np.array([hi_xy, hi_xy]))
    else:
        # Egg-box bundle: chains along z, carboxylates facing ±x
        Rz = rotation_matrix([0, 1, 0], -90)       # x → z
        T2 = T @ Rz.T
        # after x→z, carboxylate normals (±z) became ±x
        sx = r_up + r_dn + p.gap
        sy = max(p.spacing, 6.0)
        offx = (np.arange(N) - (N - 1) / 2) * sx
        offy = (np.arange(N) - (N - 1) / 2) * sy
        for i, x in enumerate(offx):
            for j, y in enumerate(offy):
                R = Rz if i % 2 == 0 else rotation_matrix([0, 0, 1], 180) @ Rz
                placements.append((R, np.array([x, y, 0.0]), f"col{i + 1}-row{j + 1}"))
        layer_sep = sx
        region_xy = (np.array([offx.min(), offy.min()]), np.array([offx.max(), offy.max()]))
        notes.append(f"Junction (x) spacing set to {sx:.1f} Å from the gap; sheet (y) spacing {sy:.1f} Å.")
    # ------------------------------------------------------------------ atoms
    mols, atom_chain, atom_res, residues = [], [], [], []
    site_pos, site_n, site_chain, site_res, site_type, site_o = [], [], [], [], [], []
    nres = len(chain.residues)
    base = chain.mol
    offset = 0
    for ci, (R, t, label) in enumerate(placements):
        X = T @ R.T + t
        m = base.with_coords(X)
        cid = CHAIN_IDS[ci % len(CHAIN_IDS)]
        m.chains = np.array([cid] * m.n, dtype=object)
        m.segids = np.array([f"C{ci + 1:03d}"[-4:]] * m.n, dtype=object)
        mols.append(m)
        atom_chain.append(np.full(m.n, ci))
        atom_res.append(np.where(chain.atom_res >= 0, chain.atom_res + ci * nres, -1))
        for k, r in enumerate(chain.residues):
            residues.append({"chain": ci, "res": k, "type": r["type"], "label": label})
        for s in sites_t:
            site_pos.append(R @ s["pos"] + t)
            site_n.append(R @ s["normal"])
            site_chain.append(ci)
            site_res.append(ci * nres + s["res"])
            site_type.append(s["type"])
            site_o.append((s["O"][0] + offset, s["O"][1] + offset))
        offset += m.n
        if progress:
            progress(0.5 * (ci + 1) / len(placements), f"Placed chain {ci + 1}/{len(placements)}")
    site_pos, site_n = np.array(site_pos), np.array(site_n)
    site_chain, site_type = np.array(site_chain), np.array(site_type)
    # ------------------------------------------------------------- Ca placement
    tree = cKDTree(site_pos)
    pairs = tree.query_pairs(p.max_bridge, output_type="ndarray")
    cand = []
    cosf = math.cos(math.radians(p.facing_deg))
    for a, b in pairs:
        if site_chain[a] == site_chain[b]:
            continue
        v = site_pos[b] - site_pos[a]
        d = float(np.linalg.norm(v))
        u = v / max(d, 1e-9)
        if np.dot(site_n[a], u) < cosf or np.dot(site_n[b], -u) < cosf:
            continue
        ng = int(site_type[a] == "G") + int(site_type[b] == "G")
        if p.g_only and ng < 2:
            continue
        cand.append((d / (p.g_weight ** ng), d, a, b))
    cand.sort()
    used = np.zeros(len(site_pos), bool)
    ca_rows, ca_pos = [], []
    for score, d, a, b in cand:
        if used[a] or used[b]:
            continue
        pos = 0.5 * (site_pos[a] + site_pos[b])
        if ca_pos and np.min(np.linalg.norm(np.array(ca_pos) - pos, axis=1)) < p.ca_spacing:
            continue
        used[a] = used[b] = True
        ca_pos.append(pos)
        ta, tb = site_type[a], site_type[b]
        ca_rows.append({"Ca": len(ca_rows) + 1, "Chain A": int(site_chain[a]) + 1, "Residue A": ta,
                        "Chain B": int(site_chain[b]) + 1, "Residue B": tb,
                        "Junction type": "".join(sorted(ta + tb, reverse=True)),
                        "Site–site distance (Å)": round(d, 2), "x": pos[0], "y": pos[1], "z": pos[2],
                        "_site_a": int(a), "_site_b": int(b)})
    if ca_pos:
        n_ca = len(ca_pos)
        ca_mol = make_molecule(["CA"] * n_ca, np.array(ca_pos), names=["CA"] * n_ca, resnames=["CA"] * n_ca,
                               resids=np.arange(1, n_ca + 1), chains=[CA_CHAIN] * n_ca, charges=[2] * n_ca)
        ca_mol.segids = np.array(["ION"] * n_ca, dtype=object)
        mols.append(ca_mol)
        atom_chain.append(np.full(n_ca, -1))
        atom_res.append(np.full(n_ca, -1))
    else:
        notes.append("No facing carboxylate pairs were found within the bridging distance – no Ca²⁺ placed. "
                     "Reduce the junction gap or increase the max bridging distance.")
    mol = Molecule.concat(mols, title=f"Crosslinked_Hydrogel_Net_{N}x{N}")
    ac = np.concatenate(atom_chain)
    ar = np.concatenate(atom_res)
    zlo, zhi = float(mol.coords[:, 2].min()), float(mol.coords[:, 2].max())
    region = {"lo": np.array([region_xy[0][0], region_xy[0][1], zlo - 4.0]),
              "hi": np.array([region_xy[1][0], region_xy[1][1], zhi + 4.0]),
              "slab_lo": zlo, "slab_hi": zhi, "axis": 2}
    sites = {"pos": site_pos, "normal": site_n, "chain": site_chain, "res": np.array(site_res),
             "type": site_type, "O": np.array(site_o), "used": used}
    return Network(mol, p, ac, ar, residues, sites, ca_rows, region, len(placements), length, layer_sep, notes)


def network_with_coords(net: Network, X: np.ndarray) -> Network:
    """Shallow copy of a network with new coordinates (Ca rows updated)."""
    m = net.mol.with_coords(X)
    ca_idx = np.flatnonzero(m.elements == "CA")
    rows = []
    for r, k in zip(net.ca, ca_idx):
        r2 = dict(r)
        r2["x"], r2["y"], r2["z"] = X[k]
        rows.append(r2)
    zlo, zhi = float(X[:, 2].min()), float(X[:, 2].max())
    region = dict(net.region)
    region["lo"] = np.array([region["lo"][0], region["lo"][1], zlo - 4.0])
    region["hi"] = np.array([region["hi"][0], region["hi"][1], zhi + 4.0])
    region["slab_lo"], region["slab_hi"] = zlo, zhi
    return Network(m, net.params, net.atom_chain, net.atom_res, net.residues, net.sites, rows, region,
                   net.n_chains, net.chain_len, net.layer_sep, list(net.notes))


def network_stats(net: Network) -> dict:
    m = net.mol
    n_ca = len(net.ca)
    types = [r["type"] for r in net.residues]
    nG = types.count("G")
    nres = len(types)
    poly = net.atom_chain >= 0
    X = m.coords
    box = net.region["hi"] - net.region["lo"]
    slab_vol = float(box[0] * box[1] * max(net.region["slab_hi"] - net.region["slab_lo"], 1.0))  # Å^3
    jt = {}
    for r in net.ca:
        jt[r["Junction type"]] = jt.get(r["Junction type"], 0) + 1
    ca_xyz = X[m.elements == "CA"]
    nn = float("nan")
    if len(ca_xyz) > 1:
        d, _ = cKDTree(ca_xyz).query(ca_xyz, k=2)
        nn = float(np.mean(d[:, 1]))
    chains_with = len({r["Chain A"] for r in net.ca} | {r["Chain B"] for r in net.ca})
    return {
        "Topology": net.params.topology,
        "Chains": net.n_chains,
        "Residues": nres,
        "G residues": nG,
        "Atoms (incl. Ca²⁺)": m.n,
        "Ca²⁺ ions": n_ca,
        "R = Ca²⁺ / G (molar)": round(n_ca / nG, 3) if nG else float("nan"),
        "Ca²⁺ per residue": round(n_ca / nres, 3) if nres else float("nan"),
        "Charge neutralisation (%)": round(100 * 2 * n_ca / nres, 1) if nres else float("nan"),
        "G–G junctions": jt.get("GG", 0),
        "G–M junctions": jt.get("GM", 0),
        "M–M junctions": jt.get("MM", 0),
        "Chains crosslinked": chains_with,
        "Mean Ca–Ca nearest-neighbour (Å)": round(nn, 2) if not math.isnan(nn) else float("nan"),
        "Crosslink density (mmol/L of slab)": round(n_ca / (slab_vol * 1e-27) / N_A * 1e3, 1) if slab_vol > 0 else float("nan"),
        "Layer / junction separation (Å)": round(net.layer_sep, 2),
        "Chain length (Å)": round(net.chain_len, 1),
        "Analysis box x × y × z (Å)": " × ".join(f"{v:.0f}" for v in box),
        "Polymer heavy atoms": int((poly & (m.elements != "H")).sum()),
    }


def ca_nn_distances(X: np.ndarray, elements: np.ndarray) -> np.ndarray:
    ca = X[elements == "CA"]
    if len(ca) < 2:
        return np.zeros(0)
    d, _ = cKDTree(ca).query(ca, k=2)
    return d[:, 1]


def place_particle(net: Network, particle: Molecule, best_dir=None, height: float = 10.0) -> Molecule:
    """Return a copy of `particle` centred above the mesh, smallest cross-section facing the gel."""
    X = particle.coords - particle.coords.mean(0)
    if best_dir is not None:
        u = np.asarray(best_dir, float)
        u = u / np.linalg.norm(u)
        z = np.array([0, 0, 1.0])
        v = np.cross(u, z)
        s = np.linalg.norm(v)
        if s > 1e-6:
            ang = math.degrees(math.atan2(s, np.dot(u, z)))
            X = X @ rotation_matrix(v, ang).T
    lo, hi = net.region["lo"], net.region["hi"]
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, net.region["slab_hi"] + height + np.ptp(X[:, 2]) / 2])
    m = particle.with_coords(X + centre)
    m.chains = np.array(["P"] * m.n, dtype=object)
    m.resnames = np.array(["MPL"] * m.n, dtype=object)
    return m
