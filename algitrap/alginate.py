"""Alginate chemistry: residue templates, 3-D chain builder, M/G recognition
from 3-D coordinates, sequence statistics and chain metrics.

Stereochemistry
---------------
* M = beta-D-mannopyranuronate in the 4C1 chair (O2 axial, other substituents equatorial)
* G = alpha-L-gulopyranuronate in the 1C4 chair (O1, O3, O4 axial; C6 and O2 equatorial)
G is the C5 epimer of M, so the two templates differ only at C5 (and the ring flip).
A built-in CIP check confirms C5 = S for M and C5 = R for G.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from .constants import (MW_H_URONATE, MW_NA_URONATE, RES_G, RES_M, RISE_POLY_G, RISE_POLY_M)
from .geom import align_to_axes, angle, dihedral, kabsch, nerf, tetra_pair, unit
from .molio import Molecule, make_molecule, perceive_bonds

# ---------------------------------------------------------------------------
# Residue templates
# ---------------------------------------------------------------------------
_SPEC = {
    "M": {"conf": "4C1", "up": {"O1", "O2", "O3", "C6"}},
    "G": {"conf": "1C4", "up": {"O1", "O2", "O3"}},
}
_RING = ["C1", "C2", "C3", "C4", "C5", "O5"]
_SUBST = {"C1": ("O1", "H1"), "C2": ("O2", "H2"), "C3": ("O3", "H3"), "C4": ("O4", "H4"), "C5": ("C6", "H5")}
BOND = {"CO": 1.42, "CC6": 1.52, "CH": 1.09, "OH": 0.97, "CO2": 1.26}
GLYC_ANGLE = 116.5
HEAVY_ORDER = ["C1", "C2", "C3", "C4", "C5", "O5", "C6", "O6A", "O6B", "O1", "O2", "O3", "O4"]
H_ORDER = ["H1", "H2", "H3", "H4", "H5", "HO1", "HO2", "HO3", "HO4"]
H_PARENT = {"H1": "C1", "H2": "C2", "H3": "C3", "H4": "C4", "H5": "C5",
            "HO1": "O1", "HO2": "O2", "HO3": "O3", "HO4": "O4"}
_INTRA_BONDS = [("C1", "C2", 1), ("C2", "C3", 1), ("C3", "C4", 1), ("C4", "C5", 1), ("C5", "O5", 1),
                ("O5", "C1", 1), ("C1", "O1", 1), ("C2", "O2", 1), ("C3", "O3", 1), ("C4", "O4", 1),
                ("C5", "C6", 1), ("C6", "O6A", 2), ("C6", "O6B", 1)] + [(p, h, 1) for h, p in H_PARENT.items()]


@lru_cache(maxsize=4)
def residue_template(kind: str) -> dict[str, np.ndarray]:
    """Ideal 3-D coordinates (dict atom → xyz) of an isolated M or G residue."""
    spec = _SPEC[kind]
    L = 1.50                     # mean ring bond
    r, z0 = 0.9545 * L, 0.149 * L  # chair with 111.5° ring angles
    sign = np.array([-1, 1, -1, 1, -1, 1], float)
    if spec["conf"] == "1C4":
        sign = -sign
    P = {}
    for k, nm in enumerate(_RING):
        th = np.radians(-60.0 * k)       # clockwise numbering seen from +z (Haworth "up" = +z)
        P[nm] = np.array([r * np.cos(th), r * np.sin(th), sign[k] * z0])
    for k, nm in enumerate(_RING[:5]):
        prev_ = _RING[(k - 1) % 6]
        next_ = _RING[(k + 1) % 6]
        d1, d2 = tetra_pair(P[nm], P[prev_], P[next_])
        up, down = (d1, d2) if d1[2] > d2[2] else (d2, d1)
        sub, hyd = _SUBST[nm]
        dsub = up if sub in spec["up"] else down
        dh = down if sub in spec["up"] else up
        P[sub] = P[nm] + dsub * (BOND["CC6"] if sub == "C6" else BOND["CO"])
        P[hyd] = P[nm] + dh * BOND["CH"]
    # carboxylate
    P["O6A"] = nerf(P["O5"], P["C5"], P["C6"], BOND["CO2"], 117.5, 0.0)
    P["O6B"] = nerf(P["O5"], P["C5"], P["C6"], BOND["CO2"], 117.5, 180.0)
    # hydroxyl hydrogens
    P["HO1"] = nerf(P["O5"], P["C1"], P["O1"], BOND["OH"], 108.5, 180.0)
    P["HO2"] = nerf(P["C1"], P["C2"], P["O2"], BOND["OH"], 108.5, 180.0)
    P["HO3"] = nerf(P["C2"], P["C3"], P["O3"], BOND["OH"], 108.5, 180.0)
    P["HO4"] = nerf(P["C3"], P["C4"], P["O4"], BOND["OH"], 108.5, 180.0)
    return P


def cip_c5(P: dict) -> str:
    """CIP label at C5 (O5 > C6(carboxyl) > C4 > H)."""
    a, b, c = P["O5"] - P["C5"], P["C6"] - P["C5"], P["C4"] - P["C5"]
    return "S" if np.dot(np.cross(a, b), c) > 0 else "R"


def cip_c1(P: dict) -> str:
    """CIP label at C1 for the free monosaccharide (O5 > O1 > C2 > H)."""
    a, b, c = P["O5"] - P["C1"], P["O1"] - P["C1"], P["C2"] - P["C1"]
    return "S" if np.dot(np.cross(a, b), c) > 0 else "R"


# ---------------------------------------------------------------------------
# Glycosidic linkage geometry
# ---------------------------------------------------------------------------
def _place_next(prev: dict, kind_next: str, phi: float, psi: float) -> tuple[dict, np.ndarray, np.ndarray]:
    T = residue_template(kind_next)
    o_link = prev["O1"]
    c4 = nerf(prev["O5"], prev["C1"], o_link, BOND["CO"], GLYC_ANGLE, phi)
    b43 = float(np.linalg.norm(T["C4"] - T["C3"]))
    a = angle(T["O4"], T["C4"], T["C3"])
    c3 = nerf(prev["C1"], o_link, c4, b43, a, psi)
    R, t = kabsch(np.array([T["O4"], T["C4"], T["C3"]]), np.array([o_link, c4, c3]))
    new = {k: R @ v + t for k, v in T.items()}
    return new, R, t


def _dimer_clash(A: dict, B: dict) -> float:
    """Steric clash between two linked residues (pairs > 3 bonds apart)."""
    # atoms excluded because they are within 3 bonds across the O1(A)=O4(B) link
    exA_near = {"C1", "O1", "HO1", "C2", "O5", "H1"}
    exB_near = {"O4", "HO4", "C4", "C3", "C5", "H4"}
    clash = 0.0
    keysA = [k for k in A if k not in ("O1", "HO1")]
    keysB = [k for k in B if k not in ("O4", "HO4")]
    XA = np.array([A[k] for k in keysA])
    XB = np.array([B[k] for k in keysB])
    d = np.linalg.norm(XA[:, None] - XB[None], axis=-1)
    isHA = np.array([k.startswith("H") for k in keysA])
    isHB = np.array([k.startswith("H") for k in keysB])
    sig = np.where(isHA[:, None] & isHB[None], 1.9, np.where(isHA[:, None] | isHB[None], 2.4, 2.9))
    # exclusions: C1(A) with near-B atoms; near-A atoms with C4(B)
    for i, ka in enumerate(keysA):
        for j, kb in enumerate(keysB):
            if (ka == "C1" and kb in exB_near) or (kb == "C4" and ka in exA_near):
                sig[i, j] = 0.0
    over = np.clip(sig - d, 0, None)
    clash = float((over ** 2).sum())
    return clash


def _helix_params(R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    rv = Rotation.from_matrix(R).as_rotvec()
    th = float(np.linalg.norm(rv))
    if th < 1e-6:
        return 0.0, float(np.linalg.norm(t))
    ax = rv / th
    return math.degrees(th), abs(float(np.dot(t, ax)))


def _centroid(P: dict) -> np.ndarray:
    return np.mean([P[k] for k in _RING], axis=0)


# Result of linkage_table() for the built-in templates (cached so start-up is instant).
LINKAGE_DEFAULTS = {"MM": (-106.0, 106.0), "GG": (-100.0, 100.0), "MG": (-120.0, 80.0), "GM": (-80.0, 120.0)}


def _compose(R1, t1, R2, t2):
    return R1 @ R2, R1 @ t2 + t1


@lru_cache(maxsize=1)
def linkage_table() -> dict[str, tuple[float, float]]:
    """Find (phi, psi) for MM, GG, MG and GM glycosidic linkages.

    * MM and GG: clash-free 2-fold helices closest to the fibre-diffraction
      rises (5.17 Å poly-M, 4.35 Å poly-G).
    * MG and GM: chosen jointly so that an M-G-M step behaves like a straight
      2-fold ribbon (M→M transform ≈ pure translation of 5.17 + 4.35 Å)."""
    grid = np.arange(-180, 180, 10.0)
    targets = {"MM": RISE_POLY_M, "GG": RISE_POLY_G}
    out = {}
    for key in ("MM", "GG"):
        A = residue_template(key[0])
        best = (1e9, 0.0, 0.0)

        def sc(phi, psi):
            B, R, t = _place_next(A, key[1], phi, psi)
            th, h = _helix_params(R, t)
            return _dimer_clash(A, B) + 0.002 * (th - 180.0) ** 2 + 3.0 * (h - targets[key]) ** 2

        for phi in grid:
            for psi in grid:
                s_ = sc(phi, psi)
                if s_ < best[0]:
                    best = (s_, phi, psi)
        _, p0, q0 = best
        for phi in np.arange(p0 - 8, p0 + 8.1, 2.0):
            for psi in np.arange(q0 - 8, q0 + 8.1, 2.0):
                s_ = sc(phi, psi)
                if s_ < best[0]:
                    best = (s_, phi, psi)
        out[key] = (float(best[1]), float(best[2]))
    # hetero linkages – keep the 40 least-clashing candidates of each, then pair them
    cands = {}
    for key in ("MG", "GM"):
        A = residue_template(key[0])
        rows = []
        for phi in grid:
            for psi in grid:
                B, R, t = _place_next(A, key[1], phi, psi)
                rows.append((_dimer_clash(A, B), phi, psi, R, t))
        rows.sort(key=lambda r: r[0])
        cands[key] = rows[:40]
    target = RISE_POLY_M + RISE_POLY_G
    best = (1e9, None, None)
    for c1, p1, q1, R1, t1 in cands["MG"]:
        for c2, p2, q2, R2, t2 in cands["GM"]:
            R, t = _compose(R1, t1, R2, t2)
            th, h = _helix_params(R, t)
            s_ = c1 + c2 + 0.002 * min(th, 360 - th) ** 2 + 3.0 * (h - target) ** 2
            if s_ < best[0]:
                best = (s_, (p1, q1), (p2, q2))
    out["MG"] = (float(best[1][0]), float(best[1][1]))
    out["GM"] = (float(best[2][0]), float(best[2][1]))
    return out


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------
@dataclass
class AlginateChain:
    mol: Molecule
    sequence: str
    residues: list[dict]          # per residue: type, atom index dict
    atom_res: np.ndarray          # residue index per atom (-1 none)
    source: str = "builder"
    warnings: list[str] = field(default_factory=list)
    linkages: list[dict] = field(default_factory=list)


def build_chain(sequence: str, include_h: bool = False, torsions: dict | None = None,
                title: str = "single_chain") -> AlginateChain:
    seq = "".join(ch for ch in sequence.upper() if ch in "MG")
    if not seq:
        raise ValueError("Sequence must contain at least one M or G.")
    tors = dict(LINKAGE_DEFAULTS)
    if torsions:
        tors.update(torsions)
    placed: list[dict] = []
    for k, kind in enumerate(seq):
        if k == 0:
            placed.append(dict(residue_template(kind)))
        else:
            phi, psi = tors[seq[k - 1] + kind]
            nxt, _, _ = _place_next(placed[-1], kind, phi, psi)
            placed.append(nxt)
    els, xyz, names, resn, resi, charges = [], [], [], [], [], []
    index: list[dict] = []
    n = len(seq)
    for k, (kind, P) in enumerate(zip(seq, placed)):
        keep = list(HEAVY_ORDER)
        if k < n - 1:
            keep.remove("O1")            # O1 replaced by O4 of next residue (glycosidic O)
        hs = list(H_ORDER) if include_h else []
        if include_h:
            if k < n - 1:
                hs.remove("HO1")
            if k > 0:
                hs.remove("HO4")
        idx = {}
        for nm in keep + hs:
            idx[nm] = len(els)
            els.append(nm[0] if not nm.startswith("H") else "H")
            xyz.append(P[nm])
            names.append(nm)
            resn.append(RES_M if kind == "M" else RES_G)
            resi.append(k + 1)
            charges.append(-1 if nm == "O6B" else 0)
        index.append({"type": kind, "atoms": idx})
    bonds, orders = [], []
    for k, rec in enumerate(index):
        idx = rec["atoms"]
        for a, b, o in _INTRA_BONDS:
            if a in idx and b in idx:
                bonds.append((idx[a], idx[b]))
                orders.append(o)
        if k < n - 1:
            bonds.append((idx["C1"], index[k + 1]["atoms"]["O4"]))
            orders.append(1)
    xyz = np.array(xyz)
    xyz_aligned, _, _ = align_to_axes(xyz, long_axis=0)
    mol = make_molecule(els, xyz_aligned, names=names, resnames=resn, resids=resi, chains=["A"] * len(els),
                        charges=charges, bonds=bonds, orders=orders, title=title)
    atom_res = np.array(resi, int) - 1
    chain = AlginateChain(mol, seq, index, atom_res, source="AlgiTrap fibre-model builder")
    chain.linkages = linkage_report(chain)
    return chain


# ---------------------------------------------------------------------------
# Recognition of alginate residues in arbitrary 3-D structures
# ---------------------------------------------------------------------------
def _find_pyranose_rings(mol: Molecule, adj) -> list[list[int]]:
    rings = []
    seen = set()
    for o in np.flatnonzero(mol.elements == "O"):
        cn = [a for a in adj[o] if mol.elements[a] == "C"]
        if len(cn) != 2:
            continue
        start, goal = cn
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) == 5:
                if node == goal:
                    ring = [o] + path
                    key = frozenset(ring)
                    if key not in seen:
                        seen.add(key)
                        rings.append(ring)
                continue
            for nb in adj[node]:
                if mol.elements[nb] == "C" and nb not in path and (nb != goal or len(path) == 4):
                    stack.append((nb, path + [nb]))
    return rings


def analyze_alginate(mol: Molecule) -> AlginateChain:
    """Recognise uronate residues, assign M/G from C5 chirality, order the chain."""
    m = mol if len(mol.bonds) else perceive_bonds(mol)
    adj = m.adjacency()
    warnings = []
    rings = _find_pyranose_rings(m, adj)
    residues = []
    for ring in rings:
        o5 = ring[0]
        ca, cb = ring[1], ring[-1]

        def exo_c(c):
            return [x for x in adj[c] if m.elements[x] == "C" and x not in ring]

        def exo_o(c):
            return [x for x in adj[c] if m.elements[x] == "O" and x not in ring]

        if exo_c(cb) and not exo_c(ca):
            c5, c1 = cb, ca
        elif exo_c(ca) and not exo_c(cb):
            c5, c1 = ca, cb
        else:
            continue
        # walk ring from C1 to C5
        path = ring[1:]  # carbons from ca to cb
        if path[0] != c1:
            path = path[::-1]
        c1_, c2, c3, c4, c5_ = path
        c6 = exo_c(c5)[0]
        o6 = [x for x in adj[c6] if m.elements[x] == "O"]
        if len(o6) < 2:
            continue  # not a uronic acid (e.g. CH2OH)
        a, b, c = m.coords[o5] - m.coords[c5], m.coords[c6] - m.coords[c5], m.coords[c4] - m.coords[c5]
        chir = float(np.dot(np.cross(a, b), c))
        kind = "M" if chir > 0 else "G"
        atoms = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "O5": o5, "C6": c6,
                 "O6A": o6[0], "O6B": o6[1]}
        for cname, oname in (("C1", "O1"), ("C2", "O2"), ("C3", "O3"), ("C4", "O4")):
            ox = exo_o(atoms[cname])
            if ox:
                atoms[oname] = ox[0]
        residues.append({"type": kind, "atoms": atoms, "chirality": chir})
    if not residues:
        raise ValueError("No uronic-acid pyranose rings found. Is this an alginate structure?")
    # linkages: C1(i) – O – C4(j)
    c4_of = {r["atoms"]["C4"]: k for k, r in enumerate(residues)}
    nxt = {}
    prv = {}
    for k, r in enumerate(residues):
        o1 = r["atoms"].get("O1")
        if o1 is None:
            continue
        for x in adj[o1]:
            if x in c4_of and c4_of[x] != k:
                j = c4_of[x]
                nxt[k] = j
                prv[j] = k
                # the bridging O belongs to residue j (as its O4)
                residues[j]["atoms"]["O4"] = o1
                r["atoms"].pop("O1", None)
    starts = [k for k in range(len(residues)) if k not in prv]
    order = []
    used = set()
    for s in starts:
        k = s
        while k is not None and k not in used:
            order.append(k)
            used.add(k)
            k = nxt.get(k)
    order += [k for k in range(len(residues)) if k not in used]
    if len(starts) > 1:
        warnings.append(f"{len(starts)} separate alginate chains found; they are concatenated in the sequence.")
    residues = [residues[k] for k in order]
    seq = "".join(r["type"] for r in residues)
    atom_res = -np.ones(m.n, int)
    for k, r in enumerate(residues):
        for nm, ai in list(r["atoms"].items()):
            atom_res[ai] = k
    # hydrogens and any leftover substituent atoms follow their heavy neighbour
    for i in range(m.n):
        if atom_res[i] < 0:
            nb = [x for x in adj[i] if atom_res[x] >= 0]
            if nb:
                atom_res[i] = atom_res[nb[0]]
    chain = AlginateChain(m, seq, residues, atom_res, source="recognised from 3-D structure", warnings=warnings)
    chain.linkages = linkage_report(chain)
    return chain


def standardize(chain: AlginateChain, title: str = "single_chain") -> AlginateChain:
    """Rename residues/atoms to BEM/LGU conventions and set formal charges."""
    m = chain.mol.copy()
    adj = m.adjacency()
    for k, r in enumerate(chain.residues):
        for nm, ai in r["atoms"].items():
            m.names[ai] = nm
        for ai in np.flatnonzero(chain.atom_res == k):
            m.resnames[ai] = RES_M if r["type"] == "M" else RES_G
            m.resids[ai] = k + 1
            m.chains[ai] = "A"
            if m.elements[ai] == "H":
                parent = [x for x in adj[ai] if m.elements[x] != "H"]
                if parent:
                    pn = str(m.names[parent[0]])
                    m.names[ai] = ("H" + pn[1:]) if pn.startswith("C") else ("HO" + pn[1:])
        # carboxylate: one O double-bonded, one O(-) unless already charged
        o6a, o6b = r["atoms"]["O6A"], r["atoms"]["O6B"]
        has_h = any(m.elements[x] == "H" for x in adj[o6a] + adj[o6b])
        if not has_h:
            qa, qb = m.charges[o6a], m.charges[o6b]
            dbl = None
            if qa == 0 and qb == 0:
                m.charges[o6b] = -1
                dbl = o6a
            elif qa < 0 and qb == 0:
                dbl = o6b
            elif qb < 0 and qa == 0:
                dbl = o6a
            if dbl is not None:
                for bi, (a, b) in enumerate(m.bonds):
                    if {int(a), int(b)} == {int(r["atoms"]["C6"]), int(dbl)}:
                        m.orders[bi] = 2
                    elif {int(a), int(b)} == {int(r["atoms"]["C6"]), int(o6a if dbl == o6b else o6b)}:
                        m.orders[bi] = 1
    m.title = title
    out = AlginateChain(m, chain.sequence, chain.residues, chain.atom_res, chain.source, list(chain.warnings))
    out.linkages = linkage_report(out)
    return out


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def linkage_report(chain: AlginateChain) -> list[dict]:
    X = chain.mol.coords
    rows = []
    for k in range(len(chain.residues) - 1):
        a, b = chain.residues[k]["atoms"], chain.residues[k + 1]["atoms"]
        if not all(x in a for x in ("O5", "C1")) or not all(x in b for x in ("O4", "C4", "C3")):
            continue
        phi = dihedral(X[a["O5"]], X[a["C1"]], X[b["O4"]], X[b["C4"]])
        psi = dihedral(X[a["C1"]], X[b["O4"]], X[b["C4"]], X[b["C3"]])
        ang = angle(X[a["C1"]], X[b["O4"]], X[b["C4"]])
        rows.append({"Linkage": f"{k + 1}→{k + 2}", "Type": chain.residues[k]["type"] + chain.residues[k + 1]["type"],
                     "φ (°)": round(phi, 1), "ψ (°)": round(psi, 1), "C1–O4–C4 angle (°)": round(ang, 1)})
    return rows


def ring_centroids(chain: AlginateChain, coords=None) -> np.ndarray:
    X = chain.mol.coords if coords is None else coords
    return np.array([X[[r["atoms"][k] for k in _RING]].mean(0) for r in chain.residues])


def chain_metrics(chain: AlginateChain) -> dict:
    cen = ring_centroids(chain)
    n = len(cen)
    steps = np.linalg.norm(np.diff(cen, axis=0), axis=1) if n > 1 else np.zeros(0)
    contour = float(steps.sum())
    e2e = float(np.linalg.norm(cen[-1] - cen[0])) if n > 1 else 0.0
    rise = float("nan")
    if n > 2:
        _, R, c = align_to_axes(cen, 0)
        proj = (cen - c) @ R[0]
        rise = float(abs(proj[-1] - proj[0]) / (n - 1))
    heavy = chain.mol.heavy_mask()
    width = float("nan")
    if chain.mol.n > 3:
        Xa, _, _ = align_to_axes(chain.mol.coords[heavy], 0)
        width = float(np.ptp(Xa[:, 1]) + 2 * 1.6)
    nG = chain.sequence.count("G")
    return {
        "Residues (DP)": n,
        "M units": chain.sequence.count("M"),
        "G units": nG,
        "Contour length (Å)": round(contour, 2),
        "End-to-end distance (Å)": round(e2e, 2),
        "Extension ratio (end-to-end / contour)": round(e2e / contour, 3) if contour > 0 else float("nan"),
        "Mean rise per residue (Å)": round(rise, 3) if not math.isnan(rise) else float("nan"),
        "Ribbon width incl. vdW (Å)": round(width, 2) if not math.isnan(width) else float("nan"),
        "Radius of gyration, heavy atoms (Å)": round(chain.mol.radius_of_gyration(heavy), 2),
        "Mw, Na-salt (g/mol)": round(n * MW_NA_URONATE + 18.02, 1),
        "Mw, acid form (g/mol)": round(n * MW_H_URONATE + 18.02, 1),
        "Charge at pH 7 (carboxylates)": -n,
    }


def sequence_stats(seq: str) -> dict:
    """NMR-style monad/dyad/triad frequencies and block lengths (Grasdalen notation)."""
    s = seq.upper()
    n = len(s)
    out = {"N": n}
    if n == 0:
        return out
    fG = s.count("G") / n
    out["F_G"] = fG
    out["F_M"] = 1 - fG
    pairs = [s[i:i + 2] for i in range(n - 1)]
    tri = [s[i:i + 3] for i in range(n - 2)]
    npair = max(len(pairs), 1)
    ntri = max(len(tri), 1)
    out["F_GG"] = pairs.count("GG") / npair
    out["F_MM"] = pairs.count("MM") / npair
    out["F_MG"] = (pairs.count("MG") + pairs.count("GM")) / (2 * npair)
    out["F_GM"] = out["F_MG"]
    out["F_GGG"] = tri.count("GGG") / ntri
    out["F_MGM"] = tri.count("MGM") / ntri
    out["F_GGM"] = (tri.count("GGM") + tri.count("MGG")) / (2 * ntri)
    out["F_MMM"] = tri.count("MMM") / ntri
    out["N_G>1"] = (out["F_G"] - out["F_MGM"]) / out["F_GGM"] if out["F_GGM"] > 0 else float("nan")
    out["M/G ratio"] = (1 - fG) / fG if fG > 0 else float("inf")
    runs = []
    cur, ln = s[0], 1
    for ch in s[1:]:
        if ch == cur:
            ln += 1
        else:
            runs.append((cur, ln))
            cur, ln = ch, 1
    runs.append((cur, ln))
    out["runs"] = runs
    g_runs = [l for c, l in runs if c == "G"]
    m_runs = [l for c, l in runs if c == "M"]
    out["Mean G-block length"] = float(np.mean(g_runs)) if g_runs else 0.0
    out["Mean M-block length"] = float(np.mean(m_runs)) if m_runs else 0.0
    out["Egg-box capable GG dyads"] = pairs.count("GG")
    return out


def generate_sequence(pattern: str, n: int, fg: float = 0.6, block: int = 4, seed: int = 7,
                      custom: str = "") -> str:
    n = max(1, int(n))
    if pattern == "poly-G":
        return "G" * n
    if pattern == "poly-M":
        return "M" * n
    if pattern == "alternating MG":
        return ("MG" * n)[:n]
    if pattern == "G-block / M-block":
        mlen = max(1, int(round(block * (1 - fg) / max(fg, 1e-6))))
        unit_ = "G" * block + "M" * mlen
        return (unit_ * (n // len(unit_) + 1))[:n]
    if pattern == "custom":
        s = "".join(ch for ch in custom.upper() if ch in "MG")
        return s or "G" * n
    # Markov random
    rng = np.random.default_rng(seed)
    fg = min(max(fg, 0.01), 0.99)
    lg = max(block, 1.0)
    lm = max(lg * (1 - fg) / fg, 1.0)
    pgg, pmm = 1 - 1 / lg, 1 - 1 / lm
    s = ["G" if rng.random() < fg else "M"]
    for _ in range(n - 1):
        if s[-1] == "G":
            s.append("G" if rng.random() < pgg else "M")
        else:
            s.append("M" if rng.random() < pmm else "G")
    return "".join(s)


def carboxylate_sites(chain: AlginateChain, coords=None) -> list[dict]:
    """Ca2+-binding sites = carboxylate groups; position = midpoint of the two O6."""
    X = chain.mol.coords if coords is None else coords
    sites = []
    for k, r in enumerate(chain.residues):
        a = r["atoms"]
        if not all(x in a for x in ("C6", "O6A", "O6B")):
            continue
        mid = 0.5 * (X[a["O6A"]] + X[a["O6B"]])
        nrm = unit(mid - X[a["C6"]])
        sites.append({"res": k, "type": r["type"], "pos": mid, "normal": nrm,
                      "O": (a["O6A"], a["O6B"]), "C6": a["C6"]})
    return sites
