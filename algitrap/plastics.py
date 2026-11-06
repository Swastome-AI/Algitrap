"""Microplastic oligomer builders (PE, PS), polymer identification and
size descriptors used for sieving predictions."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from .constants import PLASTIC_INFO, VDW_RADII
from .geom import align_to_axes, fibonacci_sphere, min_enclosing_circle, nerf, tetra_pair, unit
from .molio import Molecule, make_molecule


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _backbone(nc: int, torsions: list[float]) -> np.ndarray:
    X = [np.array([0.0, 0.0, 0.0]), np.array([1.53, 0.0, 0.0])]
    X.append(X[1] + 1.53 * unit(np.array([np.cos(np.radians(68)), np.sin(np.radians(68)), 0.0])))
    for k in range(3, nc):
        X.append(nerf(X[k - 3], X[k - 2], X[k - 1], 1.53, 112.0, torsions[k % len(torsions)]))
    return np.array(X[:nc])


def _add_end_methyl_h(X, k, nbr, ref, els, xyz, bonds, parent):
    for tor in (180.0, 60.0, -60.0):
        h = nerf(X[ref], X[nbr], X[k], 1.09, 109.5, tor)
        bonds.append((parent, len(els)))
        els.append("H")
        xyz.append(h)


def build_pe(n_monomers: int = 10, include_h: bool = True) -> Molecule:
    """All-trans (crystalline-like) polyethylene oligomer H-(CH2-CH2)n-H."""
    nc = max(2, 2 * int(n_monomers))
    X = _backbone(nc, [180.0])
    els = ["C"] * nc
    xyz = [x for x in X]
    bonds = [(k, k + 1) for k in range(nc - 1)]
    if include_h:
        for k in range(nc):
            if 0 < k < nc - 1:
                d1, d2 = tetra_pair(X[k], X[k - 1], X[k + 1])
                for d in (d1, d2):
                    bonds.append((k, len(els)))
                    els.append("H")
                    xyz.append(X[k] + 1.09 * d)
            elif k == 0:
                _add_end_methyl_h(X, 0, 1, 2, els, xyz, bonds, 0)
            else:
                _add_end_methyl_h(X, nc - 1, nc - 2, nc - 3, els, xyz, bonds, nc - 1)
    xyz = np.array(xyz)
    xyz, _, _ = align_to_axes(xyz, 0)
    return make_molecule(els, xyz, resnames=["PE"] * len(els), bonds=bonds, title=f"PE_{n_monomers}mer")


def build_ps(n_monomers: int = 6, tacticity: str = "syndiotactic", include_h: bool = True, seed: int = 3,
             relax: bool = True) -> Molecule:
    """Polystyrene oligomer CH3-(CH(Ph)-CH2)n-H.

    syndiotactic → all-trans zigzag (as in sPS crystals); isotactic → TG 3₁ helix;
    atactic → random configuration on a zigzag, then a short clash-relief minimisation."""
    n = max(1, int(n_monomers))
    nc = 2 * n + 1
    tors = {"isotactic": [180.0, 60.0], "syndiotactic": [180.0], "atactic": [180.0]}[tacticity]
    X = _backbone(nc, tors)
    rng = np.random.default_rng(seed)
    els = ["C"] * nc
    xyz = [x for x in X]
    bonds = [(k, k + 1) for k in range(nc - 1)]
    resn = ["PS"] * nc
    for k in range(nc):
        if 0 < k < nc - 1:
            d1, d2 = tetra_pair(X[k], X[k - 1], X[k + 1])
            if k % 2 == 1:   # CH(Ph) carbon
                s = np.dot(np.cross(X[k - 1] - X[k], X[k + 1] - X[k]), d1)
                pick_pos = True
                if tacticity == "syndiotactic":
                    pick_pos = ((k // 2) % 2 == 0)
                elif tacticity == "atactic":
                    pick_pos = bool(rng.random() < 0.5)
                dp, dh = (d1, d2) if (s > 0) == pick_pos else (d2, d1)
                ipso = X[k] + 1.51 * dp
                axis = dp
                e = unit(np.cross(axis, X[k + 1] - X[k - 1]))
                center = ipso + 1.39 * axis
                ring_idx = []
                for j in range(6):
                    th = np.radians(60 * j)
                    pos = center + 1.39 * (-np.cos(th) * axis + np.sin(th) * e)
                    ring_idx.append(len(els))
                    els.append("C")
                    xyz.append(pos)
                    resn.append("PS")
                bonds.append((k, ring_idx[0]))
                for j in range(6):
                    bonds.append((ring_idx[j], ring_idx[(j + 1) % 6]))
                if include_h:
                    for j in range(1, 6):
                        hpos = xyz[ring_idx[j]] + 1.08 * unit(xyz[ring_idx[j]] - center)
                        bonds.append((ring_idx[j], len(els)))
                        els.append("H"); xyz.append(hpos); resn.append("PS")
                    bonds.append((k, len(els)))
                    els.append("H"); xyz.append(X[k] + 1.09 * dh); resn.append("PS")
            elif include_h:
                for d in (d1, d2):
                    bonds.append((k, len(els)))
                    els.append("H"); xyz.append(X[k] + 1.09 * d); resn.append("PS")
        elif include_h:
            if k == 0:
                _add_end_methyl_h(X, 0, 1, 2, els, xyz, bonds, 0)
            else:
                _add_end_methyl_h(X, nc - 1, nc - 2, nc - 3, els, xyz, bonds, nc - 1)
            resn += ["PS"] * 3
    xyz = np.array(xyz)
    orders = np.ones(len(bonds), int)
    mol = make_molecule(els, xyz, resnames=resn, bonds=bonds, orders=orders, title=f"PS_{n}mer_{tacticity}")
    _kekulize_rings(mol)
    if relax and tacticity == "atactic":
        try:
            from .forcefield import quick_clash_relief
            mol = mol.with_coords(quick_clash_relief(mol))
        except Exception:
            pass
    xyz, _, _ = align_to_axes(mol.coords, 0)
    return mol.with_coords(xyz)


def _kekulize_rings(mol: Molecule):
    """Assign alternating bond orders to all-carbon 6-rings (in place)."""
    rings = aromatic_rings(mol, check_geometry=False)
    bindex = {(min(a, b), max(a, b)): i for i, (a, b) in enumerate(mol.bonds)}
    for ring in rings:
        for j in range(0, 6, 2):
            a, b = ring[j], ring[(j + 1) % 6]
            k = bindex.get((min(a, b), max(a, b)))
            if k is not None:
                mol.orders[k] = 2


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------
def aromatic_rings(mol: Molecule, check_geometry: bool = True) -> list[list[int]]:
    adj = mol.adjacency()
    rings, seen = [], set()
    carbons = np.flatnonzero(mol.elements == "C")
    for s in carbons:
        stack = [(s, [s])]
        while stack:
            node, path = stack.pop()
            if len(path) == 6:
                if s in adj[node]:
                    key = frozenset(path)
                    if key not in seen:
                        seen.add(key)
                        rings.append(path)
                continue
            for nb in adj[node]:
                if mol.elements[nb] == "C" and nb not in path and nb > s:
                    stack.append((nb, path + [nb]))
    if not check_geometry:
        return rings
    out = []
    for r in rings:
        X = mol.coords[r]
        c = X.mean(0)
        _, _, vt = np.linalg.svd(X - c)
        dev = np.abs((X - c) @ vt[2]).max()
        bl = np.mean([np.linalg.norm(X[i] - X[(i + 1) % 6]) for i in range(6)])
        if dev < 0.15 and 1.33 < bl < 1.45:
            out.append(r)
    return out


def identify_polymer(mol: Molecule) -> dict:
    els = list(mol.elements)
    nC, nH = els.count("C"), els.count("H")
    other = sorted({e for e in els if e not in ("C", "H")})
    rings = aromatic_rings(mol)
    adj = mol.adjacency()
    ch3 = sum(1 for i in np.flatnonzero(mol.elements == "C")
              if sum(1 for x in adj[i] if mol.elements[x] == "H") == 3)
    branch = sum(1 for i in np.flatnonzero(mol.elements == "C")
                 if sum(1 for x in adj[i] if mol.elements[x] == "C") >= 3)
    evidence = []
    label, conf = "Other", "low"
    ratio = nH / nC if nC else float("nan")
    if "CL" in other:
        label, conf = "Other", "medium"
        evidence.append("Contains chlorine → PVC-like (not PS/PE)")
    elif "O" in other and rings:
        label, conf = "Other", "medium"
        evidence.append("Aromatic rings + oxygen → PET-like polyester")
    elif not other and rings:
        label = "PS"
        conf = "high" if (nH == 0 or 0.9 <= ratio <= 1.35) else "medium"
        evidence.append(f"{len(rings)} planar aromatic C₆ ring(s) (phenyl side groups)")
    elif not other and not rings:
        if nH and nC and ch3 > max(2, nC / 4):
            label, conf = "PP", "medium"
            evidence.append(f"{ch3} CH₃ side groups → polypropylene-like")
        else:
            label = "PE"
            conf = "high" if (nH == 0 or 1.9 <= ratio <= 2.4) and branch <= 2 else "medium"
            evidence.append("Saturated hydrocarbon chain, no rings")
    if nH:
        evidence.append(f"H:C ratio = {ratio:.2f} (PE ≈ 2.0, PS ≈ 1.0)")
    else:
        evidence.append("No explicit hydrogens – identification based on carbon skeleton only")
    units = len(rings) if label == "PS" else (nC // 2 if label == "PE" else (nC // 3 if label == "PP" else 0))
    return {"label": label, "confidence": conf, "evidence": evidence, "aromatic_rings": len(rings),
            "CH3_groups": ch3, "branch_points": branch, "H:C": ratio, "repeat_units_est": units,
            "name": PLASTIC_INFO[label]["name"]}


# ---------------------------------------------------------------------------
# Size descriptors
# ---------------------------------------------------------------------------
def size_descriptors(mol: Molecule, n_dirs: int = 300) -> dict:
    X = mol.coords
    rad = np.array([VDW_RADII.get(e, 1.7) for e in mol.elements])
    if mol.n == 1:
        d = 2 * rad[0]
        return {"Max dimension (Å)": d, "Min cross-section diameter (Å)": d, "Equivalent-sphere diameter (Å)": d,
                "Radius of gyration (Å)": 0.0, "L1 (Å)": d, "L2 (Å)": d, "L3 (Å)": d, "Aspect ratio L1/L2": 1.0,
                "vdW volume (Å³)": 4 / 3 * np.pi * rad[0] ** 3}
    # max dimension incl. vdW
    if mol.n <= 3000:
        from scipy.spatial.distance import squareform
        D = squareform(pdist(X))
        D = D + rad[:, None] + rad[None]
        np.fill_diagonal(D, 0)
        dmax = float(D.max())
    else:
        dmax = float(pdist(X).max() + 2 * rad.max())
    Xa, R, c = align_to_axes(X, 0)
    L = [float(np.max(Xa[:, k] + rad) - np.min(Xa[:, k] - rad)) for k in range(3)]
    # minimum cross-section: smallest enclosing circle over viewing directions
    _, axes, _ = __import__("algitrap.geom", fromlist=["principal_axes"]).principal_axes(X)
    dirs = np.vstack([fibonacci_sphere(n_dirs), axes])
    rng = np.random.default_rng(0)
    pick = rng.choice(mol.n, 400, replace=False) if mol.n > 400 else np.arange(mol.n)
    sub, subr = X[pick], rad[pick]

    def width(u):
        u = unit(u)
        a = unit(np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0]))
        b = np.cross(u, a)
        P2 = np.stack([sub @ a, sub @ b], 1)
        c2, _ = min_enclosing_circle(P2, return_center=True)
        return 2 * float(np.max(np.linalg.norm(P2 - c2, axis=1) + subr))

    scores = [width(u) for u in dirs]
    k0 = int(np.argmin(scores))
    best, best_dir = scores[k0], unit(dirs[k0])
    for scale in (10.0, 4.0, 1.5):
        for _ in range(30):
            cand = unit(best_dir + np.radians(scale) * rng.normal(size=3))
            w = width(cand)
            if w < best:
                best, best_dir = w, cand
    # vdW volume on a grid
    lo, hi = X.min(0) - rad.max(), X.max(0) + rad.max()
    h = 0.35 if mol.n < 600 else 0.6
    gx = [np.arange(lo[k], hi[k] + h, h) for k in range(3)]
    G = np.stack(np.meshgrid(*gx, indexing="ij"), -1).reshape(-1, 3)
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    inside = np.zeros(len(G), bool)
    for start in range(0, len(G), 400000):
        blk = G[start:start + 400000]
        lst = tree.query_ball_point(blk, r=rad.max())
        for k, ids in enumerate(lst):
            if ids:
                dd = np.linalg.norm(X[ids] - blk[k], axis=1)
                inside[start + k] = bool(np.any(dd <= rad[ids]))
    vol = float(inside.sum() * h ** 3)
    deq = (6 * vol / np.pi) ** (1 / 3)
    Ls = sorted(L, reverse=True)
    return {
        "Max dimension (Å)": round(dmax, 2),
        "Min cross-section diameter (Å)": round(float(best), 2),
        "Equivalent-sphere diameter (Å)": round(deq, 2),
        "Radius of gyration (Å)": round(mol.radius_of_gyration(), 2),
        "L1 (Å)": round(Ls[0], 2), "L2 (Å)": round(Ls[1], 2), "L3 (Å)": round(Ls[2], 2),
        "Aspect ratio L1/L2": round(Ls[0] / Ls[1], 2) if Ls[1] > 0 else float("nan"),
        "vdW volume (Å³)": round(vol, 1),
        "_best_dir": best_dir.tolist(),
    }
