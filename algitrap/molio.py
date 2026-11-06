"""Molecule container, file readers/writers (PDB, SDF/MOL V2000+V3000, XYZ)
and generic structure checks.  Pure numpy/scipy – no RDKit needed."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .constants import COVALENT_RADII, MASSES, METAL_IONS, VDW_RADII

TWO_LETTER = {"CL", "BR", "NA", "MG", "CA", "ZN", "FE", "SI", "LI", "AL", "MN", "CU", "CO", "NI", "SE", "K"}


@dataclass
class Molecule:
    elements: np.ndarray            # upper-case symbols, e.g. 'C', 'O', 'CA'
    coords: np.ndarray              # (N, 3) float Å
    names: np.ndarray               # atom names
    resnames: np.ndarray
    resids: np.ndarray              # int
    chains: np.ndarray              # chain ID strings
    charges: np.ndarray             # formal charges (int)
    bonds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))
    orders: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    title: str = ""
    segids: np.ndarray | None = None

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return int(len(self.elements))

    def copy(self) -> "Molecule":
        return Molecule(
            self.elements.copy(), self.coords.copy(), self.names.copy(), self.resnames.copy(),
            self.resids.copy(), self.chains.copy(), self.charges.copy(), self.bonds.copy(),
            self.orders.copy(), self.title, None if self.segids is None else self.segids.copy())

    def with_coords(self, xyz: np.ndarray) -> "Molecule":
        m = self.copy()
        m.coords = np.asarray(xyz, float).reshape(-1, 3).copy()
        return m

    def subset(self, mask) -> "Molecule":
        idx = np.flatnonzero(mask) if np.asarray(mask).dtype == bool else np.asarray(mask, int)
        remap = -np.ones(self.n, int)
        remap[idx] = np.arange(len(idx))
        if len(self.bonds):
            keep = (remap[self.bonds[:, 0]] >= 0) & (remap[self.bonds[:, 1]] >= 0)
            b = remap[self.bonds[keep]]
            o = self.orders[keep]
        else:
            b, o = np.zeros((0, 2), int), np.zeros(0, int)
        return Molecule(self.elements[idx], self.coords[idx], self.names[idx], self.resnames[idx],
                        self.resids[idx], self.chains[idx], self.charges[idx], b, o, self.title,
                        None if self.segids is None else self.segids[idx])

    @staticmethod
    def concat(mols: list["Molecule"], title: str = "") -> "Molecule":
        if not mols:
            return empty_molecule(title)
        offs = np.cumsum([0] + [m.n for m in mols[:-1]])
        bonds = [m.bonds + o for m, o in zip(mols, offs) if len(m.bonds)]
        orders = [m.orders for m in mols if len(m.bonds)]
        segs = None
        if any(m.segids is not None for m in mols):
            segs = np.concatenate([m.segids if m.segids is not None else np.array([""] * m.n, dtype=object)
                                   for m in mols])
        return Molecule(
            np.concatenate([m.elements for m in mols]), np.vstack([m.coords for m in mols]),
            np.concatenate([m.names for m in mols]), np.concatenate([m.resnames for m in mols]),
            np.concatenate([m.resids for m in mols]), np.concatenate([m.chains for m in mols]),
            np.concatenate([m.charges for m in mols]),
            np.vstack(bonds) if bonds else np.zeros((0, 2), int),
            np.concatenate(orders) if orders else np.zeros(0, int), title, segs)

    # ------------------------------------------------------------- properties
    def masses(self) -> np.ndarray:
        return np.array([MASSES.get(e, 12.0) for e in self.elements])

    def vdw(self) -> np.ndarray:
        return np.array([VDW_RADII.get(e, 1.8) for e in self.elements])

    def heavy_mask(self) -> np.ndarray:
        return self.elements != "H"

    def formula(self) -> str:
        return hill_formula(self.elements)

    def mol_weight(self) -> float:
        return float(self.masses().sum())

    def adjacency(self) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(self.n)]
        for a, b in self.bonds:
            adj[a].append(int(b))
            adj[b].append(int(a))
        return adj

    def radius_of_gyration(self, mask=None) -> float:
        x = self.coords if mask is None else self.coords[mask]
        if len(x) == 0:
            return float("nan")
        return float(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean()))

    def fragments(self) -> np.ndarray:
        """Connected-component label for every atom (by bonds)."""
        if self.n == 0:
            return np.zeros(0, int)
        if len(self.bonds) == 0:
            return np.arange(self.n)
        g = coo_matrix((np.ones(len(self.bonds)), (self.bonds[:, 0], self.bonds[:, 1])), shape=(self.n, self.n))
        _, lab = connected_components(g, directed=False)
        return lab


def empty_molecule(title="") -> Molecule:
    z = np.zeros(0)
    return Molecule(np.array([], dtype=object), np.zeros((0, 3)), np.array([], dtype=object),
                    np.array([], dtype=object), z.astype(int), np.array([], dtype=object), z.astype(int),
                    title=title)


def make_molecule(elements, coords, names=None, resnames=None, resids=None, chains=None, charges=None,
                  bonds=None, orders=None, title="") -> Molecule:
    n = len(elements)
    el = np.array([str(e).upper() for e in elements], dtype=object)
    return Molecule(
        el, np.asarray(coords, float).reshape(n, 3),
        np.array(names if names is not None else [f"{e.title()}{i + 1}" for i, e in enumerate(el)], dtype=object),
        np.array(resnames if resnames is not None else ["UNL"] * n, dtype=object),
        np.asarray(resids if resids is not None else np.ones(n, int), int),
        np.array(chains if chains is not None else ["A"] * n, dtype=object),
        np.asarray(charges if charges is not None else np.zeros(n, int), int),
        np.asarray(bonds if bonds is not None and len(bonds) else np.zeros((0, 2)), int).reshape(-1, 2),
        np.asarray(orders if orders is not None and len(orders) else np.ones(len(bonds) if bonds is not None else 0),
                   int),
        title)


def hill_formula(elements: Iterable[str]) -> str:
    from collections import Counter
    c = Counter(e.title() if len(e) > 1 else e for e in elements)
    parts = []
    for key in ("C", "H"):
        if key in c:
            parts.append(key + (str(c[key]) if c[key] > 1 else ""))
            del c[key]
    for key in sorted(c):
        parts.append(key + (str(c[key]) if c[key] > 1 else ""))
    return "".join(parts)


# ============================================================================
# Bond perception
# ============================================================================
def perceive_bonds(mol: Molecule, tol: float = 0.45) -> Molecule:
    """Distance-based covalent bonds. Metal ions never get covalent bonds."""
    m = mol.copy()
    cov = np.array([COVALENT_RADII.get(e, 0.0) for e in m.elements])
    ok = (cov > 0) & ~np.isin(m.elements, list(METAL_IONS))
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        m.bonds, m.orders = np.zeros((0, 2), int), np.zeros(0, int)
        return m
    tree = cKDTree(m.coords[idx])
    pairs = tree.query_pairs(r=2.0 * cov[idx].max() + tol, output_type="ndarray")
    if len(pairs) == 0:
        m.bonds, m.orders = np.zeros((0, 2), int), np.zeros(0, int)
        return m
    i, j = idx[pairs[:, 0]], idx[pairs[:, 1]]
    d = np.linalg.norm(m.coords[i] - m.coords[j], axis=1)
    keep = (d > 0.4) & (d < cov[i] + cov[j] + tol)
    # H–H bonds only if really short (H2); avoids bonding neighbouring H atoms
    hh = (m.elements[i] == "H") & (m.elements[j] == "H")
    keep &= ~hh | (d < 0.85)
    b = np.stack([i[keep], j[keep]], 1)
    m.bonds = np.sort(b, axis=1)
    m.orders = np.ones(len(b), int)
    return m


def deprotonate_carboxylic_acids(mol: Molecule) -> tuple[Molecule, int]:
    """COOH → COO⁻ (alginate is ionised at pH 7; pKa ≈ 3.4–3.7). Returns (molecule, groups changed)."""
    adj = mol.adjacency()
    remove, charge_o = [], []
    for c in np.flatnonzero(mol.elements == "C"):
        ox = [o for o in adj[c] if mol.elements[o] == "O" and sum(1 for x in adj[o] if mol.elements[x] != "H") == 1]
        if len(ox) != 2:
            continue
        hs = [(o, h) for o in ox for h in adj[o] if mol.elements[h] == "H"]
        if len(hs) == 1:
            remove.append(hs[0][1])
            charge_o.append(hs[0][0])
    if not remove:
        return mol, 0
    m = mol.copy()
    for o in charge_o:
        m.charges[o] = -1
    keep = np.ones(m.n, bool)
    keep[remove] = False
    return m.subset(keep), len(remove)


def assign_carboxylates(mol: Molecule) -> Molecule:
    """Mark carboxylate groups C(=O)O- when bond orders are unknown.

    A carbon bound to exactly two terminal oxygens (no H on them) becomes
    one C=O and one C–O(–) with formal charge −1."""
    m = mol.copy()
    adj = m.adjacency()
    border = {(min(a, b), max(a, b)): k for k, (a, b) in enumerate(m.bonds)}
    for c in np.flatnonzero(m.elements == "C"):
        ox = [o for o in adj[c] if m.elements[o] == "O" and len(adj[o]) == 1]
        if len(ox) == 2 and len(adj[c]) == 3:
            k1 = border[(min(c, ox[0]), max(c, ox[0]))]
            m.orders[k1] = 2
            if m.charges[ox[1]] == 0 and m.charges[ox[0]] == 0:
                m.charges[ox[1]] = -1
    return m


# ============================================================================
# Readers
# ============================================================================
def _infer_element(name: str, resname: str, rec: str) -> str:
    nm = name.strip().upper()
    rn = resname.strip().upper()
    if rn in METAL_IONS | {"CL"} and nm.rstrip("0123456789+-") == rn:
        return rn
    letters = "".join(ch for ch in nm if ch.isalpha())
    if not letters:
        return "X"
    if rec == "HETATM" and name[:2].strip().upper() in TWO_LETTER and name[0] != " ":
        return name[:2].strip().upper()
    return letters[0]


def parse_pdb(text: str) -> tuple[Molecule, dict]:
    els, xyz, names, rn, ri, ch, q, seg = [], [], [], [], [], [], [], []
    serial_to_idx: dict[int, int] = {}
    conect: list[tuple[int, int]] = []
    n_models = 0
    in_first = True
    for line in text.splitlines():
        rec = line[:6].strip()
        if rec == "MODEL":
            n_models += 1
            in_first = n_models == 1
            continue
        if rec == "ENDMDL":
            in_first = False
            continue
        if rec in ("ATOM", "HETATM") and in_first:
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            name = line[12:16]
            resname = line[17:20]
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem or not elem.isalpha():
                elem = _infer_element(name, resname, rec)
            chg = 0
            if len(line) >= 80:
                cs = line[78:80].strip()
                if cs:
                    try:
                        chg = int(cs[0]) * (-1 if cs[-1] == "-" else 1) if cs[0].isdigit() else int(cs)
                    except ValueError:
                        chg = 0
            try:
                serial = int(line[6:11])
            except ValueError:
                serial = len(els) + 1
            serial_to_idx[serial] = len(els)
            els.append(elem)
            xyz.append((x, y, z))
            names.append(name.strip())
            rn.append(resname.strip() or "UNL")
            try:
                ri.append(int(line[22:26]))
            except ValueError:
                ri.append(1)
            ch.append(line[21:22].strip() or "A")
            q.append(chg)
            seg.append(line[72:76].strip() if len(line) >= 76 else "")
        elif rec == "CONECT":
            try:
                a = int(line[6:11])
                for k in range(11, 31, 5):
                    s = line[k:k + 5].strip()
                    if s:
                        conect.append((a, int(s)))
            except ValueError:
                pass
    if not els:
        raise ValueError("No ATOM/HETATM records found – is this really a PDB file?")
    mol = Molecule(np.array(els, dtype=object), np.array(xyz, float), np.array(names, dtype=object),
                   np.array(rn, dtype=object), np.array(ri, int), np.array(ch, dtype=object), np.array(q, int),
                   segids=np.array(seg, dtype=object))
    mol = perceive_bonds(mol)
    if conect:
        extra = {(min(serial_to_idx[a], serial_to_idx[b]), max(serial_to_idx[a], serial_to_idx[b]))
                 for a, b in conect if a in serial_to_idx and b in serial_to_idx and a != b}
        have = {tuple(b) for b in mol.bonds.tolist()}
        add = [b for b in extra if b not in have
               and not (mol.elements[b[0]] in METAL_IONS or mol.elements[b[1]] in METAL_IONS)]
        if add:
            mol.bonds = np.vstack([mol.bonds, np.array(add, int)])
            mol.orders = np.concatenate([mol.orders, np.ones(len(add), int)])
    info = {"format": "PDB", "models": max(n_models, 1), "conect_records": len(conect)}
    return mol, info


_V2000_CHG = {1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3}


def parse_sdf(text: str) -> tuple[Molecule, dict]:
    records = [r for r in text.split("$$$$") if r.strip()]
    if not records:
        raise ValueError("Empty SDF/MOL file.")
    block = records[0]
    lines = block.splitlines()
    # skip leading blank lines produced by "$$$$\n"
    while lines and not lines[0].strip() and len(lines) > 4 and "V2000" not in lines[3] and "V3000" not in lines[3]:
        lines = lines[1:]
    if len(lines) < 4:
        raise ValueError("SDF/MOL header is incomplete (fewer than 4 lines).")
    title = lines[0].strip()
    counts = lines[3]
    if "V3000" in counts:
        mol = _parse_v3000(lines, title)
        fmt = "SDF/MOL V3000"
    else:
        try:
            na, nb = int(counts[0:3]), int(counts[3:6])
        except ValueError:
            raise ValueError("Cannot read the atom/bond counts line of the MOL block.")
        els, xyz, chg = [], [], []
        for k in range(na):
            ln = lines[4 + k]
            xyz.append((float(ln[0:10]), float(ln[10:20]), float(ln[20:30])))
            els.append(ln[31:34].strip().upper())
            try:
                cc = int(ln[36:39])
            except ValueError:
                cc = 0
            chg.append(_V2000_CHG.get(cc, 0))
        bonds, orders = [], []
        for k in range(nb):
            ln = lines[4 + na + k]
            a, b, t = int(ln[0:3]), int(ln[3:6]), int(ln[6:9])
            bonds.append((a - 1, b - 1))
            orders.append(t if t in (1, 2, 3) else (4 if t == 4 else 1))
        chg = np.array(chg, int)
        for ln in lines[4 + na + nb:]:
            if ln.startswith("M  CHG"):
                parts = ln.split()
                cnt = int(parts[2])
                for j in range(cnt):
                    chg[int(parts[3 + 2 * j]) - 1] = int(parts[4 + 2 * j])
        mol = make_molecule(els, xyz, resnames=["MOL"] * na, charges=chg, bonds=bonds, orders=orders, title=title)
        fmt = "SDF/MOL V2000"
    info = {"format": fmt, "records": len(records), "title": title}
    return mol, info


def _parse_v3000(lines: list[str], title: str) -> Molecule:
    els, xyz, chg, bonds, orders = [], [], [], [], []
    mode = None
    idmap = {}
    for ln in lines:
        if not ln.startswith("M  V30"):
            continue
        body = ln[6:].strip()
        if body.startswith("BEGIN ATOM"):
            mode = "atom"; continue
        if body.startswith("END ATOM"):
            mode = None; continue
        if body.startswith("BEGIN BOND"):
            mode = "bond"; continue
        if body.startswith("END BOND"):
            mode = None; continue
        p = body.split()
        if mode == "atom" and len(p) >= 5:
            idmap[int(p[0])] = len(els)
            els.append(p[1].upper())
            xyz.append((float(p[2]), float(p[3]), float(p[4])))
            c = 0
            for tok in p[6:]:
                if tok.startswith("CHG="):
                    c = int(tok[4:])
            chg.append(c)
        elif mode == "bond" and len(p) >= 4:
            t = int(p[1])
            bonds.append((idmap[int(p[2])], idmap[int(p[3])]))
            orders.append(t if t in (1, 2, 3) else 1)
    if not els:
        raise ValueError("V3000 block has no atoms.")
    return make_molecule(els, xyz, resnames=["MOL"] * len(els), charges=chg, bonds=bonds, orders=orders, title=title)


def parse_xyz(text: str) -> tuple[Molecule, dict]:
    lines = text.strip().splitlines()
    n = int(lines[0].split()[0])
    els, xyz = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        els.append(p[0].upper())
        xyz.append(tuple(map(float, p[1:4])))
    mol = perceive_bonds(make_molecule(els, xyz, title=lines[1].strip() if len(lines) > 1 else ""))
    return mol, {"format": "XYZ", "models": 1}


def load_structure(filename: str, data: bytes | str) -> tuple[Molecule, dict]:
    text = data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else data
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext in ("pdb", "ent", "pdbqt"):
        mol, info = parse_pdb(text)
    elif ext in ("sdf", "mol", "sd"):
        mol, info = parse_sdf(text)
        if len(mol.bonds) == 0:
            mol = perceive_bonds(mol)
    elif ext == "xyz":
        mol, info = parse_xyz(text)
    else:  # sniff
        if "V2000" in text or "V3000" in text:
            mol, info = parse_sdf(text)
        elif "ATOM" in text or "HETATM" in text:
            mol, info = parse_pdb(text)
        else:
            raise ValueError(f"Unsupported file type '.{ext}'. Use .pdb, .sdf, .mol or .xyz")
    info["filename"] = filename
    info["size_kb"] = round(len(text.encode()) / 1024, 2)
    mol.title = mol.title or filename
    return mol, info


# ============================================================================
# Writers
# ============================================================================
def _fmt_name(name: str, elem: str) -> str:
    name = str(name)[:4]
    if len(name) < 4 and len(elem) == 1:
        return (" " + name).ljust(4)
    return name.ljust(4)


def _chg_str(c: int) -> str:
    if c == 0:
        return "  "
    return f"{abs(c)}{'+' if c > 0 else '-'}"


def pdb_atom_lines(mol: Molecule, coords: np.ndarray | None = None, het_resnames=("CA", "NA", "HOH", "MOL", "UNL"),
                   bfac: np.ndarray | None = None):
    xyz = mol.coords if coords is None else coords
    out = []
    for i in range(mol.n):
        el = mol.elements[i]
        rec = "HETATM" if (mol.resnames[i] in het_resnames or el in METAL_IONS) else "ATOM  "
        seg = (mol.segids[i] if mol.segids is not None else "")[:4]
        out.append(
            f"{rec}{(i + 1) % 100000:>5} {_fmt_name(mol.names[i], el)} {str(mol.resnames[i])[:3]:>3} "
            f"{str(mol.chains[i])[:1]:1}{int(mol.resids[i]) % 10000:>4}    "
            f"{xyz[i, 0]:8.3f}{xyz[i, 1]:8.3f}{xyz[i, 2]:8.3f}{1.0:6.2f}{(0.0 if bfac is None else float(bfac[i])):6.2f}      "
            f"{seg:<4}{(el.title() if len(el) > 1 else el):>2}{_chg_str(int(mol.charges[i]))}")
    return out


def conect_lines(mol: Molecule) -> list[str]:
    if mol.n >= 100000 or len(mol.bonds) == 0:
        return []
    nb: dict[int, list[int]] = {}
    for a, b in mol.bonds:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    out = []
    for a in sorted(nb):
        partners = sorted(nb[a])
        for k in range(0, len(partners), 4):
            out.append("CONECT" + f"{a + 1:>5}" + "".join(f"{p + 1:>5}" for p in partners[k:k + 4]))
    return out


def write_pdb(mol: Molecule, coords: np.ndarray | None = None, conect: bool = True, remarks: list[str] | None = None,
              bfac: np.ndarray | None = None) -> str:
    lines = [f"HEADER    {mol.title[:40]}", "REMARK   1 Generated by AlgiTrap"]
    for r in remarks or []:
        lines.append(f"REMARK   2 {r[:68]}")
    lines += pdb_atom_lines(mol, coords, bfac=bfac)
    if conect:
        lines += conect_lines(mol)
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pdb_models(mol: Molecule, frames: list[np.ndarray], conect: bool = False, bfac: np.ndarray | None = None) -> str:
    lines = [f"HEADER    {mol.title[:40]} TRAJECTORY", "REMARK   1 Generated by AlgiTrap (multi-model)"]
    for k, fr in enumerate(frames):
        lines.append(f"MODEL     {k + 1:>4}")
        lines += pdb_atom_lines(mol, fr, bfac=bfac)
        lines.append("ENDMDL")
    if conect:
        lines += conect_lines(mol)
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_sdf(mol: Molecule, coords: np.ndarray | None = None) -> str:
    xyz = mol.coords if coords is None else coords
    if mol.n > 999 or len(mol.bonds) > 999:
        return write_sdf_v3000(mol, xyz)
    buf = io.StringIO()
    buf.write(f"{mol.title[:80]}\n  AlgiTrap          3D\n\n")
    buf.write(f"{mol.n:>3}{len(mol.bonds):>3}  0  0  0  0  0  0  0  0999 V2000\n")
    inv = {v: k for k, v in _V2000_CHG.items()}
    for i in range(mol.n):
        el = mol.elements[i].title() if len(mol.elements[i]) > 1 else mol.elements[i]
        cc = inv.get(int(mol.charges[i]), 0) if mol.charges[i] != 0 else 0
        buf.write(f"{xyz[i, 0]:>10.4f}{xyz[i, 1]:>10.4f}{xyz[i, 2]:>10.4f} {el:<3} 0{cc:>3}  0  0  0  0  0  0  0  0  0  0\n")
    for (a, b), o in zip(mol.bonds, mol.orders):
        buf.write(f"{a + 1:>3}{b + 1:>3}{int(o):>3}  0  0  0  0\n")
    ch = [(i + 1, int(c)) for i, c in enumerate(mol.charges) if c != 0]
    for k in range(0, len(ch), 8):
        part = ch[k:k + 8]
        buf.write(f"M  CHG{len(part):>3}" + "".join(f" {a:>3} {c:>3}" for a, c in part) + "\n")
    buf.write("M  END\n$$$$\n")
    return buf.getvalue()


def write_sdf_v3000(mol: Molecule, xyz: np.ndarray) -> str:
    out = [mol.title[:80], "  AlgiTrap          3D", "", "  0  0  0     0  0            999 V3000",
           "M  V30 BEGIN CTAB", f"M  V30 COUNTS {mol.n} {len(mol.bonds)} 0 0 0", "M  V30 BEGIN ATOM"]
    for i in range(mol.n):
        el = mol.elements[i].title() if len(mol.elements[i]) > 1 else mol.elements[i]
        c = f" CHG={int(mol.charges[i])}" if mol.charges[i] else ""
        out.append(f"M  V30 {i + 1} {el} {xyz[i, 0]:.4f} {xyz[i, 1]:.4f} {xyz[i, 2]:.4f} 0{c}")
    out.append("M  V30 END ATOM")
    out.append("M  V30 BEGIN BOND")
    for k, ((a, b), o) in enumerate(zip(mol.bonds, mol.orders)):
        out.append(f"M  V30 {k + 1} {int(o)} {a + 1} {b + 1}")
    out += ["M  V30 END BOND", "M  V30 END CTAB", "M  END", "$$$$"]
    return "\n".join(out) + "\n"


def write_xyz(mol: Molecule, coords=None) -> str:
    xyz = mol.coords if coords is None else coords
    lines = [str(mol.n), mol.title]
    for e, (x, y, z) in zip(mol.elements, xyz):
        lines.append(f"{e.title():<3}{x:12.5f}{y:12.5f}{z:12.5f}")
    return "\n".join(lines) + "\n"


# ============================================================================
# Generic structure statistics and checks
# ============================================================================
def structure_stats(mol: Molecule) -> dict:
    from collections import Counter
    heavy = mol.heavy_mask()
    ext = mol.coords.max(0) - mol.coords.min(0) if mol.n else np.zeros(3)
    frag = mol.fragments()
    d_min = float("nan")
    if mol.n > 1:
        tree = cKDTree(mol.coords)
        dd, _ = tree.query(mol.coords, k=2)
        d_min = float(dd[:, 1].min())
    comp = Counter(e.title() if len(e) > 1 else e for e in mol.elements)
    return {
        "Atoms": mol.n,
        "Heavy atoms": int(heavy.sum()),
        "Hydrogens": int((~heavy).sum()),
        "Formula": mol.formula(),
        "Molecular weight (g/mol)": round(mol.mol_weight(), 2),
        "Net formal charge": int(mol.charges.sum()),
        "Bonds": int(len(mol.bonds)),
        "Fragments (molecules/ions)": int(len(np.unique(frag))) if mol.n else 0,
        "Residues": int(len({(c, r, n) for c, r, n in zip(mol.chains, mol.resids, mol.resnames)})),
        "Residue names": ", ".join(sorted(set(map(str, mol.resnames))))[:80],
        "Chains": int(len(set(mol.chains))),
        "Extent x/y/z (Å)": " × ".join(f"{v:.1f}" for v in ext),
        "Radius of gyration (Å)": round(mol.radius_of_gyration(), 2) if mol.n else float("nan"),
        "Shortest interatomic distance (Å)": round(d_min, 3) if not math.isnan(d_min) else float("nan"),
        "Composition": ", ".join(f"{k}:{v}" for k, v in sorted(comp.items())),
    }


def generic_checks(mol: Molecule, info: dict) -> list[dict]:
    """Return a list of {Check, Status, Detail} rows (Status PASS/WARN/FAIL/INFO)."""
    rows = []

    def add(check, status, detail):
        rows.append({"Check": check, "Status": status, "Detail": detail})

    add("File parsed", "PASS", f"{info.get('format', '?')} · {info.get('size_kb', '?')} kB · {mol.n} atoms")
    if info.get("models", 1) > 1:
        add("Multiple models", "WARN", f"{info['models']} MODEL blocks – only the first is used")
    if info.get("records", 1) > 1:
        add("Multiple SDF records", "WARN", f"{info['records']} molecules – only the first is used")
    if mol.n == 0:
        add("Atoms present", "FAIL", "No atoms")
        return rows
    unknown = sorted({e for e in mol.elements if e not in MASSES})
    add("Element symbols", "FAIL" if unknown else "PASS",
        f"Unknown: {', '.join(unknown)}" if unknown else "All elements recognised")
    span = mol.coords.max(0) - mol.coords.min(0)
    if mol.n > 3 and np.any(span < 1e-3):
        add("3-D coordinates", "FAIL", "One axis is flat (2-D drawing). Please provide 3-D coordinates.")
    else:
        add("3-D coordinates", "PASS", "Structure has depth in x, y and z")
    tree = cKDTree(mol.coords)
    close = tree.query_pairs(0.7, output_type="ndarray")
    add("Atom clashes (< 0.7 Å)", "FAIL" if len(close) else "PASS",
        f"{len(close)} overlapping atom pairs" if len(close) else "No overlapping atoms")
    if len(mol.bonds):
        d = np.linalg.norm(mol.coords[mol.bonds[:, 0]] - mol.coords[mol.bonds[:, 1]], axis=1)
        heavy_b = (mol.elements[mol.bonds[:, 0]] != "H") & (mol.elements[mol.bonds[:, 1]] != "H")
        bad = int(((d > 1.75) | (d < 1.15))[heavy_b].sum())
        add("Heavy-atom bond lengths", "WARN" if bad else "PASS",
            f"{bad} unusual bonds (outside 1.15–1.75 Å)" if bad else f"{int(heavy_b.sum())} bonds, all in 1.15–1.75 Å")
    adj = mol.adjacency()
    over = [i for i in range(mol.n) if mol.elements[i] == "C" and len(adj[i]) > 4]
    add("Carbon valence ≤ 4", "WARN" if over else "PASS",
        f"{len(over)} carbons with >4 neighbours" if over else "All carbons have ≤ 4 neighbours")
    has_h = bool((mol.elements == "H").any())
    add("Explicit hydrogens", "INFO", "Present" if has_h else "Absent (united-atom model; fine for mesh building)")
    return rows
