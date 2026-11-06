"""AlgiTrap-ENM: a fast elastic-network + ion-binding force field for
alginate/Ca²⁺ networks, an L-BFGS minimiser that records an animation
trajectory, a Langevin-dynamics integrator, and optional bridges to
OpenBabel (UFF) and RDKit (UFF / MMFF94).

Energy terms (kcal/mol, Å):
  bonds        k_b (r − r0)²        (r0 = starting geometry)
  1-3 pairs    k_a (r − r0)²        (keeps bond angles)
  ring 1-4     k_r (r − r0)²        (keeps chair / aromatic rings rigid)
  other 1-4    k_t (r − r0)²        (weak torsional stiffness, lets chains bend)
  repulsion    k_rep (σ − r)² for r < σ,  σ = s·(R_vdW,i + R_vdW,j)
  Ca–O         Morse  D[(1 − e^{−a(r − r0)})² − 1]   (deeper for carboxylate O)
  electrostatic Debye–Hückel 332.06 q_i q_j e^{−r/λ_D} /(ε_r r)
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import minimize as sp_minimize
from scipy.spatial import cKDTree

from .constants import COULOMB_KCAL, METAL_IONS, R_KCAL, VDW_RADII
from .molio import Molecule


@dataclass
class FFParams:
    k_bond: float = 300.0
    k_angle: float = 80.0
    k_ring14: float = 40.0
    k_torsion14: float = 2.0
    k_rep: float = 20.0
    rep_scale: float = 0.80
    ca_o_r0: float = 2.40
    depth_carboxylate: float = 6.0
    depth_other_o: float = 2.5
    morse_a: float = 1.6
    morse_cut: float = 7.0
    coulomb: bool = True
    eps_r: float = 78.5
    ionic_strength: float = 0.10
    coul_cut: float = 12.0

    def debye_length(self) -> float:
        return 3.04 / math.sqrt(self.ionic_strength) if self.ionic_strength > 0 else 1e9

    def as_table(self) -> list[dict]:
        labels = {
            "k_bond": ("Bond spring constant", "kcal/mol/Å²"),
            "k_angle": ("1-3 (angle) spring constant", "kcal/mol/Å²"),
            "k_ring14": ("Ring 1-4 spring constant", "kcal/mol/Å²"),
            "k_torsion14": ("Torsional 1-4 spring constant", "kcal/mol/Å²"),
            "k_rep": ("Steric repulsion constant", "kcal/mol/Å²"),
            "rep_scale": ("Contact distance scale (× sum of vdW radii)", "–"),
            "ca_o_r0": ("Ca–O equilibrium distance", "Å"),
            "depth_carboxylate": ("Ca–O(carboxylate) well depth", "kcal/mol"),
            "depth_other_o": ("Ca–O(hydroxyl/ring) well depth", "kcal/mol"),
            "morse_a": ("Morse width parameter a", "1/Å"),
            "morse_cut": ("Morse cut-off", "Å"),
            "coulomb": ("Screened electrostatics", "on/off"),
            "eps_r": ("Relative permittivity ε_r", "–"),
            "ionic_strength": ("Ionic strength", "mol/L"),
            "coul_cut": ("Electrostatic cut-off", "Å"),
        }
        rows = [{"Parameter": labels[k][0], "Value": v, "Unit": labels[k][1]} for k, v in asdict(self).items()]
        rows.append({"Parameter": "Debye screening length λ_D", "Value": round(self.debye_length(), 2), "Unit": "Å"})
        return rows


@dataclass
class Topology:
    pairs: np.ndarray       # (P,2) harmonic pairs
    r0: np.ndarray
    k: np.ndarray
    kind: np.ndarray        # 0 bond, 1 angle, 2 ring14, 3 tors14
    excl_codes: np.ndarray  # sorted i*N+j codes (i<j) excluded from non-bonded
    radii: np.ndarray
    is_ca: np.ndarray
    o_carbox: np.ndarray
    o_other: np.ndarray
    q: np.ndarray           # partial charges for electrostatics
    masses: np.ndarray
    n: int


def build_topology(mol: Molecule, p: FFParams) -> Topology:
    n = mol.n
    adj = mol.adjacency()
    X = mol.coords
    bonds = {(min(a, b), max(a, b)) for a, b in mol.bonds.tolist()}
    p13 = set()
    for j in range(n):
        nb = adj[j]
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                i, l = nb[a], nb[b]
                p13.add((min(i, l), max(i, l)))
    p13 -= bonds
    count14: dict[tuple, int] = {}
    for (j, k) in bonds:
        for i in adj[j]:
            if i == k:
                continue
            for l in adj[k]:
                if l == j or l == i:
                    continue
                key = (min(i, l), max(i, l))
                if key in bonds or key in p13:
                    continue
                count14[key] = count14.get(key, 0) + 1
    ring14 = {k for k, c in count14.items() if c >= 2}
    tors14 = set(count14) - ring14

    def arr(s):
        return np.array(sorted(s), int).reshape(-1, 2)

    groups = [(arr(bonds), p.k_bond, 0), (arr(p13), p.k_angle, 1), (arr(ring14), p.k_ring14, 2),
              (arr(tors14), p.k_torsion14, 3)]
    pairs = np.vstack([g[0] for g in groups])
    kk = np.concatenate([np.full(len(g[0]), g[1]) for g in groups])
    kind = np.concatenate([np.full(len(g[0]), g[2]) for g in groups])
    r0 = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1) if len(pairs) else np.zeros(0)
    excl = np.unique(pairs[:, 0].astype(np.int64) * n + pairs[:, 1]) if len(pairs) else np.zeros(0, np.int64)
    radii = np.array([VDW_RADII.get(e, 1.7) for e in mol.elements])
    is_ca = mol.elements == "CA"
    # carboxylate oxygens: terminal O on a C that carries two terminal O
    o_carb = np.zeros(n, bool)
    for c in np.flatnonzero(mol.elements == "C"):
        ox = [o for o in adj[c] if mol.elements[o] == "O" and sum(1 for x in adj[o] if mol.elements[x] != "H") == 1
              and not any(mol.elements[x] == "H" for x in adj[o])]
        if len(ox) == 2:
            o_carb[ox] = True
    o_other = (mol.elements == "O") & ~o_carb
    q = np.zeros(n)
    q[o_carb] = -0.5
    for i in np.flatnonzero(np.isin(mol.elements, list(METAL_IONS))):
        q[i] = float(mol.charges[i]) if mol.charges[i] != 0 else (2.0 if mol.elements[i] in ("CA", "MG", "ZN") else 1.0)
    return Topology(pairs, r0, kk, kind, excl, radii, is_ca, o_carb, o_other, q, mol.masses(), n)


class EnergyModel:
    def __init__(self, top: Topology, p: FFParams):
        self.t, self.p = top, p
        self.ca_idx = np.flatnonzero(top.is_ca)
        self.o_idx = np.flatnonzero(top.o_carbox | top.o_other)
        self.o_depth = np.where(top.o_carbox[self.o_idx], p.depth_carboxylate, p.depth_other_o)
        self.q_idx = np.flatnonzero(top.q != 0)
        self.sig_max = 2 * p.rep_scale * top.radii.max()
        self.lam = p.debye_length()
        self.nonion = ~top.is_ca
        self.evals = 0

    def _excluded(self, i, j):
        codes = i.astype(np.int64) * self.t.n + j
        pos = np.searchsorted(self.t.excl_codes, codes)
        pos = np.clip(pos, 0, max(len(self.t.excl_codes) - 1, 0))
        if len(self.t.excl_codes) == 0:
            return np.zeros(len(codes), bool)
        return self.t.excl_codes[pos] == codes

    def energy(self, X: np.ndarray, want_grad: bool = True):
        self.evals += 1
        t, p = self.t, self.p
        n = t.n
        G = np.zeros((n, 3))
        comp = {}
        # ---- harmonic network
        if len(t.pairs):
            i, j = t.pairs[:, 0], t.pairs[:, 1]
            d = X[i] - X[j]
            r = np.linalg.norm(d, axis=1)
            dr = r - t.r0
            e = t.k * dr ** 2
            for kname, kid in (("Bonds", 0), ("Angles (1-3)", 1), ("Ring 1-4", 2), ("Torsional 1-4", 3)):
                comp[kname] = float(e[t.kind == kid].sum())
            if want_grad:
                f = (2 * t.k * dr / np.maximum(r, 1e-9))[:, None] * d
                for ax in range(3):
                    G[:, ax] += np.bincount(i, f[:, ax], n) - np.bincount(j, f[:, ax], n)
        # ---- steric repulsion
        tree = cKDTree(X)
        pr = tree.query_pairs(self.sig_max, output_type="ndarray")
        erep = 0.0
        if len(pr):
            i, j = pr[:, 0], pr[:, 1]
            sw = i > j
            i, j = np.where(sw, j, i), np.where(sw, i, j)
            keep = ~self._excluded(i, j)
            # Ca–O handled by Morse
            cao = (t.is_ca[i] & ((t.o_carbox | t.o_other)[j])) | (t.is_ca[j] & ((t.o_carbox | t.o_other)[i]))
            keep &= ~cao
            i, j = i[keep], j[keep]
            d = X[i] - X[j]
            r = np.linalg.norm(d, axis=1)
            sig = p.rep_scale * (t.radii[i] + t.radii[j])
            m = r < sig
            i, j, d, r, sig = i[m], j[m], d[m], r[m], sig[m]
            ov = sig - r
            erep = float((p.k_rep * ov ** 2).sum())
            if want_grad and len(i):
                f = (-2 * p.k_rep * ov / np.maximum(r, 1e-9))[:, None] * d
                for ax in range(3):
                    G[:, ax] += np.bincount(i, f[:, ax], n) - np.bincount(j, f[:, ax], n)
        comp["Steric repulsion"] = erep
        # ---- Ca–O Morse
        emorse = 0.0
        if len(self.ca_idx) and len(self.o_idx):
            to = cKDTree(X[self.o_idx])
            tc = cKDTree(X[self.ca_idx])
            sdm = tc.sparse_distance_matrix(to, p.morse_cut, output_type="ndarray")
            if len(sdm):
                ci = self.ca_idx[sdm["i"]]
                oi = self.o_idx[sdm["j"]]
                D = self.o_depth[sdm["j"]]
                d = X[ci] - X[oi]
                r = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
                ex = np.exp(-p.morse_a * (r - p.ca_o_r0))
                emorse = float((D * ((1 - ex) ** 2 - 1)).sum())
                if want_grad:
                    dEdr = 2 * D * p.morse_a * (1 - ex) * ex
                    f = (dEdr / r)[:, None] * d
                    for ax in range(3):
                        G[:, ax] += np.bincount(ci, f[:, ax], n) - np.bincount(oi, f[:, ax], n)
        comp["Ca–O binding (Morse)"] = emorse
        # ---- screened electrostatics
        ecoul = 0.0
        if p.coulomb and len(self.q_idx) > 1:
            tq = cKDTree(X[self.q_idx])
            prq = tq.query_pairs(p.coul_cut, output_type="ndarray")
            if len(prq):
                i, j = self.q_idx[prq[:, 0]], self.q_idx[prq[:, 1]]
                sw = i > j
                i, j = np.where(sw, j, i), np.where(sw, i, j)
                keep = ~self._excluded(i, j)
                i, j = i[keep], j[keep]
                d = X[i] - X[j]
                r = np.maximum(np.linalg.norm(d, axis=1), 0.5)
                qq = t.q[i] * t.q[j] * COULOMB_KCAL / p.eps_r
                sc = np.exp(-r / self.lam)
                e = qq * sc / r
                ecoul = float(e.sum())
                if want_grad:
                    dEdr = -qq * sc * (1 / r ** 2 + 1 / (self.lam * r))
                    f = (dEdr / r)[:, None] * d
                    for ax in range(3):
                        G[:, ax] += np.bincount(i, f[:, ax], n) - np.bincount(j, f[:, ax], n)
        comp["Electrostatics (Debye–Hückel)"] = ecoul
        E = float(sum(comp.values()))
        return E, G, comp


@dataclass
class RelaxResult:
    frames: list
    frame_labels: list
    history: list
    final: np.ndarray
    converged: bool
    message: str
    n_iter: int
    seconds: float
    engine: str
    mol: Molecule | None = None    # molecule the frames refer to (may include added H)
    units: str = "kcal/mol"
    extra: dict = field(default_factory=dict)


def minimize(mol: Molecule, p: FFParams | None = None, max_iter: int = 300, n_frames: int = 20,
             gtol: float = 0.05, progress=None) -> RelaxResult:
    p = p or FFParams()
    top = build_topology(mol, p)
    model = EnergyModel(top, p)
    x0 = mol.coords.ravel().copy()
    cache = {"x": None, "E": None, "G": None, "comp": None}

    def fun(x):
        E, G, comp = model.energy(x.reshape(-1, 3))
        cache.update(x=x.copy(), E=E, G=G, comp=comp)
        return E, G.ravel()

    E0, G0, comp0 = model.energy(mol.coords)
    history = [dict(Iteration=0, **{"Total energy": E0}, **comp0,
                    **{"RMS gradient": float(np.sqrt((G0 ** 2).sum(1).mean()))})]
    frames = [mol.coords.copy()]
    labels = ["Start (rigid)"]
    stride = max(1, int(round(max_iter / max(n_frames - 1, 1))))
    it = {"k": 0}
    t0 = time.time()

    def cb(xk, *args):
        it["k"] += 1
        if cache["x"] is not None and np.array_equal(cache["x"], xk):
            E, G, comp = cache["E"], cache["G"], cache["comp"]
        else:
            E, G, comp = model.energy(xk.reshape(-1, 3))
        history.append(dict(Iteration=it["k"], **{"Total energy": E}, **comp,
                            **{"RMS gradient": float(np.sqrt((G ** 2).sum(1).mean()))}))
        if it["k"] % stride == 0:
            frames.append(xk.reshape(-1, 3).copy())
            labels.append(f"Iteration {it['k']}")
        if progress is not None:
            progress(min(it["k"] / max_iter, 1.0), f"Iteration {it['k']} · E = {E:,.1f} kcal/mol")

    res = sp_minimize(fun, x0, jac=True, method="L-BFGS-B", callback=cb,
                      options={"maxiter": int(max_iter), "gtol": gtol, "maxcor": 20, "ftol": 1e-10})
    final = res.x.reshape(-1, 3)
    if not np.array_equal(frames[-1], final):
        frames.append(final.copy())
        labels.append(f"Final (iteration {it['k']})")
    return RelaxResult(frames, labels, history, final, bool(res.success), str(res.message), it["k"],
                       time.time() - t0, "AlgiTrap-ENM (L-BFGS)", mol=mol, extra={"evals": model.evals})


def run_langevin(mol: Molecule, coords: np.ndarray, p: FFParams | None = None, n_steps: int = 600,
                 dt_fs: float = 1.0, temperature: float = 298.15, gamma_ps: float = 5.0,
                 save_every: int = 30, seed: int = 11, progress=None) -> RelaxResult:
    """BAOAB Langevin dynamics on the AlgiTrap-ENM surface (implicit solvent)."""
    p = p or FFParams()
    top = build_topology(mol.with_coords(coords), p)
    model = EnergyModel(top, p)
    rng = np.random.default_rng(seed)
    m = top.masses[:, None]
    conv = 4.184e-4                      # (kcal/mol/Å/amu) → Å/fs²
    kT = R_KCAL * temperature
    sig_v = np.sqrt(kT * conv / m)       # Å/fs
    c1 = math.exp(-gamma_ps * 1e-3 * dt_fs)
    c2 = math.sqrt(1 - c1 * c1)
    x = coords.copy()
    v = rng.normal(size=x.shape) * sig_v
    E, G, comp = model.energy(x)
    a = -G / m * conv
    frames, labels, hist = [x.copy()], ["t = 0 ps"], []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        v += 0.5 * dt_fs * a
        x += 0.5 * dt_fs * v
        v = c1 * v + c2 * sig_v * rng.normal(size=x.shape)
        x += 0.5 * dt_fs * v
        E, G, comp = model.energy(x)
        a = -G / m * conv
        v += 0.5 * dt_fs * a
        if step % save_every == 0 or step == n_steps:
            ke = 0.5 * float((m * v ** 2).sum()) / conv
            dof = 3 * top.n - 3
            T_inst = 2 * ke / (dof * R_KCAL)
            hist.append({"Step": step, "Time (ps)": step * dt_fs / 1000, "Potential energy": E,
                         "Kinetic energy": ke, "Temperature (K)": T_inst})
            frames.append(x.copy())
            labels.append(f"t = {step * dt_fs / 1000:.2f} ps")
            if progress is not None:
                progress(step / n_steps, f"MD step {step}/{n_steps} · T = {T_inst:.0f} K")
    return RelaxResult(frames, labels, hist, x.copy(), True, "completed", n_steps, time.time() - t0,
                       "AlgiTrap-ENM Langevin (BAOAB)", mol=mol,
                       extra={"dt_fs": dt_fs, "temperature": temperature, "gamma_ps": gamma_ps})


def quick_clash_relief(mol: Molecule, iters: int = 150) -> np.ndarray:
    p = FFParams(k_torsion14=0.5, coulomb=False)
    return minimize(mol, p, max_iter=iters, n_frames=2).final


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def ca_coordination(mol: Molecule, coords: np.ndarray, cutoff: float = 3.0, chain_of_atom=None) -> list[dict]:
    """For every Ca²⁺: number of O within cutoff and number of distinct chains donating O."""
    ca = np.flatnonzero(mol.elements == "CA")
    ox = np.flatnonzero(mol.elements == "O")
    if len(ca) == 0 or len(ox) == 0:
        return []
    tree = cKDTree(coords[ox])
    out = []
    lists = tree.query_ball_point(coords[ca], cutoff)
    for k, ids in zip(ca, lists):
        o = ox[ids]
        chains = set()
        if chain_of_atom is not None:
            chains = {int(chain_of_atom[x]) for x in o}
        dists = np.linalg.norm(coords[o] - coords[k], axis=1) if len(o) else np.array([])
        out.append({"Ca atom": int(k), "O within cutoff": int(len(o)), "Chains bridged": len(chains),
                    "Mean Ca–O (Å)": round(float(dists.mean()), 3) if len(dists) else float("nan")})
    return out


def radial_distribution(coords: np.ndarray, a_idx, b_idx, r_max: float = 8.0, nbins: int = 80):
    """Histogram-based g(r)-like pair distribution (normalised counts per Ca)."""
    if len(a_idx) == 0 or len(b_idx) == 0:
        return np.linspace(0, r_max, nbins), np.zeros(nbins)
    tree = cKDTree(coords[b_idx])
    sdm = cKDTree(coords[a_idx]).sparse_distance_matrix(tree, r_max, output_type="ndarray")
    edges = np.linspace(0, r_max, nbins + 1)
    h, _ = np.histogram(sdm["v"], bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 * np.pi * centers ** 2 * (edges[1] - edges[0])
    return centers, h / max(len(a_idx), 1) / shell


# ---------------------------------------------------------------------------
# Optional external engines
# ---------------------------------------------------------------------------
def available_engines() -> dict:
    eng = {"AlgiTrap-ENM (built-in)": True}
    try:
        from openbabel import openbabel  # noqa: F401
        eng["OpenBabel UFF"] = True
    except Exception:
        eng["OpenBabel UFF"] = False
    try:
        from rdkit import Chem  # noqa: F401
        eng["RDKit UFF"] = True
        eng["RDKit MMFF94"] = True
    except Exception:
        eng["RDKit UFF"] = False
        eng["RDKit MMFF94"] = False
    return eng


def to_rdkit(mol: Molecule, add_h: bool = True):
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    rw = Chem.RWMol()
    for e, q in zip(mol.elements, mol.charges):
        a = Chem.Atom(e.title() if len(e) > 1 else e)
        a.SetFormalCharge(int(q))
        rw.AddAtom(a)
    bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    for (a, b), o in zip(mol.bonds.tolist(), mol.orders.tolist()):
        rw.AddBond(int(a), int(b), bt.get(int(o), Chem.BondType.SINGLE))
    rm = rw.GetMol()
    Chem.SanitizeMol(rm)
    conf = Chem.Conformer(mol.n)
    for i, (x, y, z) in enumerate(mol.coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    rm.AddConformer(conf, assignId=True)
    if add_h and not (mol.elements == "H").any():
        rm = Chem.AddHs(rm, addCoords=True)
    return rm


def _extend_molecule_with_h(mol: Molecule, rm) -> Molecule:
    """Return a Molecule matching an RDKit molecule that may contain extra H atoms."""
    from .molio import make_molecule
    n0 = mol.n
    n = rm.GetNumAtoms()
    if n == n0:
        return mol
    pos = rm.GetConformer().GetPositions()
    els, names, rn, ri, ch = list(mol.elements), list(mol.names), list(mol.resnames), list(mol.resids), list(mol.chains)
    q = list(mol.charges)
    for k in range(n0, n):
        nb = rm.GetAtomWithIdx(k).GetNeighbors()[0].GetIdx()
        els.append("H"); names.append("H" + str(mol.names[nb])[1:3]); rn.append(mol.resnames[nb])
        ri.append(mol.resids[nb]); ch.append(mol.chains[nb]); q.append(0)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in rm.GetBonds()]
    orders = [int(round(b.GetBondTypeAsDouble())) if b.GetBondTypeAsDouble() != 1.5 else 4 for b in rm.GetBonds()]
    out = make_molecule(els, pos, names, rn, ri, ch, q, bonds, orders, title=mol.title)
    if mol.segids is not None:
        out.segids = np.array(list(mol.segids) + [mol.segids[rm.GetAtomWithIdx(k).GetNeighbors()[0].GetIdx()]
                                                 for k in range(n0, n)], dtype=object)
    return out


def run_external(engine: str, mol: Molecule, max_iter: int = 300, n_frames: int = 20, progress=None) -> RelaxResult:
    t0 = time.time()
    chunk = max(1, max_iter // max(n_frames - 1, 1))
    frames, labels, hist = [], [], []
    if engine.startswith("RDKit"):
        from rdkit.Chem import AllChem
        rm = to_rdkit(mol, add_h=True)
        mol2 = _extend_molecule_with_h(mol, rm)
        if "MMFF" in engine:
            props = AllChem.MMFFGetMoleculeProperties(rm)
            if props is None:
                raise RuntimeError("MMFF94 has no parameters for this system.")
            ff = AllChem.MMFFGetMoleculeForceField(rm, props, nonBondedThresh=9.0, ignoreInterfragInteractions=False)
        else:
            if not AllChem.UFFHasAllMoleculeParams(rm):
                raise RuntimeError("UFF is missing parameters for some atoms.")
            ff = AllChem.UFFGetMoleculeForceField(rm, vdwThresh=10.0, ignoreInterfragInteractions=False)
        ff.Initialize()
        X = np.array(ff.Positions()).reshape(-1, 3)
        frames.append(X.copy()); labels.append("Start (rigid)")
        hist.append({"Iteration": 0, "Total energy": ff.CalcEnergy()})
        done, k = 0, 0
        while done < max_iter:
            flag = ff.Minimize(maxIts=chunk)
            done += chunk
            k += 1
            X = np.array(ff.Positions()).reshape(-1, 3)
            frames.append(X.copy()); labels.append(f"Iteration {done}")
            hist.append({"Iteration": done, "Total energy": ff.CalcEnergy()})
            if progress:
                progress(min(done / max_iter, 1), f"{engine}: {done} iterations")
            if flag == 0:
                break
        return RelaxResult(frames, labels, hist, X, flag == 0, "converged" if flag == 0 else "max iterations",
                           done, time.time() - t0, engine, mol=mol2)
    if engine.startswith("OpenBabel"):
        from openbabel import openbabel as ob
        from openbabel import pybel
        from .molio import write_sdf
        pm = pybel.readstring("sdf", write_sdf(mol))
        ff = ob.OBForceField.FindForceField("UFF")
        if ff is None or not ff.Setup(pm.OBMol):
            raise RuntimeError("OpenBabel could not set up UFF for this system.")
        unit = ff.GetUnit()
        scale = 1 / 4.184 if "kJ" in unit else 1.0
        X = np.array([a.coords for a in pm.atoms])
        frames.append(X.copy()); labels.append("Start (rigid)")
        hist.append({"Iteration": 0, "Total energy": ff.Energy(False) * scale})
        ff.ConjugateGradientsInitialize(int(max_iter), 1e-6)
        done, more = 0, True
        while more and done < max_iter:
            more = ff.ConjugateGradientsTakeNSteps(chunk)
            done += chunk
            ff.GetCoordinates(pm.OBMol)
            X = np.array([a.coords for a in pm.atoms])
            frames.append(X.copy()); labels.append(f"Iteration {done}")
            hist.append({"Iteration": done, "Total energy": ff.Energy(False) * scale})
            if progress:
                progress(min(done / max_iter, 1), f"OpenBabel UFF: {done} steps")
        return RelaxResult(frames, labels, hist, X, not more, "converged" if not more else "max iterations",
                           done, time.time() - t0, engine, mol=mol)
    raise ValueError(f"Unknown engine {engine}")
