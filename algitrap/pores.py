"""Geometric pore analysis of the hydrogel network and microplastic capture
prediction.

* Distance field: for every grid point, the distance to the nearest atom
  surface (vdW radius; Ca²⁺ uses its ionic radius).
* PLD (pore-limiting diameter): the largest sphere that can travel all the
  way through the network along the filtration axis (z). A particle whose
  smallest cross-section exceeds the PLD cannot pass – it is captured by
  size exclusion.
* LCD (largest cavity diameter) and the volume-weighted pore-size
  distribution (each void point gets the diameter of the largest empty
  sphere that covers it – the Gelb–Gubbins geometric PSD).
* Pore centres: local maxima of the distance field (count-based PSD).
* Monte-Carlo transport: biased random walkers of a given size driven
  through the mesh – an animation-ready breakthrough experiment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .constants import VDW_RADII
from .molio import Molecule


@dataclass
class PoreGrid:
    field: np.ndarray       # (nx, ny, nz) distance to nearest atom surface (Å)
    origin: np.ndarray
    h: float
    slab_k: tuple           # z-index range of the polymer slab (k0, k1)

    @property
    def shape(self):
        return self.field.shape

    def coords(self, ijk):
        return self.origin + np.asarray(ijk) * self.h


def distance_field(X: np.ndarray, radii: np.ndarray, lo: np.ndarray, hi: np.ndarray, h: float = 0.6,
                   max_points: int = 1_800_000, slab=(None, None)) -> PoreGrid:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    vol = float(np.prod(hi - lo))
    h = max(h, (vol / max_points) ** (1 / 3))
    axes = [np.arange(lo[k], hi[k] + 1e-9, h) for k in range(3)]
    shape = tuple(len(a) for a in axes)
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    # only atoms near the box matter
    rmax = float(radii.max())
    near = np.all((X > lo - 12) & (X < hi + 12), axis=1)
    Xn, rn = X[near], radii[near]
    cap = 25.0
    if len(Xn) == 0:
        f = np.full(len(G), cap)
    else:
        tree = cKDTree(Xn)
        k = min(8, len(Xn))
        f = np.empty(len(G))
        for s in range(0, len(G), 400_000):
            d, idx = tree.query(G[s:s + 400_000], k=k, distance_upper_bound=cap + rmax)
            d = np.atleast_2d(d.T).T if k == 1 else d
            idx = np.atleast_2d(idx.T).T if k == 1 else idx
            valid = idx < len(Xn)
            rr = np.where(valid, rn[np.minimum(idx, len(Xn) - 1)], 0.0)
            val = np.where(valid, d - rr, cap)
            f[s:s + 400_000] = np.minimum(val.min(1), cap)
    field = f.reshape(shape)
    zl, zh = slab
    k0 = 0 if zl is None else int(max(0, math.floor((zl - lo[2]) / h)))
    k1 = shape[2] - 1 if zh is None else int(min(shape[2] - 1, math.ceil((zh - lo[2]) / h)))
    return PoreGrid(field, lo, h, (k0, k1))


def percolates(grid: PoreGrid, r: float) -> bool:
    acc = grid.field >= r
    if not acc[:, :, 0].any() or not acc[:, :, -1].any():
        return False
    lab, n = ndimage.label(acc)
    top = set(np.unique(lab[:, :, -1])) - {0}
    bot = set(np.unique(lab[:, :, 0])) - {0}
    return bool(top & bot)


def pore_limiting_diameter(grid: PoreGrid, tol: float = 0.05) -> float:
    lo_r, hi_r = 0.0, float(grid.field.max())
    if not percolates(grid, 0.0):
        return 0.0
    while hi_r - lo_r > tol:
        mid = 0.5 * (lo_r + hi_r)
        if percolates(grid, mid):
            lo_r = mid
        else:
            hi_r = mid
    return 2 * lo_r


def slab_view(grid: PoreGrid) -> np.ndarray:
    k0, k1 = grid.slab_k
    return grid.field[:, :, k0:k1 + 1]


def largest_cavity_diameter(grid: PoreGrid) -> float:
    return 2 * float(max(slab_view(grid).max(), 0.0))


def porosity(grid: PoreGrid, probe_r: float = 0.0) -> float:
    s = slab_view(grid)
    return float((s >= probe_r).mean()) if probe_r > 0 else float((s > 0).mean())


def covering_psd(grid: PoreGrid, step: float | None = None):
    """Volume-weighted geometric PSD inside the slab.

    Returns (diameters_per_void_voxel, levels)."""
    f = grid.field
    k0, k1 = grid.slab_k
    rmax = float(slab_view(grid).max())
    if rmax <= 0:
        return np.zeros(0), np.zeros(0)
    margin = int(math.ceil(rmax / grid.h)) + 1
    z0, z1 = max(0, k0 - margin), min(f.shape[2], k1 + 1 + margin)
    sub = f[:, :, z0:z1]
    step = step or max(grid.h / 2, 0.25)
    levels = np.arange(rmax, 0, -step)
    R = np.zeros(sub.shape)
    for r in levels:
        centers = sub >= r
        if not centers.any():
            continue
        dist = ndimage.distance_transform_edt(~centers) * grid.h
        newly = (dist <= r) & (R == 0) & (sub > 0)
        R[newly] = r
    Rs = R[:, :, k0 - z0:k1 - z0 + 1]
    void = slab_view(grid) > 0
    return 2 * Rs[void], 2 * levels


def pore_centres(grid: PoreGrid, min_d: float = 1.0, window: float = 3.0) -> np.ndarray:
    s = slab_view(grid)
    size = max(3, int(round(window / grid.h)) | 1)
    mx = ndimage.maximum_filter(s, size=size, mode="nearest")
    peaks = (s == mx) & (2 * s >= min_d)
    return 2 * s[peaks]


def pore_summary(grid: PoreGrid) -> dict:
    d_vox, _ = covering_psd(grid)
    centres = pore_centres(grid)
    pld = pore_limiting_diameter(grid)
    out = {
        "PLD – pore-limiting diameter (Å)": round(pld, 2),
        "LCD – largest cavity diameter (Å)": round(largest_cavity_diameter(grid), 2),
        "Porosity (void fraction in slab)": round(porosity(grid), 3),
        "Volume-weighted mean pore diameter (Å)": round(float(d_vox.mean()), 2) if len(d_vox) else float("nan"),
        "Median pore diameter d50 (Å)": round(float(np.median(d_vox)), 2) if len(d_vox) else float("nan"),
        "d10 (Å)": round(float(np.percentile(d_vox, 10)), 2) if len(d_vox) else float("nan"),
        "d90 (Å)": round(float(np.percentile(d_vox, 90)), 2) if len(d_vox) else float("nan"),
        "Pore centres detected": int(len(centres)),
        "Mean pore-centre diameter (Å)": round(float(centres.mean()), 2) if len(centres) else float("nan"),
        "Grid spacing (Å)": round(grid.h, 3),
    }
    return {"summary": out, "psd": d_vox, "centres": centres, "pld": pld}


# ---------------------------------------------------------------------------
# Capture physics
# ---------------------------------------------------------------------------
def renkin(lam: float) -> dict:
    """Renkin (1954) steric partition and hindered diffusion for a sphere in a cylindrical pore."""
    if lam >= 1:
        return {"Partition coefficient Φ": 0.0, "Hindrance factor K_d": 0.0, "D_eff / D0": 0.0}
    phi = (1 - lam) ** 2
    kd = 1 - 2.104 * lam + 2.09 * lam ** 3 - 0.95 * lam ** 5
    return {"Partition coefficient Φ": round(phi, 4), "Hindrance factor K_d": round(max(kd, 0), 4),
            "D_eff / D0": round(phi * max(kd, 0), 4)}


def ogston(d_particle: float, fiber_d: float, phi_polymer: float) -> float:
    """Ogston partition coefficient for a sphere among random fibres."""
    rs, rf = d_particle / 2, fiber_d / 2
    return float(math.exp(-phi_polymer * (1 + rs / rf) ** 2))


def capture_verdict(d_particle: float, pld: float, psd: np.ndarray) -> dict:
    blocked = d_particle > pld
    acc = float((psd >= d_particle).mean()) if len(psd) else 0.0
    lam = d_particle / pld if pld > 0 else float("inf")
    out = {"Particle diameter (Å)": round(d_particle, 2), "PLD (Å)": round(pld, 2),
           "λ = d / PLD": round(lam, 3), "Size-exclusion capture": "YES" if blocked else "NO",
           "Accessible pore-volume fraction": round(acc, 4),
           "Safety margin (d − PLD, Å)": round(d_particle - pld, 2)}
    out.update(renkin(lam))
    return out


def transport_mc(grid: PoreGrid, r_probe: float, n_walkers: int = 300, n_steps: int = 3000, drift: float = 0.35,
                 seed: int = 0, n_record: int = 25, record_every: int = 30) -> dict:
    """Biased lattice random walk of particle centres through probe-accessible space."""
    acc = grid.field >= r_probe
    nx, ny, nz = acc.shape
    rng = np.random.default_rng(seed)
    top = np.argwhere(acc[:, :, nz - 2])
    if len(top) == 0:
        return {"transmission": 0.0, "passed": 0, "walkers": 0, "mean_steps": float("nan"), "frames": [],
                "note": "no accessible start positions"}
    pick = top[rng.integers(0, len(top), n_walkers)]
    pos = np.column_stack([pick, np.full(n_walkers, nz - 2)])
    moves = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    pr = np.array([1, 1, 1, 1, 1 - drift, 1 + drift]) / 6.0
    cum = np.cumsum(pr)
    active = np.ones(n_walkers, bool)
    passed_at = np.full(n_walkers, -1)
    frames = []
    rec = min(n_record, n_walkers)
    for step in range(n_steps):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        choice = np.searchsorted(cum, rng.random(len(idx)))
        cand = pos[idx] + moves[np.minimum(choice, 5)]
        cand[:, 0] = np.clip(cand[:, 0], 0, nx - 1)
        cand[:, 1] = np.clip(cand[:, 1], 0, ny - 1)
        cand[:, 2] = np.clip(cand[:, 2], 0, nz - 1)
        ok = acc[cand[:, 0], cand[:, 1], cand[:, 2]]
        pos[idx[ok]] = cand[ok]
        done = pos[idx, 2] <= 1
        passed_at[idx[done]] = step
        active[idx[done]] = False
        if step % record_every == 0:
            frames.append(grid.origin + pos[:rec] * grid.h)
    frames.append(grid.origin + pos[:rec] * grid.h)
    npass = int((passed_at >= 0).sum())
    return {"transmission": npass / n_walkers, "passed": npass, "walkers": n_walkers,
            "mean_steps": float(passed_at[passed_at >= 0].mean()) if npass else float("nan"), "frames": frames,
            "passed_mask": (passed_at[:rec] >= 0)}


def grid_for(mol: Molecule, region: dict, X: np.ndarray | None = None, h: float = 0.6, pad: float = 10.0,
             max_points: int = 1_800_000) -> PoreGrid:
    X = mol.coords if X is None else X
    radii = np.array([VDW_RADII.get(e, 1.7) for e in mol.elements])
    lo = np.array(region["lo"], float).copy()
    hi = np.array(region["hi"], float).copy()
    zlo, zhi = float(X[:, 2].min()), float(X[:, 2].max())
    lo[2], hi[2] = zlo - pad, zhi + pad
    return distance_field(X, radii, lo, hi, h=h, max_points=max_points, slab=(zlo, zhi))
