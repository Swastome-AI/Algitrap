"""Small geometry helpers (NeRF placement, Kabsch fit, dihedrals, PCA)."""
from __future__ import annotations

import numpy as np


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def nerf(a, b, c, bond: float, angle_deg: float, torsion_deg: float) -> np.ndarray:
    """Place atom d so that |cd| = bond, angle(b,c,d) = angle, dihedral(a,b,c,d) = torsion."""
    a, b, c = (np.asarray(x, float) for x in (a, b, c))
    ang, tor = np.radians(angle_deg), np.radians(torsion_deg)
    bc = unit(c - b)
    n = unit(np.cross(b - a, bc))
    m = np.cross(n, bc)
    d2 = np.array([-bond * np.cos(ang), bond * np.sin(ang) * np.cos(tor), bond * np.sin(ang) * np.sin(tor)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def dihedral(a, b, c, d) -> float:
    b0 = np.asarray(a, float) - np.asarray(b, float)
    b1 = np.asarray(c, float) - np.asarray(b, float)
    b2 = np.asarray(d, float) - np.asarray(c, float)
    b1n = unit(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def angle(a, b, c) -> float:
    u, v = unit(np.asarray(a, float) - b), unit(np.asarray(c, float) - b)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1))))


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Return (R, t) minimising |R P + t − Q| (rows are points)."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = qc - R @ pc
    return R, t


def rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    axis = unit(np.asarray(axis, float))
    a = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def principal_axes(X: np.ndarray):
    """Return (center, axes[3x3] rows sorted by decreasing variance, eigenvalues)."""
    c = X.mean(0)
    cov = np.cov((X - c).T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return c, v[:, order].T, w[order]


def align_to_axes(X: np.ndarray, long_axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate X so its principal axis lies along x (long_axis=0) or z (2).
    Returns (X_new, R, center) with X_new = (X - center) @ R.T"""
    c, ax, _ = principal_axes(X)
    R = ax.copy()  # rows become new basis
    if np.linalg.det(R) < 0:
        R[2] *= -1
    if long_axis == 2:
        R = np.array([R[1], R[2], R[0]])
        if np.linalg.det(R) < 0:
            R[0] *= -1
    Xn = (X - c) @ R.T
    return Xn, R, c


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def min_enclosing_circle(points: np.ndarray, return_center: bool = False):
    """Radius of the minimal enclosing circle of 2-D points (Welzl, iterative)."""
    P = np.asarray(points, float)
    if len(P) == 0:
        return (np.zeros(2), 0.0) if return_center else 0.0
    rng = np.random.default_rng(0)
    P = P[rng.permutation(len(P))]

    def circ2(a, b):
        c = (a + b) / 2
        return c, np.linalg.norm(a - c)

    def circ3(a, b, c):
        ax, ay = a; bx, by = b; cx, cy = c
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            pts = np.array([a, b, c])
            dm = ((pts[:, None] - pts[None]) ** 2).sum(-1)
            i, j = np.unravel_index(dm.argmax(), dm.shape)
            return circ2(pts[i], pts[j])
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        cc = np.array([ux, uy])
        return cc, np.linalg.norm(a - cc)

    eps = 1e-9
    c, r = P[0], 0.0
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - c) > r + eps:
            c, r = P[i], 0.0
            for j in range(i):
                if np.linalg.norm(P[j] - c) > r + eps:
                    c, r = circ2(P[i], P[j])
                    for k in range(j):
                        if np.linalg.norm(P[k] - c) > r + eps:
                            c, r = circ3(P[i], P[j], P[k])
    return (np.asarray(c, float), float(r)) if return_center else float(r)


def tetra_pair(center, n1, n2, alpha_deg: float = 54.75):
    """Two remaining tetrahedral directions at `center` given two bonded neighbours."""
    center = np.asarray(center, float)
    u = unit((center - n1) + (center - n2))
    w = unit(np.cross(n1 - center, n2 - center))
    a = np.radians(alpha_deg)
    return unit(np.cos(a) * u + np.sin(a) * w), unit(np.cos(a) * u - np.sin(a) * w)


def tetra_third(center, n1, n2, n3):
    """Fourth tetrahedral direction at `center` given three neighbours."""
    center = np.asarray(center, float)
    return unit(3 * center - n1 - n2 - n3)
