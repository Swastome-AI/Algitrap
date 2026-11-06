"""UV-C (254 nm) photodegradation: dose maths, a transparent kinetic model
for planning experiments, fitting of real lab data, FTIR oxidation indices
and microscope-image particle analysis (numpy/scipy only)."""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import curve_fit

from .constants import C_LIGHT, H_PLANCK, N_A, PLASTIC_INFO, R_GAS

# Illustrative planning defaults (replace with your measured values).
UV_DEFAULTS = {
    "PS": {"k_ref_per_h": 0.020, "ci_max": 0.60, "mass_loss_max": 12.0, "scission_per_h": 0.0040},
    "PE": {"k_ref_per_h": 0.006, "ci_max": 0.35, "mass_loss_max": 5.0, "scission_per_h": 0.0012},
    "PP": {"k_ref_per_h": 0.010, "ci_max": 0.45, "mass_loss_max": 7.0, "scission_per_h": 0.0020},
    "Other": {"k_ref_per_h": 0.008, "ci_max": 0.40, "mass_loss_max": 6.0, "scission_per_h": 0.0015},
}


def photon_energy_j(wavelength_nm: float = 254.0) -> float:
    return H_PLANCK * C_LIGHT / (wavelength_nm * 1e-9)


def dose_table(irradiance_mw_cm2: float, hours: list[float], wavelength_nm: float = 254.0,
               gel_absorbance_per_mm: float = 0.10, gel_thickness_mm: float = 2.0) -> pd.DataFrame:
    trans = 10 ** (-gel_absorbance_per_mm * gel_thickness_mm)
    e_ph = photon_energy_j(wavelength_nm)
    rows = []
    for h in hours:
        dose = irradiance_mw_cm2 * h * 3600 / 1000          # J/cm² at the surface
        eff = dose * trans
        rows.append({"Time (h)": h, "Surface dose (J/cm²)": round(dose, 2), "Dose at particles (J/cm²)": round(eff, 2),
                     "Photons at particles (einstein/cm²)": eff / e_ph / N_A})
    return pd.DataFrame(rows)


def rate_constant(polymer: str, irradiance: float, temp_c: float, h2o2_mm: float, gel_trans: float,
                  ea_kj: float = 25.0, i_ref: float = 2.0, t_ref_c: float = 25.0, ox_gain: float = 1.5,
                  k_half_mm: float = 5.0, custom_k_ref: float | None = None) -> float:
    """Pseudo-first-order rate constant (1/h): k = k_ref·(I/I_ref)·Arrhenius·(1 + gain·[H₂O₂]/(K+[H₂O₂]))·T_gel."""
    k_ref = custom_k_ref if custom_k_ref is not None else UV_DEFAULTS.get(polymer, UV_DEFAULTS["Other"])["k_ref_per_h"]
    arr = math.exp(-ea_kj * 1000 / R_GAS * (1 / (temp_c + 273.15) - 1 / (t_ref_c + 273.15)))
    ox = 1 + ox_gain * h2o2_mm / (k_half_mm + h2o2_mm) if h2o2_mm > 0 else 1.0
    return k_ref * (irradiance / i_ref) * arr * ox * gel_trans


def simulate(polymer: str, k: float, hours: np.ndarray, scission_scale: float = 1.0) -> pd.DataFrame:
    d = UV_DEFAULTS.get(polymer, UV_DEFAULTS["Other"])
    k_ref = d["k_ref_per_h"]
    ci = d["ci_max"] * (1 - np.exp(-k * hours))
    ml = d["mass_loss_max"] * (1 - np.exp(-0.5 * k * hours))
    ks = d["scission_per_h"] * (k / k_ref) * scission_scale
    mn = 1 / (1 + ks * hours * 50)                      # Mn/Mn0 for random scission (DPn0 ≈ 50 bonds scale)
    scis = 1 / mn - 1
    return pd.DataFrame({"Time (h)": hours, "Carbonyl index": ci, "Mass loss (%)": ml, "Mn / Mn0": mn,
                         "Scissions per chain": scis, "Remaining mass (%)": 100 - ml})


def fit_first_order(t: np.ndarray, y: np.ndarray, mode: str = "decay") -> dict:
    """decay: y = y0·e^(−kt); decay to plateau: y = y∞ + (y0 − y∞)e^(−kt); growth: y = ymax(1 − e^(−kt)) + y0."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 3:
        raise ValueError("Need at least 3 time points to fit kinetics.")
    if mode == "decay":
        f = lambda tt, y0, k: y0 * np.exp(-k * tt)
        p0 = [max(y.max(), 1e-6), 0.01]
    elif mode == "zero-order":
        f = lambda tt, y0, k: y0 - k * tt
        p0 = [y.max(), 0.01]
    elif mode == "decay to plateau":
        f = lambda tt, yinf, k, y0: yinf + (y0 - yinf) * np.exp(-k * tt)
        p0 = [y.min(), 0.02, y.max()]
    else:
        f = lambda tt, ymax, k, y0: ymax * (1 - np.exp(-k * tt)) + y0
        p0 = [max(y.max() - y.min(), 1e-6), 0.02, y.min()]
    if mode == "decay to plateau":
        rng_ = max(np.ptp(y), 1e-6)
        bounds = ([y.min() - 5 * rng_, 1e-5, y.min() - rng_], [y.max() + rng_, 10.0, y.max() + 5 * rng_])
        popt, pcov = curve_fit(f, t, y, p0=[y.min() - 0.5 * rng_, 0.02, y.max()], bounds=bounds, maxfev=20000)
    else:
        popt, pcov = curve_fit(f, t, y, p0=p0, maxfev=20000)
    pred = f(t, *popt)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
    k = float(popt[1])
    se = float(np.sqrt(np.diag(pcov))[1]) if np.all(np.isfinite(pcov)) else float("nan")
    return {"k (1/h)": k, "k std. error": se, "k 95% CI low": k - 1.96 * se, "k 95% CI high": k + 1.96 * se,
            "Half-life (h)": (math.log(2) / k if k > 0 else float("inf")) if mode != "zero-order" else float("nan"), "R²": 1 - ss_res / ss_tot,
            "params": popt.tolist(), "model": mode, "n points": int(len(t)), "_f": f}


def arrhenius(temps_c: list[float], ks: list[float]) -> dict:
    T = np.array(temps_c, float) + 273.15
    k = np.array(ks, float)
    ok = k > 0
    if ok.sum() < 2:
        return {}
    slope, intercept = np.polyfit(1 / T[ok], np.log(k[ok]), 1)
    return {"Ea (kJ/mol)": -slope * R_GAS / 1000, "ln A": intercept}


# ---------------------------------------------------------------------------
# FTIR
# ---------------------------------------------------------------------------
def read_spectrum_table(data: bytes | str) -> pd.DataFrame:
    text = data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else data
    sep = "," if text.count(",") >= text.count("\t") else "\t"
    df = pd.read_csv(io.StringIO(text), sep=sep, comment="#")
    if df.shape[1] < 2:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", comment="#", engine="python")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return df


def to_absorbance(y: np.ndarray, kind: str) -> np.ndarray:
    y = np.asarray(y, float)
    if kind == "auto":
        kind = "transmittance %" if np.nanmax(y) > 5 else "absorbance"
    if kind == "transmittance %":
        return 2 - np.log10(np.clip(y, 1e-3, None))
    if kind == "transmittance (0–1)":
        return -np.log10(np.clip(y, 1e-5, None))
    return y


def band_area(wn: np.ndarray, a: np.ndarray, hi: float, lo: float) -> float:
    """Area of a band with a straight baseline between its window edges."""
    order = np.argsort(wn)
    wn, a = wn[order], a[order]
    m = (wn >= lo) & (wn <= hi)
    if m.sum() < 3:
        return float("nan")
    x, y = wn[m], a[m]
    base = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    return float(np.trapezoid(np.clip(y - base, 0, None), x) if hasattr(np, "trapezoid") else np.trapz(np.clip(y - base, 0, None), x))


def ftir_indices(wn: np.ndarray, a: np.ndarray, polymer: str) -> dict:
    ref_hi, ref_lo = PLASTIC_INFO.get(polymer, PLASTIC_INFO["Other"])["ftir_ref_band"]
    ref = band_area(wn, a, ref_hi, ref_lo)
    co = band_area(wn, a, 1850, 1650)
    oh = band_area(wn, a, 3600, 3100)
    vinyl = band_area(wn, a, 925, 895)
    def ratio(x):
        return x / ref if ref and np.isfinite(ref) and ref > 0 else float("nan")
    return {"Carbonyl index (1850–1650 / ref)": ratio(co), "Hydroxyl index (3600–3100 / ref)": ratio(oh),
            "Vinyl index (925–895 / ref)": ratio(vinyl), "Reference band area": ref}


def demo_spectrum(polymer: str, hours: float, k: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic FTIR absorbance for planning/demo (clearly labelled as simulated)."""
    rng = np.random.default_rng(seed)
    wn = np.linspace(4000, 600, 1700)

    def peak(c, w, h):
        return h * np.exp(-0.5 * ((wn - c) / w) ** 2)

    if polymer == "PS":
        base = (peak(3026, 12, 0.35) + peak(2920, 15, 0.45) + peak(1601, 6, 0.25) + peak(1493, 6, 0.55)
                + peak(1452, 7, 0.5) + peak(756, 8, 0.7) + peak(698, 8, 0.95))
    else:
        base = peak(2915, 14, 0.9) + peak(2848, 10, 0.75) + peak(1463, 7, 0.45) + peak(720, 6, 0.35)
    ox = 1 - math.exp(-k * hours)
    base = base + peak(1715, 25, 0.35 * ox) + peak(3400, 120, 0.18 * ox) + peak(1175, 30, 0.08 * ox)
    if polymer != "PS":
        base = base + peak(909, 6, 0.06 * ox)
    return wn, base + rng.normal(0, 0.004, wn.size) + 0.02


# ---------------------------------------------------------------------------
# Microscope image particle analysis
# ---------------------------------------------------------------------------
def otsu(img: np.ndarray) -> float:
    hist, edges = np.histogram(img.ravel(), bins=256)
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * edges[:-1])
    mt = mu[-1]
    sb = (mt * w - mu) ** 2 / np.where(w * (1 - w) == 0, np.nan, w * (1 - w))
    return float(edges[np.nanargmax(sb)])


def analyse_particles(img_rgb: np.ndarray, um_per_px: float, dark_particles: bool = True, min_px: int = 6) -> dict:
    img = img_rgb.astype(float)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    img = ndimage.gaussian_filter(img, 1.0)
    thr = otsu(img)
    mask = img < thr if dark_particles else img > thr
    mask = ndimage.binary_opening(mask, iterations=1)
    lab, n = ndimage.label(mask)
    areas = np.bincount(lab.ravel())[1:] if n else np.zeros(0)
    keep = areas >= min_px
    areas_um2 = areas[keep] * um_per_px ** 2
    deq = 2 * np.sqrt(areas_um2 / np.pi)
    return {"count": int(keep.sum()), "eq_diameters_um": deq, "area_fraction": float(mask.mean()),
            "threshold": thr, "mask": mask}


def demo_microscope(fragmented: bool, seed: int = 1, size: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 225.0)
    yy, xx = np.mgrid[:size, :size]
    n = 70 if fragmented else 22
    for _ in range(n):
        cx, cy = rng.uniform(15, size - 15, 2)
        r = rng.uniform(2.5, 6) if fragmented else rng.uniform(7, 13)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = rng.uniform(60, 110)
    img = ndimage.gaussian_filter(img, 1.2) + rng.normal(0, 6, img.shape)
    return np.clip(np.stack([img] * 3, -1), 0, 255).astype(np.uint8)


def demo_kinetics_csv(polymer: str = "PS", seed: int = 5) -> str:
    """A small synthetic lab-style data set for the upload format (clearly labelled)."""
    rng = np.random.default_rng(seed)
    rows = []
    for temp in (25, 35):
        for ox in (0.0, 10.0):
            k = rate_constant(polymer, 2.0, temp, ox, 0.63)
            for t in (0, 12, 24, 36, 48, 72):
                ci = UV_DEFAULTS[polymer]["ci_max"] * (1 - math.exp(-k * t)) + rng.normal(0, 0.01)
                rows.append({"time_h": t, "temperature_C": temp, "oxidant": "H2O2 10 mM" if ox else "No oxidant",
                             "carbonyl_index": round(max(ci, 0), 4),
                             "mass_remaining_pct": round(100 - UV_DEFAULTS[polymer]["mass_loss_max"]
                                                         * (1 - math.exp(-0.5 * k * t)) + rng.normal(0, 0.3), 3)})
    return "# SIMULATED DEMO DATA – replace with your own measurements\n" + pd.DataFrame(rows).to_csv(index=False)
