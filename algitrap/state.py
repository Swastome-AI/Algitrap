"""Session-state helpers shared by all tabs."""
from __future__ import annotations

import numpy as np
import streamlit as st

from .molio import write_pdb, write_pdb_models

PRESETS = {
    "Quick (school demo)": dict(short="q", dp=12, grid=5, spacing=16.0, layers=2, iters=250, frames=16, pore_h=0.6,
                                walkers=200, mc_steps=2500, md_steps=800, sweep_n=6, dyn_frames=6),
    "Research (pro)": dict(short="r", dp=24, grid=8, spacing=18.0, layers=3, iters=1500, frames=30, pore_h=0.4,
                           walkers=600, mc_steps=6000, md_steps=3000, sweep_n=10, dyn_frames=12),
}
STEPS = ["chain", "net", "relax", "relaxed_net", "md", "pores", "capture_curve", "dyn", "sweep", "mc", "gif"]


def init():
    ss = st.session_state
    ss.setdefault("preset", "Quick (school demo)")
    ss.setdefault("user_name", "")
    ss.setdefault("export_dpi", 300)
    ss["figs"] = {}
    ss["tables"] = {}


def P() -> dict:
    return PRESETS.get(st.session_state.get("preset"), PRESETS["Quick (school demo)"])


def k(name: str) -> str:
    """Widget key that depends on the preset so defaults switch with it."""
    return f"{name}_{P()['short']}"


def invalidate(step: str):
    i = STEPS.index(step)
    for s in STEPS[i:]:
        st.session_state.pop(s, None)


def get(name, default=None):
    return st.session_state.get(name, default)


def plastic_type() -> str:
    ov = st.session_state.get("mp_type_override", "Auto-detect")
    if ov and ov != "Auto-detect":
        return ov
    mp = st.session_state.get("mp")
    if mp:
        return mp["ident"]["label"]
    return "PS"


def network_pdb(net, coords=None) -> str:
    bf = np.where(net.atom_chain >= 0, net.atom_chain, 0).astype(float)
    return write_pdb(net.mol, coords=coords, conect=False, bfac=bf)


def network_traj_pdb(net, frames, max_atoms_total: int = 90_000) -> tuple[str, int]:
    n = net.mol.n
    k = max(2, min(len(frames), max_atoms_total // max(n, 1)))
    idx = np.unique(np.linspace(0, len(frames) - 1, k).round().astype(int))
    bf = np.where(net.atom_chain >= 0, net.atom_chain, 0).astype(float)
    return write_pdb_models(net.mol, [frames[i] for i in idx], bfac=bf), len(idx)
