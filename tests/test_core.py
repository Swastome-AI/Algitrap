"""Core scientific tests (run with `python tests/test_core.py` or `pytest`)."""
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algitrap.alginate import (analyze_alginate, build_chain, chain_metrics, cip_c1, cip_c5, residue_template,  # noqa: E402
                               sequence_stats)
from algitrap.figs import Fig, export_bytes  # noqa: E402
from algitrap.forcefield import EnergyModel, FFParams, build_topology, minimize  # noqa: E402
from algitrap.molio import load_structure, write_pdb, write_sdf  # noqa: E402
from algitrap.multiscale import bead_diameter_tate, mesh_size_rubber  # noqa: E402
from algitrap.network import NetworkParams, build_network  # noqa: E402
from algitrap.plastics import build_pe, build_ps, identify_polymer, size_descriptors  # noqa: E402
from algitrap.pores import capture_verdict, grid_for, pore_limiting_diameter, renkin  # noqa: E402
from algitrap.uv import fit_first_order  # noqa: E402


def test_stereochemistry():
    M, G = residue_template("M"), residue_template("G")
    assert cip_c5(M) == "S" and cip_c1(M) == "R"        # β-D-ManpA
    assert cip_c5(G) == "R" and cip_c1(G) == "R"        # α-L-GulpA (C5 epimer)


def test_helix_rise():
    assert abs(chain_metrics(build_chain("G" * 12))["Mean rise per residue (Å)"] - 4.35) < 0.15
    assert abs(chain_metrics(build_chain("M" * 12))["Mean rise per residue (Å)"] - 5.17) < 0.2


def test_pdb_sdf_roundtrip_and_recognition():
    seq = "GGMMGMGGGM"
    ch = build_chain(seq, include_h=True)
    for name, text in (("a.pdb", write_pdb(ch.mol)), ("a.sdf", write_sdf(ch.mol))):
        mol, _ = load_structure(name, text.encode())
        assert analyze_alginate(mol).sequence == seq


def test_sequence_stats():
    s = sequence_stats("GGGMMGGM")
    assert abs(s["F_G"] - 5 / 8) < 1e-9 and s["Egg-box capable GG dyads"] == 3


def test_plastics():
    ps, pe = build_ps(5, "syndiotactic"), build_pe(8)
    assert identify_polymer(ps)["label"] == "PS" and identify_polymer(pe)["label"] == "PE"
    assert size_descriptors(pe)["Min cross-section diameter (Å)"] < size_descriptors(ps)["Min cross-section diameter (Å)"]


def test_forcefield_gradient():
    ch = build_chain("GGGM")
    m = ch.mol.with_coords(ch.mol.coords + np.random.default_rng(1).normal(0, 0.05, ch.mol.coords.shape))
    p = FFParams()
    em = EnergyModel(build_topology(ch.mol, p), p)
    E, G, _ = em.energy(m.coords)
    h = 1e-5
    for i in (0, 7, 20):
        for ax in range(3):
            Xp, Xm = m.coords.copy(), m.coords.copy()
            Xp[i, ax] += h
            Xm[i, ax] -= h
            num = (em.energy(Xp, False)[0] - em.energy(Xm, False)[0]) / (2 * h)
            assert abs(num - G[i, ax]) < 1e-4


def test_network_relax_and_pores():
    ch = build_chain("GGGGMMGGGGMM")
    pld = []
    for sp in (12.0, 20.0):
        net = build_network(ch, NetworkParams(grid=4, spacing=sp))
        assert len(net.ca) > 0
        pld.append(pore_limiting_diameter(grid_for(net.mol, net.region, h=0.8)))
    assert pld[1] > pld[0] > 0
    net = build_network(ch, NetworkParams(grid=3, spacing=16))
    r = minimize(net.mol, max_iter=60, n_frames=4)
    assert r.history[-1]["Total energy"] < r.history[0]["Total energy"]


def test_capture_logic():
    psd = np.array([5.0, 8.0, 10.0])
    assert capture_verdict(12.0, 10.0, psd)["Size-exclusion capture"] == "YES"
    assert capture_verdict(6.0, 10.0, psd)["Size-exclusion capture"] == "NO"
    assert renkin(1.2)["D_eff / D0"] == 0.0


def test_multiscale_and_uv():
    assert 2.0 < bead_diameter_tate("22 G")["Gel bead diameter (mm)"] < 2.6
    assert 6 < mesh_size_rubber(10_000) < 9
    t = np.array([0, 10, 20, 40, 60, 80.0])
    y = 0.5 * (1 - np.exp(-0.03 * t))
    assert abs(fit_first_order(t, y, "growth")["k (1/h)"] - 0.03) < 1e-3


def test_figure_export():
    f = Fig("x", "Test")
    f.bar(["a", "b"], [1, 2]).vline(0.5, "v")
    from PIL import Image
    im = Image.open(io.BytesIO(export_bytes(f, "png", dpi=300)))
    assert im.size == (3000, 2100)
    assert export_bytes(f, "svg")[:5] in (b"<?xml", b"<svg ")
    assert export_bytes(f, "pdf")[:4] == b"%PDF"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("ALL CORE TESTS PASSED")
