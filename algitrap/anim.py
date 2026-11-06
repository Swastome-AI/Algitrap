"""Animations: Plotly 3-D (play/pause + slider) and animated GIF export."""
from __future__ import annotations

import io

import numpy as np

from .constants import COLOR_CA, COLOR_BAD, COLOR_GOOD


def residue_centroids(X: np.ndarray, atom_res: np.ndarray, heavy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sel = (atom_res >= 0) & heavy
    r = atom_res[sel]
    n = int(r.max()) + 1 if len(r) else 0
    cnt = np.bincount(r, minlength=n).astype(float)
    cen = np.stack([np.bincount(r, X[sel, k], minlength=n) for k in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    return cen, cnt > 0


def _chain_line_arrays(cen, res_chain):
    xs, ys, zs, cs = [], [], [], []
    for c in np.unique(res_chain):
        idx = np.flatnonzero(res_chain == c)
        xs += list(cen[idx, 0]) + [None]
        ys += list(cen[idx, 1]) + [None]
        zs += list(cen[idx, 2]) + [None]
        cs += [float(c)] * len(idx) + [float(c)]
    return xs, ys, zs, cs


def _ranges(arrs, pad=3.0):
    allx = np.vstack(arrs)
    lo, hi = allx.min(0) - pad, allx.max(0) + pad
    return [[float(lo[k]), float(hi[k])] for k in range(3)]


def _anim_layout(fig, labels, ranges, title, duration=350):
    import plotly.graph_objects as go  # noqa: F401
    steps = [dict(method="animate", label=str(k), args=[[lab], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                                     transition=dict(duration=0))])
             for k, lab in enumerate(labels)]
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=19, color="black"), x=0.01),
        paper_bgcolor="white", font=dict(color="black", size=13), height=620,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="black", borderwidth=1),
        scene=dict(xaxis=dict(range=ranges[0], title="x (Å)", gridcolor="#d0d0d0", backgroundcolor="white"),
                   yaxis=dict(range=ranges[1], title="y (Å)", gridcolor="#d0d0d0", backgroundcolor="white"),
                   zaxis=dict(range=ranges[2], title="z (Å)", gridcolor="#d0d0d0", backgroundcolor="white"),
                   aspectmode="manual",
                   aspectratio=dict(x=1, y=(ranges[1][1] - ranges[1][0]) / max(ranges[0][1] - ranges[0][0], 1),
                                    z=max(0.35, (ranges[2][1] - ranges[2][0]) / max(ranges[0][1] - ranges[0][0], 1)))),
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=duration, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0), mode="immediate")]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.18, y=0.02, len=0.8, currentvalue=dict(prefix="Frame: ", font=dict(size=13)),
                      steps=[dict(s, label=lab) for s, lab in zip(steps, labels)])],
    )
    return fig


def relaxation_animation(frames, labels, mol, atom_res, atom_chain, title="Egg-box formation (residue centroids)"):
    import plotly.graph_objects as go
    heavy = mol.elements != "H"
    ca = mol.elements == "CA"
    nres = int(atom_res.max()) + 1
    res_chain = np.zeros(nres, int)
    res_chain[atom_res[atom_res >= 0]] = atom_chain[atom_res >= 0]
    data_frames = []
    allpts = []
    for X in frames:
        cen, _ = residue_centroids(X, atom_res, heavy)
        allpts.append(cen)
        allpts.append(X[ca]) if ca.any() else None
        data_frames.append((cen, X[ca]))
    ranges = _ranges(allpts)
    nchain = max(int(res_chain.max()), 1)

    def traces(cen, cax):
        xs, ys, zs, cs = _chain_line_arrays(cen, res_chain)
        t1 = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines+markers", name="Alginate chains (residue centroids)",
                          line=dict(color=cs, colorscale="Rainbow", cmin=0, cmax=nchain, width=5),
                          marker=dict(size=3, color=cs, colorscale="Rainbow", cmin=0, cmax=nchain))
        t2 = go.Scatter3d(x=cax[:, 0], y=cax[:, 1], z=cax[:, 2], mode="markers", name="Ca²⁺",
                          marker=dict(size=6, color=COLOR_CA, line=dict(color="black", width=1)))
        return [t1, t2]

    fig = go.Figure(data=traces(*data_frames[0]),
                    frames=[go.Frame(data=traces(c, a), name=lab) for (c, a), lab in zip(data_frames, labels)])
    return _anim_layout(fig, labels, ranges, title)


def capture_animation(X, mol, atom_res, atom_chain, mc: dict, particle_d: float, title: str):
    import plotly.graph_objects as go
    heavy = mol.elements != "H"
    ca = mol.elements == "CA"
    cen, _ = residue_centroids(X, atom_res, heavy)
    nres = len(cen)
    res_chain = np.zeros(nres, int)
    res_chain[atom_res[atom_res >= 0]] = atom_chain[atom_res >= 0]
    xs, ys, zs, cs = _chain_line_arrays(cen, res_chain)
    frames = mc["frames"]
    passed = np.asarray(mc.get("passed_mask", np.zeros(len(frames[0]), bool)))
    cols = [COLOR_GOOD if p else COLOR_BAD for p in passed]
    ranges = _ranges([cen] + list(frames))
    base = [
        go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="Alginate network", hoverinfo="skip",
                     line=dict(color="#7f7f7f", width=4)),
        go.Scatter3d(x=X[ca, 0], y=X[ca, 1], z=X[ca, 2], mode="markers", name="Ca²⁺",
                     marker=dict(size=5, color=COLOR_CA, line=dict(color="black", width=1))),
    ]

    def walker(P):
        return go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                            name=f"Particles d = {particle_d:.1f} Å (green = passed, orange = trapped)",
                            marker=dict(size=7, color=cols, line=dict(color="black", width=1)))

    labels = [f"{k}" for k in range(len(frames))]
    fig = go.Figure(data=base + [walker(frames[0])],
                    frames=[go.Frame(data=[walker(P)], traces=[2], name=lab) for P, lab in zip(frames, labels)])
    return _anim_layout(fig, labels, ranges, title, duration=120)


def trajectory_gif(frames, mol, atom_res, atom_chain, labels, size_px=640, fps=6) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    heavy = mol.elements != "H"
    ca = mol.elements == "CA"
    nres = int(atom_res.max()) + 1
    res_chain = np.zeros(nres, int)
    res_chain[atom_res[atom_res >= 0]] = atom_chain[atom_res >= 0]
    cmap = plt.get_cmap("rainbow")
    allc = [residue_centroids(X, atom_res, heavy)[0] for X in frames]
    lo = np.min([c.min(0) for c in allc], 0) - 3
    hi = np.max([c.max(0) for c in allc], 0) + 3
    imgs = []
    nch = max(res_chain.max(), 1)
    for X, cen, lab in zip(frames, allc, labels):
        fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        for c in np.unique(res_chain):
            idx = res_chain == c
            ax.plot(cen[idx, 0], cen[idx, 1], cen[idx, 2], "-o", ms=2.5, lw=1.6, color=cmap(c / nch))
        if ca.any():
            ax.scatter(X[ca, 0], X[ca, 1], X[ca, 2], s=28, c=COLOR_CA, edgecolors="black", linewidths=0.5)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], max(hi[2] - lo[2], 0.3 * (hi[0] - lo[0]))))
        ax.view_init(elev=28, azim=-55)
        ax.set_axis_off()
        ax.set_title(lab, fontsize=12)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    out = io.BytesIO()
    imgs[0].save(out, format="GIF", save_all=True, append_images=imgs[1:] + [imgs[-1]] * 3,
                 duration=int(1000 / fps), loop=0)
    return out.getvalue()
