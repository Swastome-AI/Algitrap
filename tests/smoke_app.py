"""Headless smoke test: executes app.py end-to-end with mocked Streamlit/Plotly,
pressing every major button, and fails on any error shown in the UI."""
import os
import runpy
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import mock_ui  # noqa: E402

mock_ui.install()
C = mock_ui.CONTROL


class FakeUpload:
    def __init__(self, name, data):
        self.name, self._d = name, data if isinstance(data, bytes) else data.encode()
        self.size = len(self._d)

    def getvalue(self):
        return self._d


def run(label, buttons=(), values=None, uploads=None):
    C["buttons"] = set(buttons)
    C["values"] = values or {}
    C["uploads"] = uploads or C.get("uploads_keep", {})
    C["errors"] = []
    C["calls"] = []
    t = time.time()
    runpy.run_path(os.path.join(ROOT, "app.py"), run_name="__main__")
    n_plot = sum(1 for c in C["calls"] if c[0] == "plotly")
    n_dl = sum(1 for c in C["calls"] if c[0] == "download")
    n_html = sum(1 for c in C["calls"] if c[0] == "html")
    print(f"[{label}] {time.time() - t:6.1f}s  plots={n_plot} downloads={n_dl} 3Dviewers={n_html} "
          f"figs_registered={len(mock_ui.ST.session_state.get('figs', {}))} errors={len(C['errors'])}")
    for e in C["errors"]:
        print("ERROR:", e[:4000])
    assert not C["errors"], label


if __name__ == "__main__":
    from algitrap.pipeline import demo_alginate_file, demo_plastic_file
    from algitrap.uv import demo_kinetics_csv
    run("initial page")
    run("full pipeline", buttons={"btn_full"})
    run("render results")
    run("relax + MD", buttons={"btn_relax"}, values={"md_on": True})
    run("pores (with dynamics)", buttons={"btn_pores"})
    run("sweep + curve + gif", buttons={"btn_sweep", "btn_curve", "btn_gif"})
    run("figure export selectors", values={"fmt_t6_pore_size_distribution": "TIFF (300 dpi)",
                                           "fmt_t3_dyad_matrix": "SVG (vector)", "fmt_t5_energy_convergence": "PDF (vector)",
                                           "fmt_t2_size_scale_ladder": "PNG (300 dpi)"})
    run("pdf locked without name", buttons={"btn_pdf"})
    assert mock_ui.ST.session_state.get("pdf") is None
    run("reports + zips", buttons={"btn_pdf", "btn_html", "btn_zipfigs", "btn_zipall"},
        values={"user_name": "Anjali Sharma", "rep_fmts": ["png", "tiff", "svg", "pdf"]})
    ss = mock_ui.ST.session_state
    print("PDF bytes", len(ss["pdf"]), "HTML", len(ss["html"]), "plots zip", len(ss["zipfigs"]), "full zip", len(ss["zipall"]))
    # uploads
    n, b = demo_alginate_file("GGGMMGGGGGMG")
    n2, b2 = demo_plastic_file("PE", 6)
    C["uploads_keep"] = {"up_alg": FakeUpload("my_alginate.pdb", b), "up_mp": FakeUpload("pe.sdf", b2),
                         "uv_csv": FakeUpload("lab.csv", demo_kinetics_csv("PE"))}
    run("uploads")
    run("upload → full pipeline", buttons={"btn_full"})
    run("builder chain + bundle", buttons={"btn_build_chain", "btn_build_net"},
        values={"chain_source": "M/G sequence builder (built-in)", "seq_pattern": "random (Markov)",
                "net_topology": "Egg-box bundle", "seq_h": True})
    run("relax bundle", buttons={"btn_relax", "btn_pores"}, values={"md_on": False})
    run("research preset render", values={"preset": "Research (pro)"})
    run("manual particle size", values={"cap_src": "Enter manually", "cap_val": 2.0, "cap_unit": "µm"})
    run("demo buttons", buttons={"btn_demo_alg", "btn_demo_ps"})
    run("final render")
    print("ALL SMOKE TESTS PASSED")
