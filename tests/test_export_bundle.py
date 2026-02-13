import json
import zipfile
from pathlib import Path

import numpy as np

from minibeam.core.export_results import export_results_bundle_zip
from minibeam.core.model import Constraint, Material, Member, Point, Project, Section
from minibeam.core.pynite_adapter import SolveOutput


def _sample_project_and_results():
    prj = Project()
    p1 = Point(name="P1", x=0.0)
    p2 = Point(name="P2", x=1000.0)
    p1.constraints["DY"] = Constraint(dof="DY", enabled=True, value=0.0)
    prj.points[p1.uid] = p1
    prj.points[p2.uid] = p2

    mat = Material(name="S355")
    sec = Section(name="Rect", p1=100.0, p2=10.0)
    sec.shape_factor_z = 1.5
    sec.shape_factor_y = 1.2
    sec.shape_factor_t = 1.1
    prj.materials[mat.uid] = mat
    prj.sections[sec.uid] = sec
    m = Member(name="M1", i_uid=p1.uid, j_uid=p2.uid, material_uid=mat.uid, section_uid=sec.uid)
    prj.members[m.uid] = m

    x = np.array([0.0, 500.0, 1000.0])
    out = SolveOutput(
        combo="COMB1",
        x_nodes=[0.0, 1000.0],
        dy_nodes=[0.0, -2.0],
        dz_nodes=[0.0, 0.0],
        reactions={"P1": {"FY": 1000.0}},
        x_diag=x,
        dy_diag=np.array([0.0, -2.0, -0.1]),
        rz_diag=np.array([0.0, 0.01, 0.0]),
        dz_diag=np.zeros_like(x),
        ry_diag=np.zeros_like(x),
        N=np.array([10.0, -20.0, 3.0]),
        V=np.array([2.0, 8.0, -1.0]),
        M=np.array([0.0, 2000.0, 0.0]),
        Vz=np.zeros_like(x),
        My=np.zeros_like(x),
        T=np.array([0.0, 10.0, 0.0]),
        sigma=np.array([1.0, 50.0, 2.0]),
        tau_torsion=np.array([0.1, 0.3, 0.2]),
        margin=np.array([0.8, -0.2, 0.1]),
        margin_elastic=np.array([0.7, -0.1, 0.2]),
        margin_plastic=np.array([0.8, -0.2, 0.1]),
    )
    return prj, out


def test_export_results_bundle_zip_contains_expected_artifacts(tmp_path: Path):
    project, results = _sample_project_and_results()
    zip_path = tmp_path / "bundle.zip"

    out = export_results_bundle_zip(project, results, zip_path, base_name="case1")

    assert out.exists()
    with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())
        assert "case1.project.json" in names
        assert "case1.report.html" in names
        assert "case1.results.csv" in names
        assert "case1.results.png" in names
        assert "case1.results.svg" in names
        assert "build_info.json" in names

        build_info = json.loads(zf.read("build_info.json").decode("utf-8"))
        assert build_info["app"] == "MiniBeam"
        assert "case1.results.png" in build_info["artifacts"]["result_plots"]

        report_html = zf.read("case1.report.html").decode("utf-8")
        assert "项目摘要" in report_html
        assert "关键截面验算" in report_html
