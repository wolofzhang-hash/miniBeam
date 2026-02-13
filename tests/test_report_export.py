import unittest
import numpy as np

from minibeam.core.model import Project, Point, Member, Material, Section, Constraint, Bush, NodalLoad
from minibeam.core.pynite_adapter import SolveOutput
from minibeam.core.report_export import _format_fbd_value, build_standard_report_html


class TestReportExport(unittest.TestCase):
    def test_format_fbd_value_uses_integer_until_threshold(self):
        self.assertEqual(_format_fbd_value(99999.9), "100000")
        self.assertEqual(_format_fbd_value(-100000), "-100000")

    def test_format_fbd_value_uses_scientific_over_threshold(self):
        self.assertEqual(_format_fbd_value(100001), "1.00e+05")
        self.assertEqual(_format_fbd_value(-250000), "-2.50e+05")

    def test_build_standard_report_html_contains_required_sections(self):
        prj = Project()
        p1 = Point(name="P1", x=0.0)
        p2 = Point(name="P2", x=1000.0)
        p1.constraints["DY"] = Constraint(dof="DY", enabled=True, value=0.0)
        p1.constraints["RZ"] = Constraint(dof="RZ", enabled=True, value=0.0)
        p1.bushes["DX"] = Bush(dof="DX", enabled=True, stiffness=1200.0)
        prj.points[p1.uid] = p1
        prj.points[p2.uid] = p2
        p2.nodal_loads.append(NodalLoad(direction="FY", value=-500.0, case="LC1"))

        mat = Material(name="S355")
        mat_unused = Material(name="UnusedMat")
        sec = Section(name="Rect", p1=100.0, p2=10.0)
        sec_unused = Section(name="UnusedSec")
        prj.materials[mat.uid] = mat
        prj.materials[mat_unused.uid] = mat_unused
        prj.sections[sec.uid] = sec
        prj.sections[sec_unused.uid] = sec_unused
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

        html = build_standard_report_html(prj, out)

        self.assertIn("项目摘要", html)
        self.assertIn("荷载 / 边界条件", html)
        self.assertIn("建模截图与 FBD", html)
        self.assertIn("材料与截面信息", html)
        self.assertIn("单元属性", html)
        self.assertIn("M1", html)
        self.assertIn("S355", html)
        self.assertIn("Rect", html)
        self.assertIn("关键尺寸(mm)", html)
        self.assertIn("b=100, h=10", html)
        self.assertIn("生成时间", html)
        self.assertIn("峰值表", html)
        self.assertIn("关键截面验算", html)
        self.assertIn("最危险截面详细算例", html)
        self.assertIn("各杆件强度计算表", html)
        self.assertIn("截面属性 / x起点 (mm)", html)
        self.assertIn("许用值 / [σ] (MPa)", html)
        self.assertIn("安全裕度 / MS", html)
        self.assertIn("MS_elastic", html)
        self.assertIn("MS_plastic", html)
        self.assertIn("σeq_elastic = sqrt(σ² + 3τ²)", html)
        self.assertIn("data:image/png;base64", html)
        self.assertIn("-0.2", html)
        self.assertIn("DY=0, RZ=0", html)
        self.assertIn("DX:k=1200", html)
        self.assertNotIn("UnusedMat", html)
        self.assertNotIn("UnusedSec", html)


    def test_build_standard_report_html_3d_model_fbd_contains_two_view_titles(self):
        prj = Project()
        prj.spatial_mode = "3D"
        p1 = Point(name="P1", x=0.0)
        p2 = Point(name="P2", x=1000.0)
        p1.constraints["DY"] = Constraint(dof="DY", enabled=True, value=0.0)
        p1.constraints["DZ"] = Constraint(dof="DZ", enabled=True, value=0.0)
        p1.constraints["RZ"] = Constraint(dof="RZ", enabled=True, value=0.0)
        p1.constraints["RY"] = Constraint(dof="RY", enabled=True, value=0.0)
        p2.nodal_loads.append(NodalLoad(direction="FY", value=-200.0, case="LC1"))
        p2.nodal_loads.append(NodalLoad(direction="FZ", value=-300.0, case="LC1"))
        prj.points[p1.uid] = p1
        prj.points[p2.uid] = p2

        mat = Material(name="S355")
        sec = Section(name="Rect", p1=100.0, p2=10.0)
        prj.materials[mat.uid] = mat
        prj.sections[sec.uid] = sec
        m = Member(name="M1", i_uid=p1.uid, j_uid=p2.uid, material_uid=mat.uid, section_uid=sec.uid)
        prj.members[m.uid] = m

        x = np.array([0.0, 500.0, 1000.0])
        out = SolveOutput(
            combo="COMB1",
            x_nodes=[0.0, 1000.0],
            dy_nodes=[0.0, -1.0],
            dz_nodes=[0.0, -0.5],
            reactions={"P1": {"FY": 500.0, "FZ": 600.0}},
            x_diag=x,
            dy_diag=np.array([0.0, -1.0, -0.1]),
            rz_diag=np.array([0.0, 0.01, 0.0]),
            dz_diag=np.array([0.0, -0.5, -0.1]),
            ry_diag=np.array([0.0, 0.02, 0.0]),
            N=np.array([10.0, -20.0, 3.0]),
            V=np.array([2.0, 8.0, -1.0]),
            M=np.array([0.0, 2000.0, 0.0]),
            Vz=np.array([1.0, 3.0, 0.0]),
            My=np.array([0.0, 1200.0, 0.0]),
            T=np.array([0.0, 10.0, 0.0]),
            sigma=np.array([1.0, 50.0, 2.0]),
            tau_torsion=np.array([0.1, 0.3, 0.2]),
            margin=np.array([0.8, -0.2, 0.1]),
            margin_elastic=np.array([0.7, -0.1, 0.2]),
            margin_plastic=np.array([0.8, -0.2, 0.1]),
        )

        html = build_standard_report_html(prj, out)

        self.assertIn("data:image/png;base64", html)


if __name__ == "__main__":
    unittest.main()
