import unittest

from minibeam.core.model import Constraint, LoadCombo, Material, Member, MemberLoadUDL, NodalLoad, Point, Project, Section
SOLVER_IMPORT_ERROR = None
try:
    from minibeam.core.pynite_adapter import FEModel3D, solve_with_pynite
except ModuleNotFoundError as exc:  # numpy / PyNite missing in CI env
    FEModel3D = None
    solve_with_pynite = None
    SOLVER_IMPORT_ERROR = exc
from minibeam.core.validation import validate_project


@unittest.skipIf(FEModel3D is None, f"solver unavailable: {SOLVER_IMPORT_ERROR or 'PyNite missing'}")
class TestSolverRegression(unittest.TestCase):
    def _base_project(self) -> Project:
        prj = Project()
        mat = Material(uid="mat", name="steel", E=206000.0, G=79300.0)
        sec = Section(uid="sec", name="rect", A=2000.0, Iy=8_000_000.0, Iz=8_000_000.0, J=200_000.0, c_y=50.0, c_z=50.0)
        prj.materials = {mat.uid: mat}
        prj.sections = {sec.uid: sec}
        prj.combos = {"COMB1": LoadCombo(name="COMB1", factors={"LC1": 1.0})}
        prj.active_combo = "COMB1"
        return prj

    def test_simply_supported_midpoint_load_reactions(self):
        prj = self._base_project()
        p1 = Point(uid="p1", x=0.0, constraints={"DX": Constraint("DX"), "DY": Constraint("DY")})
        p2 = Point(uid="p2", x=500.0, nodal_loads=[NodalLoad(direction="FY", value=-1000.0)])
        p3 = Point(uid="p3", x=1000.0, constraints={"DY": Constraint("DY")})
        m1 = Member(uid="m1", i_uid="p1", j_uid="p2", material_uid="mat", section_uid="sec")
        m2 = Member(uid="m2", i_uid="p2", j_uid="p3", material_uid="mat", section_uid="sec")
        prj.points = {p.uid: p for p in (p1, p2, p3)}
        prj.members = {m.uid: m for m in (m1, m2)}
        prj.rebuild_names()

        out = solve_with_pynite(prj, "COMB1")
        self.assertAlmostEqual(abs(out.reactions["P1"]["FY"]), 500.0, delta=5.0)
        self.assertAlmostEqual(abs(out.reactions["P3"]["FY"]), 500.0, delta=5.0)

    def test_cantilever_tip_load_reaction_and_moment(self):
        prj = self._base_project()
        p1 = Point(uid="p1", x=0.0, constraints={"DX": Constraint("DX"), "DY": Constraint("DY"), "RZ": Constraint("RZ")})
        p2 = Point(uid="p2", x=1000.0, nodal_loads=[NodalLoad(direction="FY", value=-1000.0)])
        m1 = Member(uid="m1", i_uid="p1", j_uid="p2", material_uid="mat", section_uid="sec")
        prj.points = {p1.uid: p1, p2.uid: p2}
        prj.members = {m1.uid: m1}
        prj.rebuild_names()

        out = solve_with_pynite(prj, "COMB1")
        self.assertAlmostEqual(abs(out.reactions["P1"]["FY"]), 1000.0, delta=5.0)
        self.assertAlmostEqual(abs(out.reactions["P1"]["MZ"]), 1_000_000.0, delta=15_000.0)

    def test_simply_supported_udl_reactions(self):
        prj = self._base_project()
        p1 = Point(uid="p1", x=0.0, constraints={"DX": Constraint("DX"), "DY": Constraint("DY")})
        p2 = Point(uid="p2", x=1000.0, constraints={"DY": Constraint("DY")})
        m1 = Member(uid="m1", i_uid="p1", j_uid="p2", material_uid="mat", section_uid="sec", udl_loads=[MemberLoadUDL(w1=-2.0, w2=-2.0)])
        prj.points = {p1.uid: p1, p2.uid: p2}
        prj.members = {m1.uid: m1}
        prj.rebuild_names()

        out = solve_with_pynite(prj, "COMB1")
        # w=2 N/mm over 1000 mm => total 2000 N, reactions each 1000 N
        self.assertAlmostEqual(abs(out.reactions["P1"]["FY"]), 1000.0, delta=10.0)
        self.assertAlmostEqual(abs(out.reactions["P2"]["FY"]), 1000.0, delta=10.0)


class TestValidationRegression(unittest.TestCase):
    def test_underconstrained_warns_for_single_dy_support(self):
        prj = Project()
        prj.points = {
            "p1": Point(uid="p1", x=0.0, constraints={"DY": Constraint("DY")}),
            "p2": Point(uid="p2", x=1000.0),
        }
        prj.materials = {"mat": Material(uid="mat", name="steel")}
        prj.sections = {"sec": Section(uid="sec", name="rect")}
        prj.members = {
            "m1": Member(uid="m1", i_uid="p1", j_uid="p2", material_uid="mat", section_uid="sec")
        }
        prj.rebuild_names()

        msgs = validate_project(prj)
        warn_texts = [m.text for m in msgs if m.level == "WARN"]
        self.assertTrue(any("欠约束" in t for t in warn_texts))


if __name__ == "__main__":
    unittest.main()
