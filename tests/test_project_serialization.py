import unittest

from minibeam.core.model import (
    Bush,
    Constraint,
    LoadCombo,
    Material,
    Member,
    MemberLoadUDL,
    NodalLoad,
    Point,
    Project,
    Section,
)


class TestProjectSerialization(unittest.TestCase):
    def test_roundtrip_keeps_nested_dataclasses(self):
        prj = Project()
        p1 = Point(uid="p1", x=0.0, constraints={"DY": Constraint(dof="DY")}, bushes={"RZ": Bush(dof="RZ", stiffness=1000.0)}, nodal_loads=[NodalLoad(direction="FY", value=-120.0, case="LC1")])
        p2 = Point(uid="p2", x=1000.0)
        mat = Material(uid="mat1", name="steel")
        sec = Section(uid="sec1", name="rect", A=2000.0, Iy=1e6, Iz=1e6, J=2e5, c_y=20.0, c_z=20.0)
        mem = Member(uid="m1", i_uid="p1", j_uid="p2", material_uid="mat1", section_uid="sec1", udl_loads=[MemberLoadUDL(w1=-2.0, w2=-2.0)])

        prj.points = {"p1": p1, "p2": p2}
        prj.materials = {"mat1": mat}
        prj.sections = {"sec1": sec}
        prj.members = {"m1": mem}
        prj.combos = {"SLS": LoadCombo(name="SLS", factors={"LC1": 1.0})}
        prj.active_combo = "SLS"

        restored = Project.from_dict(prj.to_dict())

        self.assertIsInstance(restored.points["p1"].constraints["DY"], Constraint)
        self.assertIsInstance(restored.points["p1"].bushes["RZ"], Bush)
        self.assertIsInstance(restored.points["p1"].nodal_loads[0], NodalLoad)
        self.assertIsInstance(restored.members["m1"].udl_loads[0], MemberLoadUDL)

    def test_legacy_section_fields_are_upgraded(self):
        legacy = {
            "points": {
                "p1": {"uid": "p1", "name": "P1", "x": 0.0},
                "p2": {"uid": "p2", "name": "P2", "x": 1000.0},
            },
            "members": {},
            "materials": {},
            "sections": {
                "sec1": {
                    "uid": "sec1",
                    "name": "legacy",
                    "A": 1200.0,
                    "Iy": 50000.0,
                    "Iz": 80000.0,
                    "J": 6000.0,
                    "c_z": 30.0,
                    "Zp": 3000.0,
                    "shape_factor": 1.15,
                }
            },
            "combos": {"COMB1": {"name": "COMB1", "factors": {"LC1": 1.0}}},
        }

        prj = Project.from_dict(legacy)
        sec = prj.sections["sec1"]

        self.assertAlmostEqual(sec.Zp_y, 3000.0)
        self.assertAlmostEqual(sec.Zp_z, 3000.0)
        self.assertAlmostEqual(sec.shape_factor_y, 1.15)
        self.assertAlmostEqual(sec.shape_factor_z, 1.15)


    def test_normalize_member_assignments_remaps_stale_uids(self):
        prj = Project()
        prj.points = {
            "p1": Point(uid="p1", x=0.0),
            "p2": Point(uid="p2", x=1000.0),
        }
        prj.materials = {"mat_new": Material(uid="mat_new", name="steel")}
        prj.sections = {"sec_new": Section(uid="sec_new", name="pipe")}
        prj.members = {
            "m1": Member(uid="m1", i_uid="p1", j_uid="p2", material_uid="mat_old", section_uid="sec_old")
        }

        prj.normalize_member_assignments()

        self.assertEqual(prj.members["m1"].material_uid, "mat_new")
        self.assertEqual(prj.members["m1"].section_uid, "sec_new")

    def test_safety_factor_is_restored_from_file(self):
        legacy = {
            "points": {},
            "members": {},
            "materials": {},
            "sections": {},
            "safety_factor": 2.5,
        }

        prj = Project.from_dict(legacy)

        self.assertAlmostEqual(prj.safety_factor, 2.5)

    def test_legacy_section_dimensions_are_recovered_for_rect(self):
        legacy = {
            "points": {},
            "members": {},
            "materials": {},
            "sections": {
                "sec1": {
                    "uid": "sec1",
                    "name": "legacy_rect",
                    "type": "RectSolid",
                    "A": 1000.0,
                    "Iy": 8333.333333333334,
                    "Iz": 833333.3333333334,
                    "J": 31233.4,
                    "c_y": 5.0,
                    "c_z": 5.0,
                }
            },
        }

        prj = Project.from_dict(legacy)
        sec = prj.sections["sec1"]

        self.assertAlmostEqual(sec.p1, 100.0, places=3)
        self.assertAlmostEqual(sec.p2, 10.0, places=3)



if __name__ == "__main__":
    unittest.main()
