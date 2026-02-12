"""Minimal torsion demos for miniBeam PyNite adapter.

Usage:
    python examples/torsion_demo.py

Requires PyNiteFEA + numpy installed in runtime environment.
"""
from __future__ import annotations

from minibeam.core.model import Constraint, Material, Member, NodalLoad, Point, Project, Section
from minibeam.core.pynite_adapter import solve_with_pynite


def _base_project(mode: str) -> Project:
    prj = Project(mode=mode)

    mat = Material(name="Steel", E=210000.0, G=81000.0, sigma_y=355.0)
    sec = Section(name="Round", type="CircleSolid", A=314.159, Iy=7853.98, Iz=7853.98, J=15707.96, c_z=10.0, c_t=10.0)
    prj.materials[mat.uid] = mat
    prj.sections[sec.uid] = sec
    prj.active_material_uid = mat.uid
    prj.active_section_uid = sec.uid

    p1 = Point(x=0.0, y=0.0, z=0.0)
    p2 = Point(x=1000.0, y=0.0, z=0.0)
    p1.constraints = {
        "DX": Constraint("DX", 0.0, True),
        "DY": Constraint("DY", 0.0, True),
        "DZ": Constraint("DZ", 0.0, True),
        "RX": Constraint("RX", 0.0, True),
        "RY": Constraint("RY", 0.0, True),
        "RZ": Constraint("RZ", 0.0, True),
    }
    # Free-end nodal torsion around member axis
    p2.nodal_loads = [NodalLoad(direction="MX", value=1.0e6, case="LC1")]

    prj.points[p1.uid] = p1
    prj.points[p2.uid] = p2
    m = Member(i_uid=p1.uid, j_uid=p2.uid, material_uid=mat.uid, section_uid=sec.uid)
    prj.members[m.uid] = m
    prj.rebuild_names()
    return prj


def run_demo(mode: str) -> None:
    prj = _base_project(mode)
    out = solve_with_pynite(prj, combo_name="COMB1", n_samples_per_member=31)
    print(f"\n[{mode}] T(min,max)=({out.T.min():.3f}, {out.T.max():.3f}) N·mm")
    print(f"[{mode}] tau(min,max)=({out.tau.min():.6f}, {out.tau.max():.6f}) N/mm²")
    print(f"[{mode}] reactions@fixed={out.reactions.get('P1', {})}")


if __name__ == "__main__":
    run_demo("3D")
    run_demo("2D")
