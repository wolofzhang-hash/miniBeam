from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .model import Project

try:
    from Pynite import FEModel3D
except Exception:  # pragma: no cover
    FEModel3D = None

@dataclass
class SolveOutput:
    combo: str
    x_nodes: List[float]
    dy_nodes: List[float]
    reactions: Dict[str, Dict[str, float]]  # point_name -> {FY, MZ, FX}
    # diagrams (global x coordinate arrays)
    x_diag: np.ndarray
    V: np.ndarray
    M: np.ndarray
    # stress/margin sampled on same x_diag
    sigma: np.ndarray
    margin: np.ndarray

class PyniteSolverError(RuntimeError):
    pass

def solve_with_pynite(prj: Project, combo_name: str, n_samples_per_member: int = 20) -> SolveOutput:
    if FEModel3D is None:
        raise PyniteSolverError("未安装 PyNiteFEA/Pynite。请先 `pip install PyNiteFEA`。")

    model = FEModel3D()

    pts_sorted = prj.sorted_points()
    # nodes
    for p in pts_sorted:
        model.add_node(p.name, p.x, 0.0, 0.0)

    # materials/sections
    for mat in prj.materials.values():
        model.add_material(mat.name, E=mat.E, G=mat.G, nu=mat.nu, rho=mat.rho, fy=mat.sigma_y)
    for sec in prj.sections.values():
        model.add_section(sec.name, A=sec.A, Iy=sec.Iy, Iz=sec.Iz, J=sec.J)

    # members (assume already rebuilt names)
    mems_sorted = sorted(prj.members.values(), key=lambda m: (prj.points[m.i_uid].x, prj.points[m.j_uid].x))
    for m in mems_sorted:
        i = prj.points[m.i_uid].name
        j = prj.points[m.j_uid].name
        mat = prj.materials[m.material_uid].name
        sec = prj.sections[m.section_uid].name
        model.add_member(m.name, i_node=i, j_node=j, material_name=mat, section_name=sec)

    # supports and enforced displacements
    # Use def_support for boolean fixed DOF, and def_node_disp for imposed value
    for p in pts_sorted:
        dx = ("DX" in p.constraints and p.constraints["DX"].enabled)
        dy = ("DY" in p.constraints and p.constraints["DY"].enabled)
        rz = ("RZ" in p.constraints and p.constraints["RZ"].enabled)

        # lock DZ, RX, RY true to keep 2D? but we can just lock DZ,RX,RY by default for all nodes
        model.def_support(p.name, support_DX=dx, support_DY=dy, support_DZ=True, support_RX=True, support_RY=True, support_RZ=rz)

        if dx and abs(p.constraints["DX"].value) > 0:
            model.def_node_disp(p.name, "DX", p.constraints["DX"].value)
        if dy and abs(p.constraints["DY"].value) > 0:
            model.def_node_disp(p.name, "DY", p.constraints["DY"].value)
        if rz and abs(p.constraints["RZ"].value) > 0:
            model.def_node_disp(p.name, "RZ", p.constraints["RZ"].value)

    # load combos
    combo = prj.combos[combo_name]
    model.add_load_combo(combo.name, combo.factors)

    # nodal loads
    for p in prj.points.values():
        for ld in p.nodal_loads:
            model.add_node_load(p.name, ld.direction, ld.value, case=ld.case)

    # member loads (UDL only)
    for m in prj.members.values():
        for ld in m.udl_loads:
            # distributed load in local y: 'Fy' with w1=w2
            model.add_member_dist_load(m.name, ld.direction, ld.w, ld.w, case=ld.case)

    # analyze
    model.analyze_linear(check_statics=False, check_stability=True)

    # collect nodal DY
    x_nodes = [p.x for p in pts_sorted]
    dy_nodes = [( (model.nodes[p.name] if hasattr(model,'nodes') else model.Nodes[p.name]).DY[combo.name]) for p in pts_sorted]

    # reactions at supports
    reactions: Dict[str, Dict[str, float]] = {}
    for p in pts_sorted:
        node = (model.nodes[p.name] if hasattr(model, 'nodes') else model.Nodes[p.name])
        reactions[p.name] = {
            "FX": node.RxnFX.get(combo.name, 0.0),
            "FY": node.RxnFY.get(combo.name, 0.0),
            "MZ": node.RxnMZ.get(combo.name, 0.0),
        }

    # diagrams along entire beam by stitching member arrays
    xs_all: List[float] = []
    V_all: List[float] = []
    M_all: List[float] = []
    sigma_all: List[float] = []
    margin_all: List[float] = []

    # Allowable stress
    # Use active material (if set) else first material
    if prj.active_material_uid and prj.active_material_uid in prj.materials:
        sigma_y = prj.materials[prj.active_material_uid].sigma_y
    else:
        sigma_y = next(iter(prj.materials.values())).sigma_y
    sigma_allow = sigma_y / max(prj.safety_factor, 1e-6)

    for m in mems_sorted:
        mem = (model.members[m.name] if hasattr(model, 'members') else model.Members[m.name])
        # arrays in member local coordinates. Use n points including ends.
        n = max(2, int(n_samples_per_member))
        try:
            xloc, V = mem.shear_array("Fy", n, combo_name=combo.name)
            _, M = mem.moment_array("Mz", n, combo_name=combo.name)
        except TypeError:
            # older PyNite versions
            xloc, V = mem.shear_array("Fy", n)
            _, M = mem.moment_array("Mz", n)

        # map to global x
        xi = (model.nodes[mem.i_node.name] if hasattr(model,'nodes') else model.Nodes[mem.i_node.name]).X
        xj = (model.nodes[mem.j_node.name] if hasattr(model,'nodes') else model.Nodes[mem.j_node.name]).X
        L = xj - xi
        xg = xi + np.array(xloc)*L

        # avoid duplicating joints
        if xs_all:
            xg = xg[1:]
            V = V[1:]
            M = M[1:]

        sec = prj.sections[m.section_uid]
        # bending stress sigma = M*c/I (Mz uses Iz & c_z)
        sigma = np.array(M) * sec.c_z / max(sec.Iz, 1e-12)
        margin = sigma_allow / np.maximum(np.abs(sigma), 1e-9) - 1.0

        xs_all.extend(list(xg))
        V_all.extend(list(V))
        M_all.extend(list(M))
        sigma_all.extend(list(sigma))
        margin_all.extend(list(margin))

    return SolveOutput(
        combo=combo.name,
        x_nodes=x_nodes,
        dy_nodes=dy_nodes,
        reactions=reactions,
        x_diag=np.array(xs_all),
        V=np.array(V_all),
        M=np.array(M_all),
        sigma=np.array(sigma_all),
        margin=np.array(margin_all),
    )