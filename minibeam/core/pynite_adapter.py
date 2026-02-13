from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import inspect
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
    dz_nodes: List[float]
    reactions: Dict[str, Dict[str, float]]  # point_name -> reaction components
    # diagrams (global x coordinate arrays)
    x_diag: np.ndarray
    dy_diag: np.ndarray
    rz_diag: np.ndarray
    dz_diag: np.ndarray
    ry_diag: np.ndarray
    N: np.ndarray
    V: np.ndarray
    M: np.ndarray
    Vz: np.ndarray
    My: np.ndarray
    T: np.ndarray  # torsion/torque about local x (N·mm)
    # stress/margin sampled on same x_diag
    sigma: np.ndarray
    tau_torsion: np.ndarray
    margin: np.ndarray
    margin_elastic: np.ndarray
    margin_plastic: np.ndarray

class PyniteSolverError(RuntimeError):
    pass


def compute_ms_from_internal_forces(
    *,
    N: np.ndarray,
    Mz: np.ndarray,
    My: np.ndarray,
    T: np.ndarray,
    area: float,
    Iz: float,
    Iy: float,
    J: float,
    c_z: float,
    c_y: float,
    sigma_allow: float,
    shape_factor_z: float = 1.0,
    shape_factor_y: float = 1.0,
    shape_factor_t: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute stress and margin arrays used by MiniBeam MS post-processing."""
    N = np.asarray(N, dtype=float)
    Mz = np.asarray(Mz, dtype=float)
    My = np.asarray(My, dtype=float)
    T = np.asarray(T, dtype=float)

    k_z = max(float(shape_factor_z or 1.0), 1.0)
    k_y = max(float(shape_factor_y or 1.0), 1.0)
    k_t = max(float(shape_factor_t or 1.0), 1.0)

    sigma_axial = N / max(area, 1e-12)
    sigma_bending_z = Mz * c_z / max(Iz, 1e-12)
    sigma_bending_y = My * c_y / max(Iy, 1e-12)

    sigma_bending_elastic = np.sqrt(sigma_bending_z**2 + sigma_bending_y**2)
    sigma_bending = np.sqrt((sigma_bending_z / k_z)**2 + (sigma_bending_y / k_y)**2)

    sigma_elastic = sigma_axial + sigma_bending_elastic
    sigma = sigma_axial + sigma_bending

    r_max = max(c_y, c_z, 1e-9)
    tau_t_elastic = T * r_max / max(J, 1e-12)
    tau_t = tau_t_elastic / k_t

    sigma_eq_elastic = np.sqrt(sigma_elastic**2 + 3.0 * tau_t_elastic**2)
    sigma_eq = np.sqrt(sigma**2 + 3.0 * tau_t**2)
    margin_elastic = sigma_allow / np.maximum(np.abs(sigma_eq_elastic), 1e-9) - 1.0
    margin = sigma_allow / np.maximum(np.abs(sigma_eq), 1e-9) - 1.0

    return {
        "sigma": sigma,
        "tau_t": tau_t,
        "margin": margin,
        "margin_elastic": margin_elastic,
    }


def _define_support_spring_if_available(model, node_name: str, dof: str, stiffness: float):
    """Best-effort compatibility wrapper for PyNite spring support API."""
    if stiffness <= 0:
        return
    fn = getattr(model, "def_support_spring", None)
    if fn is None:
        return

    attempts = [
        lambda: fn(node_name, dof, stiffness),
        lambda: fn(node_name, dof, stiffness, True),
        lambda: fn(node_name, dof, stiffness, None),
    ]
    for call in attempts:
        try:
            call()
            return
        except TypeError:
            continue

    try:
        sig = inspect.signature(fn)
        kwargs = {}
        if "node_name" in sig.parameters:
            kwargs["node_name"] = node_name
        if "direction" in sig.parameters:
            kwargs["direction"] = dof
        elif "dof" in sig.parameters:
            kwargs["dof"] = dof
        if "stiffness" in sig.parameters:
            kwargs["stiffness"] = stiffness
        if "k" in sig.parameters:
            kwargs["k"] = stiffness
        if "tension_only" in sig.parameters:
            kwargs["tension_only"] = False
        if "comp_only" in sig.parameters:
            kwargs["comp_only"] = False
        if kwargs:
            fn(**kwargs)
    except Exception:
        return

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

    is_2d_mode = getattr(prj, "spatial_mode", "2D") != "3D"

    for p in pts_sorted:
        dx = ("DX" in p.constraints and p.constraints["DX"].enabled)
        dy = ("DY" in p.constraints and p.constraints["DY"].enabled)
        rz = ("RZ" in p.constraints and p.constraints["RZ"].enabled)
        rx = ("RX" in p.constraints and p.constraints["RX"].enabled)

        bushes = getattr(p, "bushes", {})
        k_dx = float(getattr(bushes.get("DX"), "stiffness", 0.0)) if getattr(bushes.get("DX"), "enabled", False) else 0.0
        k_dy = float(getattr(bushes.get("DY"), "stiffness", 0.0)) if getattr(bushes.get("DY"), "enabled", False) else 0.0
        k_dz = float(getattr(bushes.get("DZ"), "stiffness", 0.0)) if getattr(bushes.get("DZ"), "enabled", False) else 0.0
        k_rz = float(getattr(bushes.get("RZ"), "stiffness", 0.0)) if getattr(bushes.get("RZ"), "enabled", False) else 0.0
        k_rx = float(getattr(bushes.get("RX"), "stiffness", 0.0)) if getattr(bushes.get("RX"), "enabled", False) else 0.0
        k_ry = float(getattr(bushes.get("RY"), "stiffness", 0.0)) if getattr(bushes.get("RY"), "enabled", False) else 0.0

        dz = ("DZ" in p.constraints and p.constraints["DZ"].enabled)
        ry = ("RY" in p.constraints and p.constraints["RY"].enabled)

        model.def_support(
            p.name,
            support_DX=dx,
            support_DY=dy,
            support_DZ=(True if is_2d_mode else dz),
            support_RX=rx,
            support_RY=(True if is_2d_mode else ry),
            support_RZ=rz,
        )

        _define_support_spring_if_available(model, p.name, "DX", k_dx)
        _define_support_spring_if_available(model, p.name, "DY", k_dy)
        if not is_2d_mode:
            _define_support_spring_if_available(model, p.name, "DZ", k_dz)
            _define_support_spring_if_available(model, p.name, "RY", k_ry)
        _define_support_spring_if_available(model, p.name, "RZ", k_rz)
        _define_support_spring_if_available(model, p.name, "RX", k_rx)

        if dx and abs(p.constraints["DX"].value) > 0:
            model.def_node_disp(p.name, "DX", p.constraints["DX"].value)
        if dy and abs(p.constraints["DY"].value) > 0:
            model.def_node_disp(p.name, "DY", p.constraints["DY"].value)
        if not is_2d_mode and dz and abs(p.constraints["DZ"].value) > 0:
            model.def_node_disp(p.name, "DZ", p.constraints["DZ"].value)
        if rz and abs(p.constraints["RZ"].value) > 0:
            model.def_node_disp(p.name, "RZ", p.constraints["RZ"].value)
        if rx and ("RX" in p.constraints) and abs(p.constraints["RX"].value) > 0:
            model.def_node_disp(p.name, "RX", p.constraints["RX"].value)
        if not is_2d_mode and ry and ("RY" in p.constraints) and abs(p.constraints["RY"].value) > 0:
            model.def_node_disp(p.name, "RY", p.constraints["RY"].value)

    # load combos
    combo = prj.combos[combo_name]
    model.add_load_combo(combo.name, combo.factors)

    # nodal loads
    for p in prj.points.values():
        for ld in p.nodal_loads:
            model.add_node_load(p.name, ld.direction, ld.value, case=ld.case)

    # member loads (linearly distributed in local y)
    for m in prj.members.values():
        for ld in m.udl_loads:
            model.add_member_dist_load(m.name, ld.direction, ld.w1, ld.w2, case=ld.case)

    # analyze
    model.analyze_linear(check_statics=False, check_stability=True)

    # collect nodal DY
    x_nodes = [p.x for p in pts_sorted]
    dy_nodes = [((model.nodes[p.name] if hasattr(model,'nodes') else model.Nodes[p.name]).DY[combo.name]) for p in pts_sorted]
    dz_nodes = [float(getattr((model.nodes[p.name] if hasattr(model,'nodes') else model.Nodes[p.name]), "DZ", {}).get(combo.name, 0.0)) for p in pts_sorted]

    # reactions at supports
    reactions: Dict[str, Dict[str, float]] = {}
    for p in pts_sorted:
        node = (model.nodes[p.name] if hasattr(model, 'nodes') else model.Nodes[p.name])
        reactions[p.name] = {
            "FX": node.RxnFX.get(combo.name, 0.0),
            "FY": node.RxnFY.get(combo.name, 0.0),
            "FZ": getattr(node, "RxnFZ", {}).get(combo.name, 0.0) if hasattr(node, "RxnFZ") else 0.0,
            "MZ": node.RxnMZ.get(combo.name, 0.0),
            "MY": getattr(node, "RxnMY", {}).get(combo.name, 0.0) if hasattr(node, "RxnMY") else 0.0,
            "MX": getattr(node, "RxnMX", {}).get(combo.name, 0.0) if hasattr(node, "RxnMX") else 0.0,
        }

    # diagrams: build a unified global x list (default 100 divisions + all beam points)
    beam_xs = [p.x for p in pts_sorted]
    x_min = float(min(beam_xs)) if beam_xs else 0.0
    x_max = float(max(beam_xs)) if beam_xs else 0.0
    base_diag_x = np.linspace(x_min, x_max, 101) if x_max > x_min else np.array([x_min], dtype=float)
    x_diag = np.array(_merge_unique_x([*base_diag_x.tolist(), *beam_xs]), dtype=float)

    # Allowable stress is member-dependent via assigned material.

    # (x0, x1, xg, DY, RZ, N, V, M, T, sigma, tau_torsion, margin)
    member_curves: List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for m in mems_sorted:
        mem = (model.members[m.name] if hasattr(model, 'members') else model.Members[m.name])
        # arrays in member local coordinates, then interpolate to unified global x.
        n = max(21, int(n_samples_per_member))
        xloc, V = _member_array(mem, "shear_array", "Fy", n, combo.name)
        _, M = _member_array(mem, "moment_array", "Mz", n, combo.name)
        try:
            xloc_vz, Vz = _member_array(mem, "shear_array", "Fz", n, combo.name)
        except Exception:
            xloc_vz, Vz = xloc, np.zeros_like(np.asarray(xloc, dtype=float))
        try:
            xloc_my, My = _member_array(mem, "moment_array", "My", n, combo.name)
        except Exception:
            xloc_my, My = xloc, np.zeros_like(np.asarray(xloc, dtype=float))
        # axial force along local x
        try:
            xloc_n, N = _member_array(mem, "axial_array", None, n, combo.name)
        except Exception:
            try:
                xloc_n, N = _member_array(mem, "force_array", "Fx", n, combo.name)
            except Exception:
                xloc_n = xloc
                N = np.zeros_like(np.asarray(xloc, dtype=float))
        # torsion/torque about local x (often exposed as moment_array('Mx') or torque_array)
        try:
            xloc_t, T = _member_array(mem, "moment_array", "Mx", n, combo.name)
        except Exception:
            try:
                xloc_t, T = _member_array(mem, "torque_array", None, n, combo.name)
            except Exception:
                try:
                    xloc_t, T = _member_array(mem, "torsion_array", None, n, combo.name)
                except Exception:
                    xloc_t = xloc
                    T = np.zeros_like(np.asarray(xloc, dtype=float))
        has_dy_array = True
        try:
            xloc_dy, DY = _member_array(mem, "deflection_array", "dy", n, combo.name)
        except PyniteSolverError:
            has_dy_array = False
            xloc_dy = xloc
            DY = np.zeros_like(np.asarray(xloc, dtype=float))
        has_rz_array = True
        try:
            xloc_rz, RZ = _member_array(mem, "rotation_array", "rz", n, combo.name)
        except PyniteSolverError:
            has_rz_array = False
            xloc_rz = xloc
            RZ = np.zeros_like(np.asarray(xloc, dtype=float))
        has_dz_array = True
        try:
            xloc_dz, DZ = _member_array(mem, "deflection_array", "dz", n, combo.name)
        except Exception:
            has_dz_array = False
            xloc_dz = xloc
            DZ = np.zeros_like(np.asarray(xloc, dtype=float))
        has_ry_array = True
        try:
            xloc_ry, RY = _member_array(mem, "rotation_array", "ry", n, combo.name)
        except Exception:
            has_ry_array = False
            xloc_ry = xloc
            RY = np.zeros_like(np.asarray(xloc, dtype=float))

        xloc = np.asarray(xloc, dtype=float)
        xloc_dy = np.asarray(xloc_dy, dtype=float)
        xloc_rz = np.asarray(xloc_rz, dtype=float)
        xloc_dz = np.asarray(xloc_dz, dtype=float)
        xloc_ry = np.asarray(xloc_ry, dtype=float)
        xloc_vz = np.asarray(xloc_vz, dtype=float)
        xloc_my = np.asarray(xloc_my, dtype=float)
        xloc_t = np.asarray(xloc_t, dtype=float)
        xloc_n = np.asarray(xloc_n, dtype=float)
        V = np.asarray(V, dtype=float)
        M = np.asarray(M, dtype=float)
        N = np.asarray(N, dtype=float)
        T = np.asarray(T, dtype=float)
        DY = np.asarray(DY, dtype=float)
        RZ = np.asarray(RZ, dtype=float)
        DZ = np.asarray(DZ, dtype=float)
        RY = np.asarray(RY, dtype=float)
        Vz = np.asarray(Vz, dtype=float)
        My = np.asarray(My, dtype=float)

        # map to global x
        node_i = (model.nodes[mem.i_node.name] if hasattr(model,'nodes') else model.Nodes[mem.i_node.name])
        node_j = (model.nodes[mem.j_node.name] if hasattr(model,'nodes') else model.Nodes[mem.j_node.name])
        xi = node_i.X
        xj = node_j.X
        L = xj - xi

        if not has_dy_array:
            dy_i = float(getattr(node_i, "DY", {}).get(combo.name, 0.0))
            dy_j = float(getattr(node_j, "DY", {}).get(combo.name, 0.0))
            DY = np.linspace(dy_i, dy_j, max(2, len(np.asarray(xloc_dy, dtype=float))))
        if not has_rz_array:
            rz_i = float(getattr(node_i, "RZ", {}).get(combo.name, 0.0))
            rz_j = float(getattr(node_j, "RZ", {}).get(combo.name, 0.0))
            RZ = np.linspace(rz_i, rz_j, max(2, len(np.asarray(xloc_rz, dtype=float))))
        xmax = float(np.max(np.abs(xloc))) if xloc.size else 0.0
        x_local_mm = xloc * L if xmax <= 1.0 + 1e-9 and abs(L) > 1.0 + 1e-9 else xloc
        xg = xi + x_local_mm

        xmax_t = float(np.max(np.abs(xloc_t))) if xloc_t.size else 0.0
        x_local_mm_t = xloc_t * L if xmax_t <= 1.0 + 1e-9 and abs(L) > 1.0 + 1e-9 else xloc_t
        xg_t = xi + x_local_mm_t

        xmax_n = float(np.max(np.abs(xloc_n))) if xloc_n.size else 0.0
        x_local_mm_n = xloc_n * L if xmax_n <= 1.0 + 1e-9 and abs(L) > 1.0 + 1e-9 else xloc_n
        xg_n = xi + x_local_mm_n

        xmax_dy = float(np.max(np.abs(xloc_dy))) if xloc_dy.size else 0.0
        x_local_mm_dy = xloc_dy * L if xmax_dy <= 1.0 + 1e-9 and abs(L) > 1.0 + 1e-9 else xloc_dy
        xg_dy = xi + x_local_mm_dy

        xmax_rz = float(np.max(np.abs(xloc_rz))) if xloc_rz.size else 0.0
        x_local_mm_rz = xloc_rz * L if xmax_rz <= 1.0 + 1e-9 and abs(L) > 1.0 + 1e-9 else xloc_rz
        xg_rz = xi + x_local_mm_rz

        order = np.argsort(xg)
        xg = xg[order]
        V = V[order]
        M = M[order]

        order_t = np.argsort(xg_t)
        xg_t = xg_t[order_t]
        T = T[order_t]
        if xg_t.size >= 2:
            T = np.interp(xg, xg_t, T)
        elif xg.size:
            T = np.full_like(xg, T[0] if T.size else 0.0)

        order_n = np.argsort(xg_n)
        xg_n = xg_n[order_n]
        N = N[order_n]
        if xg_n.size >= 2:
            N = np.interp(xg, xg_n, N)
        elif xg.size:
            N = np.full_like(xg, N[0] if N.size else 0.0)

        order_dy = np.argsort(xg_dy)
        xg_dy = xg_dy[order_dy]
        DY = DY[order_dy]

        order_rz = np.argsort(xg_rz)
        xg_rz = xg_rz[order_rz]
        RZ = RZ[order_rz]

        if xg_dy.size >= 2:
            DY = np.interp(xg, xg_dy, DY)
        elif xg.size:
            DY = np.full_like(xg, DY[0] if DY.size else 0.0)

        if xg_rz.size >= 2:
            RZ = np.interp(xg, xg_rz, RZ)
        elif xg.size:
            RZ = np.full_like(xg, RZ[0] if RZ.size else 0.0)

        if not has_dz_array:
            dz_i = float(getattr(node_i, "DZ", {}).get(combo.name, 0.0))
            dz_j = float(getattr(node_j, "DZ", {}).get(combo.name, 0.0))
            DZ = np.linspace(dz_i, dz_j, max(2, len(np.asarray(xloc_dz, dtype=float))))
        if not has_ry_array:
            ry_i = float(getattr(node_i, "RY", {}).get(combo.name, 0.0))
            ry_j = float(getattr(node_j, "RY", {}).get(combo.name, 0.0))
            RY = np.linspace(ry_i, ry_j, max(2, len(np.asarray(xloc_ry, dtype=float))))


        xmax_dz = float(np.max(np.abs(xloc_dz))) if xloc_dz.size else 0.0
        xg_dz = xi + (xloc_dz * L if xmax_dz <= 1.0 + 1e-9 and abs(L) > 1.0 + 1e-9 else xloc_dz)
        xmax_ry = float(np.max(np.abs(xloc_ry))) if xloc_ry.size else 0.0
        xg_ry = xi + (xloc_ry * L if xmax_ry <= 1.0 + 1e-9 and abs(L) > 1.0 + 1e-9 else xloc_ry)
        xmax_vz = float(np.max(np.abs(xloc_vz))) if xloc_vz.size else 0.0
        xg_vz = xi + (xloc_vz * L if xmax_vz <= 1.0 + 1e-9 and abs(L) > 1.0 + 1e-9 else xloc_vz)
        xmax_my = float(np.max(np.abs(xloc_my))) if xloc_my.size else 0.0
        xg_my = xi + (xloc_my * L if xmax_my <= 1.0 + 1e-9 and abs(L) > 1.0 + 1e-9 else xloc_my)

        for xarr, yarr in ((xg_dz, DZ), (xg_ry, RY), (xg_vz, Vz), (xg_my, My)):
            ord2 = np.argsort(xarr)
            xarr[:] = xarr[ord2]
            yarr[:] = yarr[ord2]

        DZ = np.interp(xg, xg_dz, DZ) if xg_dz.size >= 2 else (np.full_like(xg, DZ[0]) if xg.size and DZ.size else np.zeros_like(xg))
        RY = np.interp(xg, xg_ry, RY) if xg_ry.size >= 2 else (np.full_like(xg, RY[0]) if xg.size and RY.size else np.zeros_like(xg))
        Vz = np.interp(xg, xg_vz, Vz) if xg_vz.size >= 2 else (np.full_like(xg, Vz[0]) if xg.size and Vz.size else np.zeros_like(xg))
        My = np.interp(xg, xg_my, My) if xg_my.size >= 2 else (np.full_like(xg, My[0]) if xg.size and My.size else np.zeros_like(xg))

        sec = prj.sections[m.section_uid]
        mat = prj.materials[m.material_uid]
        sigma_allow = mat.sigma_y
        c_y = float(getattr(sec, "c_y", getattr(sec, "c_z", 0.0)) or 0.0)
        c_z = float(getattr(sec, "c_z", c_y) or c_y)

        k_z = float(getattr(sec, "shape_factor_z", getattr(sec, "shape_factor", 1.0)) or 1.0)
        k_y = float(getattr(sec, "shape_factor_y", k_z) or k_z)
        k_t = float(getattr(sec, "shape_factor_t", 1.0) or 1.0)

        ms_result = compute_ms_from_internal_forces(
            N=N,
            Mz=M,
            My=My,
            T=T,
            area=sec.A,
            Iz=sec.Iz,
            Iy=sec.Iy,
            J=sec.J,
            c_z=c_z,
            c_y=c_y,
            sigma_allow=sigma_allow,
            shape_factor_z=k_z,
            shape_factor_y=k_y,
            shape_factor_t=k_t,
        )
        sigma = ms_result["sigma"]
        tau_t = ms_result["tau_t"]
        margin = ms_result["margin"]
        margin_elastic = ms_result["margin_elastic"]

        member_curves.append((float(min(xi, xj)), float(max(xi, xj)), xg, DY, RZ, DZ, RY, N, V, M, Vz, My, T, sigma, tau_t, margin, margin_elastic))

    dy_all = np.zeros_like(x_diag, dtype=float)
    rz_all = np.zeros_like(x_diag, dtype=float)
    dz_all = np.zeros_like(x_diag, dtype=float)
    ry_all = np.zeros_like(x_diag, dtype=float)
    N_all = np.zeros_like(x_diag, dtype=float)
    V_all = np.zeros_like(x_diag, dtype=float)
    M_all = np.zeros_like(x_diag, dtype=float)
    Vz_all = np.zeros_like(x_diag, dtype=float)
    My_all = np.zeros_like(x_diag, dtype=float)
    T_all = np.zeros_like(x_diag, dtype=float)
    sigma_all = np.zeros_like(x_diag, dtype=float)
    tau_all = np.zeros_like(x_diag, dtype=float)
    margin_all = np.zeros_like(x_diag, dtype=float)
    margin_elastic_all = np.zeros_like(x_diag, dtype=float)

    for idx, x in enumerate(x_diag):
        mdata = _pick_member_curve(member_curves, float(x))
        if mdata is None:
            continue
        _, _, xg, DY, RZ, DZ, RY, N, V, M, Vz, My, T, sigma, tau_t, margin, margin_elastic = mdata
        dy_all[idx] = np.interp(x, xg, DY)
        rz_all[idx] = np.interp(x, xg, RZ)
        dz_all[idx] = np.interp(x, xg, DZ)
        ry_all[idx] = np.interp(x, xg, RY)
        N_all[idx] = np.interp(x, xg, N)
        V_all[idx] = np.interp(x, xg, V)
        M_all[idx] = np.interp(x, xg, M)
        Vz_all[idx] = np.interp(x, xg, Vz)
        My_all[idx] = np.interp(x, xg, My)
        T_all[idx] = np.interp(x, xg, T)
        sigma_all[idx] = np.interp(x, xg, sigma)
        tau_all[idx] = np.interp(x, xg, tau_t)
        margin_all[idx] = np.interp(x, xg, margin)
        margin_elastic_all[idx] = np.interp(x, xg, margin_elastic)

    return SolveOutput(
        combo=combo.name,
        x_nodes=x_nodes,
        dy_nodes=dy_nodes,
        dz_nodes=dz_nodes,
        reactions=reactions,
        x_diag=np.array(x_diag),
        dy_diag=np.array(dy_all),
        rz_diag=np.array(rz_all),
        dz_diag=np.array(dz_all),
        ry_diag=np.array(ry_all),
        N=np.array(N_all),
        V=np.array(V_all),
        M=np.array(M_all),
        Vz=np.array(Vz_all),
        My=np.array(My_all),
        T=np.array(T_all),
        sigma=np.array(sigma_all),
        tau_torsion=np.array(tau_all),
        margin=np.array(margin_all),
        margin_elastic=np.array(margin_elastic_all),
        margin_plastic=np.array(margin_all),
    )


def _merge_unique_x(values: List[float], eps: float = 1e-9) -> List[float]:
    arr = sorted(float(v) for v in values)
    if not arr:
        return []
    merged = [arr[0]]
    for v in arr[1:]:
        if abs(v - merged[-1]) > eps:
            merged.append(v)
    return merged


def _pick_member_curve(
    member_curves: List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    x: float,
    eps: float = 1e-9,
):
    for m in member_curves:
        x0, x1 = m[0], m[1]
        if x0 - eps <= x <= x1 + eps:
            return m
    return member_curves[-1] if member_curves else None


def _member_array(mem, method_name: str, direction: str | None, n: int, combo_name: str):
    method = getattr(mem, method_name, None)
    if method is None:
        raise PyniteSolverError(f"PyNite member 缺少方法: {method_name}")

    attempts = []
    if direction is not None:
        attempts.extend([
            lambda: method(direction, n, combo_name=combo_name),
            lambda: method(direction, n),
            lambda: method(direction=direction, n_points=n, combo_name=combo_name),
            lambda: method(direction=direction, n_points=n),
            lambda: method(direction, n_points=n, combo_name=combo_name),
        ])
    # Some methods (e.g., torque_array) may not take a direction.
    attempts.extend([
        lambda: method(n, combo_name=combo_name),
        lambda: method(n),
        lambda: method(n_points=n, combo_name=combo_name),
        lambda: method(n_points=n),
        lambda: method(combo_name=combo_name, n_points=n),
    ])
    last_error = None
    for f in attempts:
        try:
            return f()
        except Exception as e:
            last_error = e
    raise PyniteSolverError(f"无法读取 member {method_name}({direction}) 结果: {last_error}")
