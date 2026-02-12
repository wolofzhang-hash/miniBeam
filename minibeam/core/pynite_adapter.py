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
    dy_diag: np.ndarray
    rz_diag: np.ndarray
    N: np.ndarray
    V: np.ndarray
    M: np.ndarray
    T: np.ndarray  # torsion/torque about local x (N·mm)
    # stress/margin sampled on same x_diag
    sigma: np.ndarray
    tau_torsion: np.ndarray
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

    # Phase-1 is mostly 2D bending, but we also support torsion about X via RX + nodal MX.
    # To allow torsion, RX must NOT be globally locked.
    # We still lock DZ and RY for a planar (XY) model.
    any_rx_fixed = any(("RX" in p.constraints and p.constraints["RX"].enabled) for p in pts_sorted)

    for idx, p in enumerate(pts_sorted):
        dx = ("DX" in p.constraints and p.constraints["DX"].enabled)
        dy = ("DY" in p.constraints and p.constraints["DY"].enabled)
        rz = ("RZ" in p.constraints and p.constraints["RZ"].enabled)
        rx = ("RX" in p.constraints and p.constraints["RX"].enabled)

        # If user didn't fix RX anywhere, we fix RX at the first node to remove rigid-body twist.
        if not any_rx_fixed and idx == 0:
            rx = True

        model.def_support(
            p.name,
            support_DX=dx,
            support_DY=dy,
            support_DZ=True,
            support_RX=rx,
            support_RY=True,
            support_RZ=rz,
        )

        if dx and abs(p.constraints["DX"].value) > 0:
            model.def_node_disp(p.name, "DX", p.constraints["DX"].value)
        if dy and abs(p.constraints["DY"].value) > 0:
            model.def_node_disp(p.name, "DY", p.constraints["DY"].value)
        if rz and abs(p.constraints["RZ"].value) > 0:
            model.def_node_disp(p.name, "RZ", p.constraints["RZ"].value)
        if rx and ("RX" in p.constraints) and abs(p.constraints["RX"].value) > 0:
            model.def_node_disp(p.name, "RX", p.constraints["RX"].value)

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
    dy_nodes = [( (model.nodes[p.name] if hasattr(model,'nodes') else model.Nodes[p.name]).DY[combo.name]) for p in pts_sorted]

    # reactions at supports
    reactions: Dict[str, Dict[str, float]] = {}
    for p in pts_sorted:
        node = (model.nodes[p.name] if hasattr(model, 'nodes') else model.Nodes[p.name])
        reactions[p.name] = {
            "FX": node.RxnFX.get(combo.name, 0.0),
            "FY": node.RxnFY.get(combo.name, 0.0),
            "MZ": node.RxnMZ.get(combo.name, 0.0),
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
    member_curves: List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for m in mems_sorted:
        mem = (model.members[m.name] if hasattr(model, 'members') else model.Members[m.name])
        # arrays in member local coordinates, then interpolate to unified global x.
        n = max(21, int(n_samples_per_member))
        xloc, V = _member_array(mem, "shear_array", "Fy", n, combo.name)
        _, M = _member_array(mem, "moment_array", "Mz", n, combo.name)
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

        xloc = np.asarray(xloc, dtype=float)
        xloc_dy = np.asarray(xloc_dy, dtype=float)
        xloc_rz = np.asarray(xloc_rz, dtype=float)
        xloc_t = np.asarray(xloc_t, dtype=float)
        xloc_n = np.asarray(xloc_n, dtype=float)
        V = np.asarray(V, dtype=float)
        M = np.asarray(M, dtype=float)
        N = np.asarray(N, dtype=float)
        T = np.asarray(T, dtype=float)
        DY = np.asarray(DY, dtype=float)
        RZ = np.asarray(RZ, dtype=float)

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

        sec = prj.sections[m.section_uid]
        mat = prj.materials[m.material_uid]
        sigma_allow = mat.sigma_y / max(prj.safety_factor, 1e-6)
        # combined normal stress = axial + bending
        sigma_axial = np.array(N) / max(sec.A, 1e-12)
        sigma_bending = np.array(M) * sec.c_z / max(sec.Iz, 1e-12)
        sigma = sigma_axial + sigma_bending
        # torsion shear stress (simplified): tau = T*r/J, use r ~= c_z as a conservative proxy
        r_max = max(getattr(sec, "c_z", 0.0), 1e-9)
        tau_t = np.array(T) * r_max / max(sec.J, 1e-12)
        margin = sigma_allow / np.maximum(np.abs(sigma), 1e-9) - 1.0

        member_curves.append((float(min(xi, xj)), float(max(xi, xj)), xg, DY, RZ, N, V, M, T, sigma, tau_t, margin))

    dy_all = np.zeros_like(x_diag, dtype=float)
    rz_all = np.zeros_like(x_diag, dtype=float)
    N_all = np.zeros_like(x_diag, dtype=float)
    V_all = np.zeros_like(x_diag, dtype=float)
    M_all = np.zeros_like(x_diag, dtype=float)
    T_all = np.zeros_like(x_diag, dtype=float)
    sigma_all = np.zeros_like(x_diag, dtype=float)
    tau_all = np.zeros_like(x_diag, dtype=float)
    margin_all = np.zeros_like(x_diag, dtype=float)

    for idx, x in enumerate(x_diag):
        mdata = _pick_member_curve(member_curves, float(x))
        if mdata is None:
            continue
        _, _, xg, DY, RZ, N, V, M, T, sigma, tau_t, margin = mdata
        dy_all[idx] = np.interp(x, xg, DY)
        rz_all[idx] = np.interp(x, xg, RZ)
        N_all[idx] = np.interp(x, xg, N)
        V_all[idx] = np.interp(x, xg, V)
        M_all[idx] = np.interp(x, xg, M)
        T_all[idx] = np.interp(x, xg, T)
        sigma_all[idx] = np.interp(x, xg, sigma)
        tau_all[idx] = np.interp(x, xg, tau_t)
        margin_all[idx] = np.interp(x, xg, margin)

    return SolveOutput(
        combo=combo.name,
        x_nodes=x_nodes,
        dy_nodes=dy_nodes,
        reactions=reactions,
        x_diag=np.array(x_diag),
        dy_diag=np.array(dy_all),
        rz_diag=np.array(rz_all),
        N=np.array(N_all),
        V=np.array(V_all),
        M=np.array(M_all),
        T=np.array(T_all),
        sigma=np.array(sigma_all),
        tau_torsion=np.array(tau_all),
        margin=np.array(margin_all),
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
    member_curves: List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
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
