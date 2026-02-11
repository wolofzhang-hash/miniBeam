from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .model import Project, Section

try:
    from Pynite import FEModel3D
except Exception:  # pragma: no cover
    FEModel3D = None


@dataclass
class SolveOutput:
    combo: str
    x_nodes: List[float]
    dy_nodes: List[float]
    reactions: Dict[str, Dict[str, float]]  # point_name -> {FX, FY, FZ, MX, MY, MZ}
    # diagrams (global x coordinate arrays)
    x_diag: np.ndarray
    dy_diag: np.ndarray
    rz_diag: np.ndarray
    rx_diag: np.ndarray
    V: np.ndarray
    M: np.ndarray
    T: np.ndarray
    # stress/margin sampled on same x_diag
    sigma: np.ndarray
    tau: np.ndarray
    sigma_vm: np.ndarray
    margin: np.ndarray


class PyniteSolverError(RuntimeError):
    pass


def solve_with_pynite(prj: Project, combo_name: str, n_samples_per_member: int = 20) -> SolveOutput:
    if FEModel3D is None:
        raise PyniteSolverError("未安装 PyNiteFEA/Pynite。请先 `pip install PyNiteFEA`。")

    model = FEModel3D()
    is_3d = getattr(prj, "mode", "2D") == "3D"

    pts_sorted = prj.sorted_points()
    # nodes
    for p in pts_sorted:
        model.add_node(p.name, float(p.x), float(getattr(p, "y", 0.0)), float(getattr(p, "z", 0.0)))

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
    # 2D mode adds system planar constraints; 3D mode only uses user constraints.
    for p in pts_sorted:
        constraints = p.constraints
        dx = ("DX" in constraints and constraints["DX"].enabled)
        dy = ("DY" in constraints and constraints["DY"].enabled)
        dz = ("DZ" in constraints and constraints["DZ"].enabled)
        rx = ("RX" in constraints and constraints["RX"].enabled)
        ry = ("RY" in constraints and constraints["RY"].enabled)
        rz = ("RZ" in constraints and constraints["RZ"].enabled)

        if not is_3d:
            # Keep legacy planar-beam behavior.
            dz = True
            rx = True
            ry = True

        model.def_support(
            p.name,
            support_DX=dx,
            support_DY=dy,
            support_DZ=dz,
            support_RX=rx,
            support_RY=ry,
            support_RZ=rz,
        )

        for dof in ("DX", "DY", "DZ", "RX", "RY", "RZ"):
            c = constraints.get(dof)
            if c is not None and c.enabled and abs(float(c.value)) > 0:
                model.def_node_disp(p.name, dof, float(c.value))

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
    dy_nodes = [((model.nodes[p.name] if hasattr(model, "nodes") else model.Nodes[p.name]).DY[combo.name]) for p in pts_sorted]

    # reactions at supports
    reactions: Dict[str, Dict[str, float]] = {}
    for p in pts_sorted:
        node = (model.nodes[p.name] if hasattr(model, "nodes") else model.Nodes[p.name])
        reactions[p.name] = {
            "FX": getattr(node, "RxnFX", {}).get(combo.name, 0.0),
            "FY": getattr(node, "RxnFY", {}).get(combo.name, 0.0),
            "FZ": getattr(node, "RxnFZ", {}).get(combo.name, 0.0),
            "MX": getattr(node, "RxnMX", {}).get(combo.name, 0.0),
            "MY": getattr(node, "RxnMY", {}).get(combo.name, 0.0),
            "MZ": getattr(node, "RxnMZ", {}).get(combo.name, 0.0),
        }

    # diagrams: build a unified global x list (default 100 divisions + all beam points)
    beam_xs = [p.x for p in pts_sorted]
    x_min = float(min(beam_xs)) if beam_xs else 0.0
    x_max = float(max(beam_xs)) if beam_xs else 0.0
    base_diag_x = np.linspace(x_min, x_max, 101) if x_max > x_min else np.array([x_min], dtype=float)
    x_diag = np.array(_merge_unique_x([*base_diag_x.tolist(), *beam_xs]), dtype=float)

    # Allowable stress
    if prj.active_material_uid and prj.active_material_uid in prj.materials:
        sigma_y = prj.materials[prj.active_material_uid].sigma_y
    else:
        sigma_y = next(iter(prj.materials.values())).sigma_y
    sigma_allow = sigma_y / max(prj.safety_factor, 1e-6)

    member_curves: List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for m in mems_sorted:
        mem = (model.members[m.name] if hasattr(model, "members") else model.Members[m.name])
        n = max(21, int(n_samples_per_member))
        xloc, V = _member_array(mem, "shear_array", "Fy", n, combo.name)
        _, M = _member_array(mem, "moment_array", "Mz", n, combo.name)
        xloc_t, T = _member_torsion_array(mem, n=n, combo_name=combo.name, allow_zero_fallback=not is_3d)

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
        has_rx_array = True
        try:
            xloc_rx, RX = _member_array(mem, "rotation_array", "rx", n, combo.name)
        except PyniteSolverError:
            has_rx_array = False
            xloc_rx = xloc
            RX = np.zeros_like(np.asarray(xloc, dtype=float))

        xloc = np.asarray(xloc, dtype=float)
        xloc_dy = np.asarray(xloc_dy, dtype=float)
        xloc_rz = np.asarray(xloc_rz, dtype=float)
        xloc_t = np.asarray(xloc_t, dtype=float)
        V = np.asarray(V, dtype=float)
        M = np.asarray(M, dtype=float)
        T = np.asarray(T, dtype=float)
        DY = np.asarray(DY, dtype=float)
        RZ = np.asarray(RZ, dtype=float)
        xloc_rx = np.asarray(xloc_rx, dtype=float)
        RX = np.asarray(RX, dtype=float)

        node_i = (model.nodes[mem.i_node.name] if hasattr(model, "nodes") else model.Nodes[mem.i_node.name])
        node_j = (model.nodes[mem.j_node.name] if hasattr(model, "nodes") else model.Nodes[mem.j_node.name])
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

        xg = xi + _as_local_mm(xloc, L)
        xg_t = xi + _as_local_mm(xloc_t, L)
        xg_dy = xi + _as_local_mm(xloc_dy, L)
        xg_rz = xi + _as_local_mm(xloc_rz, L)
        xg_rx = xi + _as_local_mm(xloc_rx, L)

        order = np.argsort(xg)
        xg = xg[order]
        V = V[order]
        M = M[order]

        order_t = np.argsort(xg_t)
        xg_t = xg_t[order_t]
        T = T[order_t]

        order_dy = np.argsort(xg_dy)
        xg_dy = xg_dy[order_dy]
        DY = DY[order_dy]

        order_rz = np.argsort(xg_rz)
        xg_rz = xg_rz[order_rz]
        RZ = RZ[order_rz]

        order_rx = np.argsort(xg_rx)
        xg_rx = xg_rx[order_rx]
        RX = RX[order_rx]

        if xg_t.size >= 2:
            T = np.interp(xg, xg_t, T)
        elif xg.size:
            T = np.full_like(xg, T[0] if T.size else 0.0)

        if xg_dy.size >= 2:
            DY = np.interp(xg, xg_dy, DY)
        elif xg.size:
            DY = np.full_like(xg, DY[0] if DY.size else 0.0)

        if xg_rz.size >= 2:
            RZ = np.interp(xg, xg_rz, RZ)
        elif xg.size:
            RZ = np.full_like(xg, RZ[0] if RZ.size else 0.0)

        if xg_rx.size >= 2:
            RX = np.interp(xg, xg_rx, RX)
        elif xg.size:
            RX = np.full_like(xg, RX[0] if RX.size else 0.0)

        sec: Section = prj.sections[m.section_uid]
        sigma = np.array(M) * sec.c_z / max(sec.Iz, 1e-12)
        r_max = _section_torsion_radius(sec)
        tau = np.array(T) * r_max / max(sec.J, 1e-12)
        sigma_vm = np.sqrt(np.square(sigma) + 3.0 * np.square(tau))
        margin = sigma_allow / np.maximum(np.abs(sigma_vm), 1e-9) - 1.0

        member_curves.append((float(min(xi, xj)), float(max(xi, xj)), xg, DY, RZ, RX, V, M, T, sigma, tau, sigma_vm, margin))

    dy_all = np.zeros_like(x_diag, dtype=float)
    rz_all = np.zeros_like(x_diag, dtype=float)
    rx_all = np.zeros_like(x_diag, dtype=float)
    V_all = np.zeros_like(x_diag, dtype=float)
    M_all = np.zeros_like(x_diag, dtype=float)
    T_all = np.zeros_like(x_diag, dtype=float)
    sigma_all = np.zeros_like(x_diag, dtype=float)
    tau_all = np.zeros_like(x_diag, dtype=float)
    sigma_vm_all = np.zeros_like(x_diag, dtype=float)
    margin_all = np.zeros_like(x_diag, dtype=float)

    for idx, x in enumerate(x_diag):
        mdata = _pick_member_curve(member_curves, float(x))
        if mdata is None:
            continue
        _, _, xg, DY, RZ, RX, V, M, T, sigma, tau, sigma_vm, margin = mdata
        dy_all[idx] = np.interp(x, xg, DY)
        rz_all[idx] = np.interp(x, xg, RZ)
        rx_all[idx] = np.interp(x, xg, RX)
        V_all[idx] = np.interp(x, xg, V)
        M_all[idx] = np.interp(x, xg, M)
        T_all[idx] = np.interp(x, xg, T)
        sigma_all[idx] = np.interp(x, xg, sigma)
        tau_all[idx] = np.interp(x, xg, tau)
        sigma_vm_all[idx] = np.interp(x, xg, sigma_vm)
        margin_all[idx] = np.interp(x, xg, margin)

    return SolveOutput(
        combo=combo.name,
        x_nodes=x_nodes,
        dy_nodes=dy_nodes,
        reactions=reactions,
        x_diag=np.array(x_diag),
        dy_diag=np.array(dy_all),
        rz_diag=np.array(rz_all),
        rx_diag=np.array(rx_all),
        V=np.array(V_all),
        M=np.array(M_all),
        T=np.array(T_all),
        sigma=np.array(sigma_all),
        tau=np.array(tau_all),
        sigma_vm=np.array(sigma_vm_all),
        margin=np.array(margin_all),
    )


def _as_local_mm(x_local: np.ndarray, member_len: float) -> np.ndarray:
    xmax = float(np.max(np.abs(x_local))) if x_local.size else 0.0
    if xmax <= 1.0 + 1e-9 and abs(member_len) > 1.0 + 1e-9:
        return x_local * member_len
    return x_local


def _section_torsion_radius(sec: Section) -> float:
    """Return equivalent max radius for tau = T*r/J.

    Fallback order:
    1) existing c_t if valid;
    2) round section estimate r=sqrt(A/pi);
    3) conservative generic estimate from area.
    TODO: replace by exact y_max/z_max per detailed section geometry.
    """
    if float(getattr(sec, "c_t", 0.0)) > 1e-12:
        return float(sec.c_t)
    if sec.type.lower().startswith("circle"):
        return float(np.sqrt(max(sec.A, 1e-12) / np.pi))
    return float(np.sqrt(max(sec.A, 1e-12))) / 2.0


def _member_torsion_array(mem: Any, n: int, combo_name: str, allow_zero_fallback: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Read member torsion using API probing; never silently returns zeros in 3D mode."""
    array_specs: Sequence[Tuple[str, Optional[str]]] = (
        ("torque_array", None),
        ("torsion_array", None),
        ("moment_array", "Mx"),
    )
    for method_name, direction in array_specs:
        method = getattr(mem, method_name, None)
        if method is None:
            continue
        try:
            return _invoke_array_method(method, n=n, combo_name=combo_name, direction=direction)
        except Exception:
            continue

    point_specs: Sequence[Tuple[str, Optional[str]]] = (
        ("torque", None),
        ("torsion", None),
        ("moment", "Mx"),
    )
    x = np.linspace(0.0, 1.0, int(max(2, n)))
    for method_name, direction in point_specs:
        method = getattr(mem, method_name, None)
        if method is None:
            continue
        try:
            vals = np.asarray([_invoke_point_method(method, xi, combo_name=combo_name, direction=direction) for xi in x], dtype=float)
            return x, vals
        except Exception:
            continue

    if allow_zero_fallback:
        x = np.linspace(0.0, 1.0, int(max(2, n)))
        return x, np.zeros_like(x)

    raise PyniteSolverError(
        "无法读取 member 扭转结果接口。"
        f"候选方法: {_candidate_member_methods(mem, ('tor', 'tors', 'mx', 'moment'))}"
    )


def _invoke_array_method(method: Callable[..., Any], n: int, combo_name: str, direction: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    attempts: List[Callable[[], Any]] = [
        lambda: method(n_points=n, combo_name=combo_name, direction=direction),
        lambda: method(n_points=n, combo_name=combo_name),
        lambda: method(n_points=n, direction=direction),
        lambda: method(n_points=n),
    ]
    if direction is not None:
        attempts.extend([
            lambda: method(direction, n, combo_name=combo_name),
            lambda: method(direction, n),
        ])
    attempts.extend([
        lambda: method(n, combo_name),
        lambda: method(n),
    ])

    last_error: Optional[Exception] = None
    for fn in attempts:
        try:
            out = fn()
            if isinstance(out, tuple) and len(out) == 2:
                return np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float)
        except Exception as e:
            last_error = e
    raise PyniteSolverError(f"无法调用数组扭转接口: {last_error}")


def _invoke_point_method(method: Callable[..., Any], x: float, combo_name: str, direction: Optional[str]) -> float:
    attempts: List[Callable[[], Any]] = [
        lambda: method(x=x, combo_name=combo_name, direction=direction),
        lambda: method(x=x, combo_name=combo_name),
        lambda: method(x=x, direction=direction),
        lambda: method(x=x),
    ]
    if direction is not None:
        attempts.extend([
            lambda: method(direction=direction, x=x, combo_name=combo_name),
            lambda: method(direction, x, combo_name),
            lambda: method(direction, x),
        ])
    attempts.extend([
        lambda: method(x, combo_name),
        lambda: method(x),
    ])

    last_error: Optional[Exception] = None
    for fn in attempts:
        try:
            return float(fn())
        except Exception as e:
            last_error = e
    raise PyniteSolverError(f"无法调用单点扭转接口: {last_error}")


def _candidate_member_methods(mem: Any, keys: Sequence[str]) -> str:
    names = sorted(name for name in dir(mem) if any(k in name.lower() for k in keys))
    return ", ".join(names) if names else "<none>"


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
    member_curves: List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    x: float,
    eps: float = 1e-9,
):
    for m in member_curves:
        x0, x1 = m[0], m[1]
        if x0 - eps <= x <= x1 + eps:
            return m
    return member_curves[-1] if member_curves else None


def _member_array(mem, method_name: str, direction: str, n: int, combo_name: str):
    method = getattr(mem, method_name, None)
    if method is None:
        raise PyniteSolverError(f"PyNite member 缺少方法: {method_name}")

    attempts = [
        lambda: method(direction, n, combo_name=combo_name),
        lambda: method(direction, n),
        lambda: method(direction=direction, n_points=n, combo_name=combo_name),
        lambda: method(direction=direction, n_points=n),
        lambda: method(direction, n_points=n, combo_name=combo_name),
    ]
    last_error = None
    for f in attempts:
        try:
            return f()
        except Exception as e:
            last_error = e
    raise PyniteSolverError(f"无法读取 member {method_name}({direction}) 结果: {last_error}")
