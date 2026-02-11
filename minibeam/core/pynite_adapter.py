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
    reactions: Dict[str, Dict[str, float]]
    x_diag: np.ndarray
    dy_diag: np.ndarray
    rz_diag: np.ndarray
    rx_diag: np.ndarray
    V: np.ndarray
    M: np.ndarray
    T: np.ndarray
    sigma: np.ndarray
    tau_v: np.ndarray
    tau_t: np.ndarray
    sigma_eq: np.ndarray
    margin: np.ndarray
    margin_plastic: np.ndarray


class PyniteSolverError(RuntimeError):
    pass


_XLOC_EPS = 1e-9
_DOMAIN_COVERAGE_RATIO = 0.9


def solve_with_pynite(prj: Project, combo_name: str, n_samples_per_member: int = 20) -> SolveOutput:
    if FEModel3D is None:
        raise PyniteSolverError("未安装 PyNiteFEA/Pynite。请先 `pip install PyNiteFEA`。")

    model = FEModel3D()
    pts_sorted = prj.sorted_points()

    has_torsion_load = any(ld.direction == "MX" and abs(ld.value) > 1e-12 for p in prj.points.values() for ld in p.nodal_loads)
    has_rx_constraint = any(("RX" in p.constraints and p.constraints["RX"].enabled) for p in pts_sorted)
    torsion_mode = has_torsion_load or has_rx_constraint

    length_scale_to_mm = _length_scale_to_mm(prj.units)

    for p in pts_sorted:
        model.add_node(p.name, p.x, 0.0, 0.0)

    for mat in prj.materials.values():
        model.add_material(mat.name, E=mat.E, G=mat.G, nu=mat.nu, rho=mat.rho, fy=mat.sigma_y)
    for sec in prj.sections.values():
        model.add_section(sec.name, A=sec.A, Iy=sec.Iy, Iz=sec.Iz, J=sec.J)

    mems_sorted = sorted(prj.members.values(), key=lambda m: (prj.points[m.i_uid].x, prj.points[m.j_uid].x))
    for m in mems_sorted:
        model.add_member(
            m.name,
            i_node=prj.points[m.i_uid].name,
            j_node=prj.points[m.j_uid].name,
            material_name=prj.materials[m.material_uid].name,
            section_name=prj.sections[m.section_uid].name,
        )

    for p in pts_sorted:
        dx = p.constraints.get("DX", None) is not None and p.constraints["DX"].enabled
        dy = p.constraints.get("DY", None) is not None and p.constraints["DY"].enabled
        rz = p.constraints.get("RZ", None) is not None and p.constraints["RZ"].enabled
        rx = p.constraints.get("RX", None) is not None and p.constraints["RX"].enabled
        model.def_support(p.name, support_DX=dx, support_DY=dy, support_DZ=True, support_RX=(rx if torsion_mode else True), support_RY=True, support_RZ=rz)

        if dx and abs(p.constraints["DX"].value) > 0:
            model.def_node_disp(p.name, "DX", p.constraints["DX"].value)
        if dy and abs(p.constraints["DY"].value) > 0:
            model.def_node_disp(p.name, "DY", p.constraints["DY"].value)
        if rz and abs(p.constraints["RZ"].value) > 0:
            model.def_node_disp(p.name, "RZ", p.constraints["RZ"].value)
        if torsion_mode and rx and abs(p.constraints["RX"].value) > 0:
            model.def_node_disp(p.name, "RX", p.constraints["RX"].value)

    combo = prj.combos[combo_name]
    model.add_load_combo(combo.name, combo.factors)

    for p in prj.points.values():
        for ld in p.nodal_loads:
            model.add_node_load(p.name, ld.direction, ld.value, case=ld.case)

    for m in prj.members.values():
        for ld in m.udl_loads:
            model.add_member_dist_load(m.name, ld.direction, ld.w, ld.w, case=ld.case)

    model.analyze_linear(check_statics=False, check_stability=True)

    x_nodes = [p.x * length_scale_to_mm for p in pts_sorted]
    dy_nodes = [((model.nodes[p.name] if hasattr(model, "nodes") else model.Nodes[p.name]).DY[combo.name]) for p in pts_sorted]

    reactions: Dict[str, Dict[str, float]] = {}
    for p in pts_sorted:
        node = (model.nodes[p.name] if hasattr(model, "nodes") else model.Nodes[p.name])
        reactions[p.name] = {
            "FX": node.RxnFX.get(combo.name, 0.0),
            "FY": node.RxnFY.get(combo.name, 0.0),
            "MZ": node.RxnMZ.get(combo.name, 0.0),
            "MX": node.RxnMX.get(combo.name, 0.0),
        }

    beam_xs = [p.x * length_scale_to_mm for p in pts_sorted]
    x_min = float(min(beam_xs)) if beam_xs else 0.0
    x_max = float(max(beam_xs)) if beam_xs else 0.0
    base_diag_x = np.linspace(x_min, x_max, 101) if x_max > x_min else np.array([x_min], dtype=float)
    x_diag = np.array(_merge_unique_x([*base_diag_x.tolist(), *beam_xs]), dtype=float)

    sigma_y = prj.materials[prj.active_material_uid].sigma_y if prj.active_material_uid in prj.materials else next(iter(prj.materials.values())).sigma_y
    sigma_allow = sigma_y / max(prj.safety_factor, 1e-6)

    member_curves: List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for m in mems_sorted:
        mem = (model.members[m.name] if hasattr(model, "members") else model.Members[m.name])
        n = max(21, int(n_samples_per_member))
        xloc, V = _member_array(mem, "shear_array", "Fy", n, combo.name)
        _, M = _member_array(mem, "moment_array", "Mz", n, combo.name)
        _, T = _member_array(mem, "torque_array", "Mx", n, combo.name, allow_fail=True)
        xloc_dy, DY = _member_array(mem, "deflection_array", "dy", n, combo.name, allow_fail=True)
        xloc_rz, RZ = _member_array(mem, "rotation_array", "rz", n, combo.name, allow_fail=True)
        xloc_rx, RX = _member_array(mem, "rotation_array", "rx", n, combo.name, allow_fail=True)

        xloc = np.asarray(xloc, dtype=float)
        V, M, T = np.asarray(V, dtype=float), np.asarray(M, dtype=float), np.asarray(T, dtype=float)
        DY, RZ, RX = np.asarray(DY, dtype=float), np.asarray(RZ, dtype=float), np.asarray(RX, dtype=float)

        node_i = (model.nodes[mem.i_node.name] if hasattr(model, "nodes") else model.Nodes[mem.i_node.name])
        node_j = (model.nodes[mem.j_node.name] if hasattr(model, "nodes") else model.Nodes[mem.j_node.name])
        xi, xj = node_i.X * length_scale_to_mm, node_j.X * length_scale_to_mm
        L = xj - xi

        xg = xi + _to_local_x(xloc, L=L, member_name=m.name, quantity="shear/moment/torque", scale_to_mm=length_scale_to_mm)
        xg_dy = xi + _to_local_x(np.asarray(xloc_dy, dtype=float), L=L, member_name=m.name, quantity="deflection", scale_to_mm=length_scale_to_mm)
        xg_rz = xi + _to_local_x(np.asarray(xloc_rz, dtype=float), L=L, member_name=m.name, quantity="rotation_rz", scale_to_mm=length_scale_to_mm)
        xg_rx = xi + _to_local_x(np.asarray(xloc_rx, dtype=float), L=L, member_name=m.name, quantity="rotation_rx", scale_to_mm=length_scale_to_mm)

        xg, V, M, T = _sort4(xg, V, M, T)
        xg_dy, DY = _sort2(xg_dy, DY)
        xg_rz, RZ = _sort2(xg_rz, RZ)
        xg_rx, RX = _sort2(xg_rx, RX)
        DY = _interp_or_const(xg, xg_dy, DY, member_name=m.name, quantity="deflection", expected_span=abs(L))
        RZ = _interp_or_const(xg, xg_rz, RZ, member_name=m.name, quantity="rotation_rz", expected_span=abs(L))
        RX = _interp_or_const(xg, xg_rx, RX, member_name=m.name, quantity="rotation_rx", expected_span=abs(L))

        sec = prj.sections[m.section_uid]
        sigma = M * sec.c_z / max(sec.Iz, 1e-12)
        tau_v = 1.5 * V / max(sec.A, 1e-12)
        tau_t = T * sec.c_t / max(sec.J, 1e-12)
        tau = tau_v + tau_t
        sigma_eq = np.sqrt(np.maximum(0.0, sigma ** 2 + 3.0 * tau ** 2))

        Ze = sec.Iz / max(sec.c_z, 1e-12)
        shape_factor = sec.Zp / max(Ze, 1e-12)
        sigma_allow_plastic = sigma_allow * max(shape_factor, 1.0)

        margin = sigma_allow / np.maximum(np.abs(sigma_eq), 1e-9) - 1.0
        margin_plastic = sigma_allow_plastic / np.maximum(np.abs(sigma_eq), 1e-9) - 1.0

        member_curves.append((float(min(xi, xj)), float(max(xi, xj)), xg, DY, RZ, RX, V, M, T, sigma, tau_v, tau_t, sigma_eq, margin, margin_plastic))

    dy_all = np.zeros_like(x_diag)
    rz_all = np.zeros_like(x_diag)
    rx_all = np.zeros_like(x_diag)
    V_all = np.zeros_like(x_diag)
    M_all = np.zeros_like(x_diag)
    T_all = np.zeros_like(x_diag)
    sigma_all = np.zeros_like(x_diag)
    tau_v_all = np.zeros_like(x_diag)
    tau_t_all = np.zeros_like(x_diag)
    sigma_eq_all = np.zeros_like(x_diag)
    margin_all = np.zeros_like(x_diag)
    margin_plastic_all = np.zeros_like(x_diag)

    for idx, x in enumerate(x_diag):
        mdata = _pick_member_curve(member_curves, float(x))
        if mdata is None:
            continue
        _, _, xg, DY, RZ, RX, V, M, T, sigma, tau_v, tau_t, sigma_eq, margin, margin_plastic = mdata
        dy_all[idx] = np.interp(x, xg, DY)
        rz_all[idx] = np.interp(x, xg, RZ)
        rx_all[idx] = np.interp(x, xg, RX)
        V_all[idx] = np.interp(x, xg, V)
        M_all[idx] = np.interp(x, xg, M)
        T_all[idx] = np.interp(x, xg, T)
        sigma_all[idx] = np.interp(x, xg, sigma)
        tau_v_all[idx] = np.interp(x, xg, tau_v)
        tau_t_all[idx] = np.interp(x, xg, tau_t)
        sigma_eq_all[idx] = np.interp(x, xg, sigma_eq)
        margin_all[idx] = np.interp(x, xg, margin)
        margin_plastic_all[idx] = np.interp(x, xg, margin_plastic)

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
        tau_v=np.array(tau_v_all),
        tau_t=np.array(tau_t_all),
        sigma_eq=np.array(sigma_eq_all),
        margin=np.array(margin_all),
        margin_plastic=np.array(margin_plastic_all),
    )


def _length_scale_to_mm(units: str) -> float:
    token = (units or "").split("-")[0].strip().lower()
    if token == "mm":
        return 1.0
    if token == "m":
        return 1000.0
    return 1.0


def _to_local_x(
    xloc: np.ndarray,
    L: float,
    member_name: str,
    quantity: str,
    scale_to_mm: float,
    eps: float = _XLOC_EPS,
) -> np.ndarray:
    xloc = np.asarray(xloc, dtype=float).reshape(-1)
    if xloc.size == 0:
        return xloc

    raw = xloc * scale_to_mm
    normalized = xloc * L

    span_target = abs(float(L))
    span_raw = _span(raw)
    span_normalized = _span(normalized)
    near_norm_range = float(np.min(xloc)) >= -eps and float(np.max(xloc)) <= 1.0 + eps

    if span_target <= eps:
        chosen = raw
    elif near_norm_range:
        chosen = normalized
        if abs(span_raw - span_target) < abs(span_normalized - span_target):
            chosen = raw
    else:
        chosen = normalized if abs(span_normalized - span_target) <= abs(span_raw - span_target) else raw

    _validate_x_domain(
        xg=chosen,
        expected_span=span_target,
        member_name=member_name,
        quantity=quantity,
        xloc=xloc,
        cand_a=raw,
        cand_b=normalized,
        eps=eps,
    )
    return chosen


def _span(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(x) - np.min(x))


def _validate_x_domain(
    xg: np.ndarray,
    expected_span: float,
    member_name: str,
    quantity: str,
    xloc: np.ndarray,
    cand_a: np.ndarray,
    cand_b: np.ndarray,
    eps: float = _XLOC_EPS,
):
    if xg.size < 2 or expected_span <= eps:
        return

    xg_min = float(np.min(xg))
    xg_max = float(np.max(xg))
    xg_span = xg_max - xg_min
    if xg_max >= (_DOMAIN_COVERAGE_RATIO * expected_span) and xg_span >= (_DOMAIN_COVERAGE_RATIO * expected_span):
        return

    raise PyniteSolverError(
        "x 坐标体系不匹配，可能导致插值被边界夹持而产生常数曲线: "
        f"member={member_name}, quantity={quantity}, L={expected_span:.6g}, "
        f"xloc[min,max]=({float(np.min(xloc)):.6g}, {float(np.max(xloc)):.6g}), "
        f"cand_a[min,max]=({float(np.min(cand_a)):.6g}, {float(np.max(cand_a)):.6g}), "
        f"cand_b[min,max]=({float(np.min(cand_b)):.6g}, {float(np.max(cand_b)):.6g}), "
        f"selected[min,max]=({xg_min:.6g}, {xg_max:.6g})"
    )


def _sort2(x: np.ndarray, y: np.ndarray):
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    if x.size == 0:
        return x, y
    o = np.argsort(x)
    return x[o], y[o]


def _sort4(x: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    n = min(x.size, a.size, b.size, c.size)
    x = x[:n]
    a = a[:n]
    b = b[:n]
    c = c[:n]
    if x.size == 0:
        return x, a, b, c
    o = np.argsort(x)
    return x[o], a[o], b[o], c[o]


def _interp_or_const(
    x_ref: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    member_name: str,
    quantity: str,
    expected_span: float,
):
    _validate_x_domain(
        xg=x,
        expected_span=expected_span,
        member_name=member_name,
        quantity=quantity,
        xloc=x,
        cand_a=x,
        cand_b=x,
    )
    if x.size >= 2:
        return np.interp(x_ref, x, y)
    return np.full_like(x_ref, y[0] if y.size else 0.0)


def _merge_unique_x(values: List[float], eps: float = 1e-9) -> List[float]:
    arr = sorted(float(v) for v in values)
    if not arr:
        return []
    merged = [arr[0]]
    for v in arr[1:]:
        if abs(v - merged[-1]) > eps:
            merged.append(v)
    return merged


def _pick_member_curve(member_curves, x: float, eps: float = 1e-9):
    for m in member_curves:
        if m[0] - eps <= x <= m[1] + eps:
            return m
    return member_curves[-1] if member_curves else None


def _member_array(mem, method_name: str, direction: str, n: int, combo_name: str, allow_fail: bool = False):
    method = getattr(mem, method_name, None)
    if method is None:
        if allow_fail:
            return np.array([0.0, 1.0]), np.zeros(2)
        raise PyniteSolverError(f"PyNite member 缺少方法: {method_name}")

    # PyNite APIs differ by version: some accept n_points, some x_array, and
    # some older signatures use positional args. Keep broad compatibility, but
    # reject suspiciously under-sampled outputs and continue trying.
    x_samples = np.linspace(0.0, 1.0, max(int(n), 2))
    attempts = [
        lambda: method(direction=direction, n_points=n, combo_name=combo_name),
        lambda: method(direction=direction, n_points=n),
        lambda: method(direction, n_points=n, combo_name=combo_name),
        lambda: method(direction=direction, x_array=x_samples, combo_name=combo_name),
        lambda: method(direction=direction, x_array=x_samples),
        lambda: method(direction, x_samples, combo_name=combo_name),
        lambda: method(direction, x_samples),
        lambda: method(direction, n, combo_name=combo_name),
        lambda: method(direction, n),
    ]
    last_error = None
    for f in attempts:
        try:
            x, y = _normalize_member_result(f())
            if _looks_undersampled(x, y, n):
                raise ValueError(f"undersampled curve ({x.size} points)")
            return x, y
        except Exception as e:
            last_error = e
    if allow_fail:
        return np.array([0.0, 1.0]), np.zeros(2)
    raise PyniteSolverError(f"无法读取 member {method_name}({direction}) 结果: {last_error}")


def _looks_undersampled(x: np.ndarray, y: np.ndarray, n: int) -> bool:
    # Member diagram arrays should usually have several points. If only 1-2
    # points are returned while we requested many, treat it as a signature
    # mismatch and continue trying other API variants.
    if n <= 3:
        return False
    return x.size != y.size or x.size < min(5, n)


def _normalize_member_result(result):
    """Normalize PyNite member array output into (x, y) 1D arrays.

    PyNite APIs vary by version and may return:
    - tuple: (x, y)
    - array shaped (2, N)
    - array shaped (N, 2)

    Notes
    -----
    We intentionally treat *tuple* ``(x, y)`` specially, but not generic lists.
    A plain list may also represent sampled pairs ``[(x1, y1), (x2, y2), ...]``;
    when there are only two points, blindly reading list[0]/list[1] as x/y would
    swap semantics and corrupt the curve.
    """
    if isinstance(result, tuple) and len(result) == 2:
        x = np.asarray(result[0], dtype=float).reshape(-1)
        y = np.asarray(result[1], dtype=float).reshape(-1)
        if x.size == y.size:
            return x, y

    arr = np.asarray(result, dtype=float)
    if arr.ndim == 2:
        # Prefer (N, 2) first to correctly handle list-of-pairs, including N=2.
        if arr.shape[1] == 2:
            return arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)
        if arr.shape[0] == 2:
            return arr[0].reshape(-1), arr[1].reshape(-1)

    raise ValueError(f"Unexpected member result shape: {arr.shape}")
