from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class MemberResultExportError(RuntimeError):
    """Raised when aligned member result export cannot proceed."""


def export_member_aligned_csv(
    model: Any,
    member_name: str,
    csv_path: str | Path,
    n_div: int = 100,
    combo_name: Optional[str] = None,
    axial: bool = True,
    shear: bool = True,
    moment: bool = True,
    torsion: bool = True,
    disp_dirs: Sequence[str] = ("UY",),
    rot_dirs: Sequence[str] = ("RZ",),
    shear_dir: str = "Fy",
    moment_dir: str = "Mz",
    torsion_dir: str = "Mx",
    axial_dir: str = "Fx",
    tol: float = 1e-6,
    eps: float = 1e-9,
) -> List[Dict[str, float]]:
    """Export aligned member results to CSV.

    Every exported quantity is sampled by the exact same ``x_list`` (local x coordinates),
    ensuring row-wise coordinate alignment across force, moment, displacement and rotation data.
    """

    combo = combo_name or "Combo 1"
    mem = _get_item(model, ("members", "Members"), member_name)
    if mem is None:
        raise MemberResultExportError(f"未找到 member '{member_name}'。")

    i_node, j_node = _resolve_member_end_nodes(model, mem)
    i_xyz = np.array([_node_coord(i_node, "X"), _node_coord(i_node, "Y"), _node_coord(i_node, "Z")], dtype=float)
    j_xyz = np.array([_node_coord(j_node, "X"), _node_coord(j_node, "Y"), _node_coord(j_node, "Z")], dtype=float)

    vec = j_xyz - i_xyz
    L = float(np.linalg.norm(vec))
    if L <= eps:
        raise MemberResultExportError(f"Member '{member_name}' 长度过小/为 0，无法导出沿杆分布。")
    ex = vec / L

    must_x = [0.0, L]
    for node in _iter_nodes(model):
        p = np.array([_node_coord(node, "X"), _node_coord(node, "Y"), _node_coord(node, "Z")], dtype=float)
        t = float(np.dot(p - i_xyz, ex))
        if t < -tol or t > L + tol:
            continue
        p_proj = i_xyz + np.clip(t, 0.0, L) * ex
        dist = float(np.linalg.norm(p - p_proj))
        if dist <= tol:
            must_x.append(float(np.clip(t, 0.0, L)))

    base_x = np.linspace(0.0, L, int(max(1, n_div)) + 1)
    x_list = _merge_unique_x([*base_x, *must_x], eps=eps)
    if not x_list:
        x_list = [0.0, L]

    if abs(x_list[0]) > eps:
        x_list.insert(0, 0.0)
    if abs(x_list[-1] - L) > eps:
        x_list.append(L)
    x_list = _merge_unique_x(x_list, eps=eps)

    warned: set[str] = set()

    axial_sampler = _build_quantity_sampler(
        mem,
        quantity="axial",
        single_method_names=("axial", "axial_force"),
        array_method_names=("axial_array",),
        mandatory=axial,
        x_list=x_list,
        direction=axial_dir,
        combo_name=combo,
        member_len=L,
    ) if axial else None

    shear_sampler = _build_quantity_sampler(
        mem,
        quantity="shear",
        single_method_names=("shear", "shear_force"),
        array_method_names=("shear_array",),
        mandatory=shear,
        x_list=x_list,
        direction=shear_dir,
        combo_name=combo,
        member_len=L,
    ) if shear else None

    moment_sampler = _build_quantity_sampler(
        mem,
        quantity="moment",
        single_method_names=("moment", "bending_moment"),
        array_method_names=("moment_array",),
        mandatory=moment,
        x_list=x_list,
        direction=moment_dir,
        combo_name=combo,
        member_len=L,
    ) if moment else None

    torsion_sampler = _build_quantity_sampler(
        mem,
        quantity="torsion",
        single_method_names=("torque", "torsion", "moment"),
        array_method_names=("torque_array", "torsion_array", "moment_array"),
        mandatory=torsion,
        x_list=x_list,
        direction=torsion_dir,
        combo_name=combo,
        member_len=L,
    ) if torsion else None

    disp_samplers: Dict[str, Optional[Callable[[float], float]]] = {}
    for d in disp_dirs:
        disp_samplers[d] = _build_quantity_sampler(
            mem,
            quantity=f"disp_{d}",
            single_method_names=("deflection", "displacement", "disp"),
            array_method_names=("deflection_array", "displacement_array", "disp_array"),
            mandatory=False,
            x_list=x_list,
            direction=d,
            combo_name=combo,
            member_len=L,
            warn_once=warned,
        )

    rot_samplers: Dict[str, Optional[Callable[[float], float]]] = {}
    for d in rot_dirs:
        rot_samplers[d] = _build_quantity_sampler(
            mem,
            quantity=f"rot_{d}",
            single_method_names=("rotation",),
            array_method_names=("rotation_array",),
            mandatory=False,
            x_list=x_list,
            direction=d,
            combo_name=combo,
            member_len=L,
            warn_once=warned,
        )

    rows: List[Dict[str, float]] = []
    for x in x_list:
        xyz = i_xyz + ex * x
        row: Dict[str, float] = {
            "x_local": float(x),
            "X": float(xyz[0]),
            "Y": float(xyz[1]),
            "Z": float(xyz[2]),
            "N_axial": float(axial_sampler(x)) if axial_sampler is not None else np.nan,
            "V_shear": float(shear_sampler(x)) if shear_sampler is not None else np.nan,
            "M_bending": float(moment_sampler(x)) if moment_sampler is not None else np.nan,
            "T_torsion": float(torsion_sampler(x)) if torsion_sampler is not None else np.nan,
        }

        for d in disp_dirs:
            sampler = disp_samplers.get(d)
            row[d] = float(sampler(x)) if sampler is not None else np.nan

        for d in rot_dirs:
            sampler = rot_samplers.get(d)
            row[d] = float(sampler(x)) if sampler is not None else np.nan

        rows.append(row)

    fieldnames = ["x_local", "X", "Y", "Z", "N_axial", "V_shear", "M_bending", "T_torsion", *disp_dirs, *rot_dirs]
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return rows


def _build_quantity_sampler(
    member: Any,
    quantity: str,
    single_method_names: Sequence[str],
    array_method_names: Sequence[str],
    mandatory: bool,
    x_list: Sequence[float],
    direction: Optional[str],
    combo_name: str,
    member_len: float,
    warn_once: Optional[set[str]] = None,
) -> Optional[Callable[[float], float]]:
    single = _pick_method(member, single_method_names)
    if single is not None:
        return lambda x: float(_call_single_point(single, x=float(x), combo_name=combo_name, direction=direction))

    array_m = _pick_method(member, array_method_names)
    if array_m is not None:
        interp = _build_interpolator_from_array(
            array_m=array_m,
            x_list=x_list,
            direction=direction,
            combo_name=combo_name,
            member_len=member_len,
        )
        return lambda x: float(interp(float(x)))

    if mandatory:
        _raise_missing_interface(member, quantity)

    if warn_once is not None and quantity not in warn_once:
        warnings.warn(f"member 缺少 {quantity} 沿 x 采样接口，导出列将写入 NaN。", RuntimeWarning)
        warn_once.add(quantity)
    return None


def _build_interpolator_from_array(
    array_m: Callable[..., Any],
    x_list: Sequence[float],
    direction: Optional[str],
    combo_name: str,
    member_len: float,
) -> Callable[[float], float]:
    n_points = max(2, len(x_list))
    x_arr, y_arr = _call_array_method(array_m, n_points=n_points, combo_name=combo_name, direction=direction)
    x_arr = np.asarray(x_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)

    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.size != y_arr.size:
        raise MemberResultExportError("array 采样接口返回格式异常，无法插值到对齐坐标。")

    if x_arr.size < 2:
        raise MemberResultExportError("array 采样点过少，无法插值到统一 x_list。")

    # Some versions return normalized x in [0, 1], others return local x in [0, L].
    xmax = float(np.max(np.abs(x_arr))) if x_arr.size else 0.0
    if xmax <= 1.0 + 1e-9 and member_len > 1.0 + 1e-9:
        x_arr = x_arr * member_len

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    return lambda x: float(np.interp(x, x_arr, y_arr))


def _call_single_point(method: Callable[..., Any], x: float, combo_name: str, direction: Optional[str]) -> float:
    attempts = [
        {"x": x, "combo_name": combo_name, "direction": direction},
        {"x": x, "combo_name": combo_name},
        {"direction": direction, "x": x, "combo_name": combo_name},
        {"x": x, "direction": direction},
        {"x": x},
    ]

    if direction is not None:
        pos = [direction, x, combo_name]
        attempts.extend([
            {"_positional": pos},
            {"_positional": [direction, x]},
            {"_positional": [x, combo_name]},
            {"_positional": [direction, x], "combo_name": combo_name},
        ])
    else:
        attempts.extend([
            {"_positional": [x, combo_name]},
            {"_positional": [x]},
        ])

    for payload in attempts:
        try:
            if "_positional" in payload:
                vals = payload["_positional"]
                kwargs = {k: v for k, v in payload.items() if k != "_positional" and v is not None}
                return float(method(*vals, **kwargs))
            kwargs = {k: v for k, v in payload.items() if v is not None}
            return float(method(**kwargs))
        except TypeError:
            continue

    raise MemberResultExportError(f"无法调用单点采样方法 '{getattr(method, '__name__', str(method))}'。")


def _call_array_method(array_m: Callable[..., Any], n_points: int, combo_name: str, direction: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    attempts = [
        {"n_points": n_points, "combo_name": combo_name, "direction": direction},
        {"n_points": n_points, "combo_name": combo_name},
        {"n_points": n_points, "direction": direction},
        {"n_points": n_points},
    ]

    if direction is not None:
        attempts.extend([
            {"_positional": [direction, n_points, combo_name]},
            {"_positional": [direction, n_points]},
        ])
    attempts.extend([
        {"_positional": [n_points, combo_name]},
        {"_positional": [n_points]},
    ])

    for payload in attempts:
        try:
            if "_positional" in payload:
                vals = payload["_positional"]
                kwargs = {k: v for k, v in payload.items() if k != "_positional" and v is not None}
                out = array_m(*vals, **kwargs)
            else:
                kwargs = {k: v for k, v in payload.items() if v is not None}
                out = array_m(**kwargs)
            if isinstance(out, tuple) and len(out) == 2:
                return np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float)
        except TypeError:
            continue

    raise MemberResultExportError(f"无法调用数组采样方法 '{getattr(array_m, '__name__', str(array_m))}'。")


def _pick_method(obj: Any, names: Sequence[str]) -> Optional[Callable[..., Any]]:
    for name in names:
        if hasattr(obj, name):
            m = getattr(obj, name)
            if callable(m):
                return m
    return None


def _raise_missing_interface(member: Any, quantity: str) -> None:
    keys = ("axial", "shear", "moment", "tor", "tors", "mx", "deflect", "rotation", "disp", "mz", "vy")
    candidates = [name for name in dir(member) if any(k in name.lower() for k in keys)]
    candidates_str = ", ".join(sorted(candidates)) if candidates else "<none>"
    raise MemberResultExportError(
        f"member 缺少 {quantity} 采样接口。请检查当前 PyNite 版本 API。候选方法: {candidates_str}"
    )


def _resolve_member_end_nodes(model: Any, member: Any) -> Tuple[Any, Any]:
    i_node_attr = getattr(member, "i_node", None)
    j_node_attr = getattr(member, "j_node", None)

    i_node = _resolve_node_from_attr(model, i_node_attr)
    j_node = _resolve_node_from_attr(model, j_node_attr)

    if i_node is None or j_node is None:
        raise MemberResultExportError("无法解析 member 的 i_node/j_node。")
    return i_node, j_node


def _resolve_node_from_attr(model: Any, value: Any) -> Optional[Any]:
    if value is None:
        return None
    if _is_node_obj(value):
        return value
    if isinstance(value, str):
        return _get_item(model, ("nodes", "Nodes"), value)
    if hasattr(value, "name"):
        name = getattr(value, "name")
        node = _get_item(model, ("nodes", "Nodes"), name)
        return node if node is not None else value
    return None


def _is_node_obj(obj: Any) -> bool:
    return all(hasattr(obj, k) for k in ("X", "Y", "Z"))


def _node_coord(node: Any, axis: str) -> float:
    if not hasattr(node, axis):
        raise MemberResultExportError(f"节点对象缺少坐标属性 {axis}。")
    return float(getattr(node, axis))


def _iter_nodes(model: Any) -> Iterable[Any]:
    store = getattr(model, "nodes", None)
    if store is None:
        store = getattr(model, "Nodes", None)
    if store is None:
        return []
    if isinstance(store, dict):
        return store.values()
    if hasattr(store, "values"):
        return store.values()
    return list(store)


def _get_item(model: Any, attrs: Sequence[str], key: str) -> Any:
    for attr in attrs:
        store = getattr(model, attr, None)
        if store is None:
            continue
        if isinstance(store, dict) and key in store:
            return store[key]
        if hasattr(store, "__getitem__"):
            try:
                return store[key]
            except Exception:
                pass
    return None


def _merge_unique_x(values: Sequence[float], eps: float) -> List[float]:
    arr = sorted(float(v) for v in values)
    if not arr:
        return []
    out = [arr[0]]
    for x in arr[1:]:
        if abs(x - out[-1]) > eps:
            out.append(x)
    return out
