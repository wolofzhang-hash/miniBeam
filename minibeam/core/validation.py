from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .model import Project


@dataclass
class ValidationMessage:
    level: str  # 'ERROR' or 'WARN'
    text: str


def validate_project(prj: Project) -> List[ValidationMessage]:
    msgs: List[ValidationMessage] = []
    if len(prj.points) < 2:
        msgs.append(ValidationMessage("ERROR", "需要至少 2 个点才能建立梁。"))
        return msgs

    # member existence
    if len(prj.members) == 0:
        msgs.append(ValidationMessage("ERROR", "当前没有梁（Members）。请开启 Auto Members 或手动重建。"))

    # material/section assignment
    for m in prj.members.values():
        if not m.material_uid or m.material_uid not in prj.materials:
            msgs.append(ValidationMessage("ERROR", f"{m.name} 未分配材料。"))
        if not m.section_uid or m.section_uid not in prj.sections:
            msgs.append(ValidationMessage("ERROR", f"{m.name} 未分配截面。"))

    # Stability heuristic (2D planar frame): the structure must resist
    # rigid-body translation in Y and rigid-body rotation about Z.
    dy_xs = []
    has_rz = False
    for p in prj.points.values():
        c = p.constraints
        if c.get("DY") is not None and c["DY"].enabled:
            dy_xs.append(float(p.x))
        if c.get("RZ") is not None and c["RZ"].enabled:
            has_rz = True

    if len(dy_xs) == 0:
        msgs.append(ValidationMessage("ERROR", "模型缺少 UY(DY) 约束，可能整体漂移。"))
    else:
        rot_ok = has_rz
        if not rot_ok:
            dy_xs_sorted = sorted(set(round(x, 9) for x in dy_xs))
            rot_ok = len(dy_xs_sorted) >= 2
        if not rot_ok:
            msgs.append(ValidationMessage("WARN", "模型可能欠约束：建议至少两个不同位置的 UY(DY) 约束，或在某个点锁定 RZ。"))

    has_mx_load = any(ld.direction == "MX" and abs(ld.value) > 1e-12 for p in prj.points.values() for ld in p.nodal_loads)
    has_rx = any((p.constraints.get("RX") is not None and p.constraints["RX"].enabled) for p in prj.points.values())
    if has_mx_load and not has_rx:
        msgs.append(ValidationMessage("WARN", "存在扭矩(MX)荷载但没有RX约束，可能发生扭转刚体转动。"))

    return msgs
