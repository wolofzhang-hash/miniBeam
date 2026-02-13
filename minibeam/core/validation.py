from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .constants import PLANAR_2D_NODE_SUPPORT_DOFS
from .model import Constraint, Project


@dataclass
class ValidationMessage:
    level: str  # 'ERROR' or 'WARN'
    text: str


ValidationRule = Callable[[Project], List[ValidationMessage]]


def apply_spatial_mode_constraints(prj: Project) -> None:
    """Normalize project state so 2D mode and solver assumptions stay aligned."""
    if getattr(prj, "spatial_mode", "2D") == "3D":
        return

    for p in prj.points.values():
        for dof in PLANAR_2D_NODE_SUPPORT_DOFS:
            p.constraints[dof] = Constraint(dof=dof, value=0.0, enabled=True)
            p.bushes.pop(dof, None)


def _rule_minimum_geometry(prj: Project) -> List[ValidationMessage]:
    if len(prj.points) < 2:
        return [ValidationMessage("ERROR", "需要至少 2 个点才能建立梁。")]
    return []


def _rule_member_existence(prj: Project) -> List[ValidationMessage]:
    if len(prj.members) == 0:
        return [ValidationMessage("ERROR", "当前没有梁（Members）。请开启 Auto Members 或手动重建。")]
    return []


def _rule_member_assignments(prj: Project) -> List[ValidationMessage]:
    msgs: List[ValidationMessage] = []
    for m in prj.members.values():
        if not m.material_uid or m.material_uid not in prj.materials:
            msgs.append(ValidationMessage("ERROR", f"{m.name} 未分配材料。"))
        if not m.section_uid or m.section_uid not in prj.sections:
            msgs.append(ValidationMessage("ERROR", f"{m.name} 未分配截面。"))
    return msgs


def _rule_planar_stability(prj: Project) -> List[ValidationMessage]:
    if getattr(prj, "spatial_mode", "2D") == "3D":
        return []

    msgs: List[ValidationMessage] = []
    dy_xs = []
    has_rz = False
    for p in prj.points.values():
        c = p.constraints
        b = getattr(p, "bushes", {})
        has_dy = (c.get("DY") is not None and c["DY"].enabled) or (b.get("DY") is not None and b["DY"].enabled and b["DY"].stiffness > 0)
        has_rz_here = (c.get("RZ") is not None and c["RZ"].enabled) or (b.get("RZ") is not None and b["RZ"].enabled and b["RZ"].stiffness > 0)
        if has_dy:
            dy_xs.append(float(p.x))
        if has_rz_here:
            has_rz = True

    if len(dy_xs) == 0:
        msgs.append(ValidationMessage("ERROR", "模型缺少 UY(DY) 约束/弹簧，可能整体漂移。"))
    else:
        rot_ok = has_rz
        if not rot_ok:
            dy_xs_sorted = sorted(set(round(x, 9) for x in dy_xs))
            rot_ok = len(dy_xs_sorted) >= 2
        if not rot_ok:
            msgs.append(ValidationMessage("WARN", "模型可能欠约束：建议至少两个不同位置的 UY(DY) 约束/弹簧，或在某个点锁定 RZ（约束或弹簧）。"))

    return msgs


def validate_project(prj: Project) -> List[ValidationMessage]:
    apply_spatial_mode_constraints(prj)

    rules: List[ValidationRule] = [
        _rule_minimum_geometry,
        _rule_member_existence,
        _rule_member_assignments,
        _rule_planar_stability,
    ]
    msgs: List[ValidationMessage] = []
    for rule in rules:
        rule_msgs = rule(prj)
        msgs.extend(rule_msgs)
        if rule is _rule_minimum_geometry and any(m.level == "ERROR" for m in rule_msgs):
            break
    return msgs
