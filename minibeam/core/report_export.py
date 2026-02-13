from __future__ import annotations

from base64 import b64encode
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon, Rectangle

from .model import Project, Section
from .pynite_adapter import SolveOutput


def build_standard_report_html(project: Project, results: SolveOutput, *, title: str = "MiniBeam 结构验算报告") -> str:
    _configure_matplotlib_fonts()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_rows = _project_summary_rows(project, results)
    load_rows = _load_boundary_rows(project)
    material_rows = _material_rows(project)
    section_rows = _section_rows(project)
    member_assign_rows = _member_assign_rows(project)
    peak_rows = _peak_rows(results)
    critical_rows = _critical_section_rows(results)
    critical_calc_html = _critical_section_detail_html(project, results)
    member_strength_html = _member_strength_table_html(project, results)

    model_fbd_img_tag = _build_model_fbd_image_tag(project, results)
    sec_img_tag = _build_section_image_tag(project)
    result_img_tag = _build_plot_image_tag(results)

    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif; margin: 18px 24px; color: #222; max-width: 980px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    h2 {{ font-size: 16px; margin: 18px 0 8px; border-left: 4px solid #2f6fab; padding-left: 8px; }}
    h3 {{ font-size: 13px; margin: 12px 0 6px; color: #334; }}
    .meta {{ color: #666; font-size: 12px; margin-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin: 6px 0 10px; }}
    th, td {{ border: 1px solid #d9d9d9; padding: 6px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    .muted {{ color: #777; font-style: italic; }}
    .plot img {{ width: 100%; max-width: 820px; height: auto; border: 1px solid #ddd; display: block; margin: 0 auto; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class=\"meta\">组合: {escape(results.combo)} | 单位: mm-N-Nmm | 生成时间: {escape(generated_at)}</div>

  <h2>1. 项目摘要</h2>
  {_table_html(['项目', '值'], summary_rows)}

  <h2>2. 荷载 / 边界条件</h2>
  {_table_html(['类别', '对象', '明细'], load_rows)}

  <h2>3. 建模截图与 FBD</h2>
  <div class=\"plot\">{model_fbd_img_tag}</div>

  <h2>4. 材料与截面信息</h2>
  {_table_html(['材料', 'E', 'G', 'nu', 'rho', 'σy'], material_rows)}
  {_table_html(['截面', 'type', '关键尺寸(mm)', 'A', 'Iy', 'Iz', 'J', 'cy', 'cz'], section_rows)}
  <h3>4.1 单元属性</h3>
  {_table_html(['杆件', 'i-j', '材料', '截面'], member_assign_rows)}
  <h3>4.2 截面截图</h3>
  <div class=\"plot\">{sec_img_tag}</div>

  <h2>5. 峰值表</h2>
  {_table_html(['指标', '峰值', '位置 x(mm)'], peak_rows)}

  <h2>6. 关键截面验算（按 MS_elastic 从小到大）</h2>
  {_table_html(['序号', 'x(mm)', 'N(N)', 'V(N)', 'M(N·mm)', 'T(N·mm)', 'σ(MPa)', 'τt(MPa)', 'MS_elastic', 'MS_plastic'], critical_rows)}
  <h3>6.1 最危险截面详细算例</h3>
  {critical_calc_html}

  <h2>7. 各杆件强度计算表</h2>
  {member_strength_html}

  <h2>8. 结果图形</h2>
  <div class=\"plot\">{result_img_tag}</div>
</body>
</html>
"""


def export_standard_report_html(project: Project, results: SolveOutput, output_path: str | Path) -> Path:
    path = Path(output_path)
    html = build_standard_report_html(project, results)
    path.write_text(html, encoding="utf-8")
    return path


def _table_html(headers: list[str], rows: list[list[str]]) -> str:
    thead = "".join(f"<th>{escape(h)}</th>" for h in headers)
    if not rows:
        tbody = f"<tr><td class='muted' colspan='{len(headers)}'>无数据</td></tr>"
    else:
        body_rows = []
        for row in rows:
            body_rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>")
        tbody = "".join(body_rows)
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"


def _project_summary_rows(project: Project, results: SolveOutput) -> list[list[str]]:
    points = project.sorted_points()
    span = (points[-1].x - points[0].x) if len(points) >= 2 else 0.0
    return [
        ["节点数", str(len(project.points))],
        ["杆件数", str(len(project.members))],
        ["跨长(mm)", f"{span:.3f}"],
        ["空间模式", project.spatial_mode],
        ["安全系数", "1.000"],
        ["工况组合", results.combo],
    ]


def _load_boundary_rows(project: Project) -> list[list[str]]:
    rows: list[list[str]] = []
    for p in project.sorted_points():
        constraint_items = [
            f"{dof}={float(c.value):.6g}"
            for dof, c in sorted(p.constraints.items())
            if c.enabled
        ]
        if constraint_items:
            rows.append(["约束", p.name or p.uid, ", ".join(constraint_items)])

        bush_items = [
            f"{dof}:k={float(b.stiffness):.6g}"
            for dof, b in sorted(p.bushes.items())
            if b.enabled
        ]
        if bush_items:
            rows.append(["弹簧", p.name or p.uid, ", ".join(bush_items)])

        for ld in p.nodal_loads:
            rows.append(["节点荷载", p.name or p.uid, f"{ld.case}: {ld.direction} = {float(ld.value):.6g}"])

    members = sorted(
        project.members.values(),
        key=lambda m: (project.points[m.i_uid].x, project.points[m.j_uid].x),
    )
    for m in members:
        for ld in m.udl_loads:
            rows.append(["杆件均布荷载", m.name or m.uid, f"{ld.case}: {ld.direction}, w1={float(ld.w1):.6g}, w2={float(ld.w2):.6g}"])
    return rows


def _material_rows(project: Project) -> list[list[str]]:
    used_material_uids = {m.material_uid for m in project.members.values() if m.material_uid}
    rows: list[list[str]] = []
    for mat in sorted(project.materials.values(), key=lambda x: x.name):
        if mat.uid not in used_material_uids:
            continue
        rows.append([mat.name, f"{mat.E:.6g}", f"{mat.G:.6g}", f"{mat.nu:.6g}", f"{mat.rho:.6g}", f"{mat.sigma_y:.6g}"])
    return rows


def _section_rows(project: Project) -> list[list[str]]:
    used_section_uids = {m.section_uid for m in project.members.values() if m.section_uid}
    rows: list[list[str]] = []
    for sec in sorted(project.sections.values(), key=lambda x: x.name):
        if sec.uid not in used_section_uids:
            continue
        rows.append([
            sec.name,
            sec.type,
            _section_key_dimensions(sec),
            f"{sec.A:.6g}",
            f"{sec.Iy:.6g}",
            f"{sec.Iz:.6g}",
            f"{sec.J:.6g}",
            f"{sec.c_y:.6g}",
            f"{sec.c_z:.6g}",
        ])
    return rows


def _section_key_dimensions(sec: Section) -> str:
    p1 = float(getattr(sec, "p1", 0.0))
    p2 = float(getattr(sec, "p2", 0.0))
    p3 = float(getattr(sec, "p3", 0.0))
    p4 = float(getattr(sec, "p4", 0.0))
    t = sec.type
    if t == "RectSolid":
        return f"b={p1:.6g}, h={p2:.6g}"
    if t == "RectHollow":
        return f"b={p1:.6g}, h={p2:.6g}, t={p3:.6g}"
    if t == "CircleSolid":
        return f"d={p1:.6g}"
    if t == "CircleHollow":
        return f"D={p1:.6g}, t={p2:.6g}"
    if t == "ISection":
        return f"h={p1:.6g}, bf={p2:.6g}, tf={p3:.6g}, tw={p4:.6g}"
    vals = [v for v in [p1, p2, p3, p4] if abs(v) > 0.0]
    if not vals:
        return "-"
    return ", ".join(f"p{i + 1}={v:.6g}" for i, v in enumerate([p1, p2, p3, p4]) if abs(v) > 0.0)


def _member_assign_rows(project: Project) -> list[list[str]]:
    rows: list[list[str]] = []
    members = sorted(
        project.members.values(),
        key=lambda m: (
            project.points.get(m.i_uid).x if project.points.get(m.i_uid) else 0.0,
            project.points.get(m.j_uid).x if project.points.get(m.j_uid) else 0.0,
        ),
    )
    for m in members:
        pi = project.points.get(m.i_uid)
        pj = project.points.get(m.j_uid)
        mat = project.materials.get(m.material_uid)
        sec = project.sections.get(m.section_uid)
        rows.append([
            m.name or m.uid,
            f"{pi.name if pi else m.i_uid} - {pj.name if pj else m.j_uid}",
            mat.name if mat else "-",
            sec.name if sec else "-",
        ])
    return rows


def _peak_rows(results: SolveOutput) -> list[list[str]]:
    x = np.asarray(results.x_diag, dtype=float)

    def peak(name: str, arr: np.ndarray) -> list[str]:
        a = np.asarray(arr, dtype=float)
        if a.size == 0 or x.size == 0:
            return [name, "-", "-"]
        idx = int(np.argmax(np.abs(a)))
        return [name, f"{float(a[idx]):.6g}", f"{float(x[idx]):.3f}"]

    return [
        peak("位移 dy", results.dy_diag),
        peak("轴力 N", results.N),
        peak("剪力 V", results.V),
        peak("弯矩 M", results.M),
        peak("扭矩 T", results.T),
        peak("应力 σ", results.sigma),
    ]


def _critical_section_rows(results: SolveOutput, top_n: int = 8) -> list[list[str]]:
    x = np.asarray(results.x_diag, dtype=float)
    ms_elastic = np.asarray(results.margin_elastic, dtype=float)
    if x.size == 0 or ms_elastic.size == 0:
        return []
    valid = np.where(np.isfinite(ms_elastic))[0]
    if valid.size == 0:
        return []
    count = min(top_n, valid.size)
    idxs = valid[np.argsort(ms_elastic[valid])[:count]]
    rows = []
    for k, idx in enumerate(idxs, start=1):
        rows.append([
            str(k),
            f"{float(x[idx]):.3f}",
            _arr_at(results.N, idx, digits=0),
            _arr_at(results.V, idx, digits=0),
            _arr_at(results.M, idx, digits=0),
            _arr_at(results.T, idx, digits=0),
            _arr_at(results.sigma, idx, digits=0),
            _arr_at(results.tau_torsion, idx, digits=0),
            _arr_at(results.margin_elastic, idx, digits=2),
            _arr_at(results.margin, idx, digits=2),
        ])
    return rows


def _arr_at(arr: np.ndarray, idx: int, digits: int | None = None) -> str:
    a = np.asarray(arr, dtype=float)
    if idx >= a.size:
        return "-"
    value = float(a[idx])
    if digits is None:
        return f"{value:.6g}"
    if digits <= 0:
        return str(int(np.rint(value)))
    return f"{value:.{digits}f}"


def _build_plot_image_tag(results: SolveOutput) -> str:
    x = np.asarray(results.x_diag, dtype=float)
    if x.size == 0:
        return "<div class='muted'>无图形数据</div>"

    specs = [
        ("N (N)", np.asarray(results.N, dtype=float)),
        ("V_y (N)", np.asarray(results.V, dtype=float)),
        ("M_z (N·mm)", np.asarray(results.M, dtype=float)),
        ("V_z (N)", np.asarray(results.Vz, dtype=float)),
        ("M_y (N·mm)", np.asarray(results.My, dtype=float)),
        ("T (N·mm)", np.asarray(results.T, dtype=float)),
        ("dy (mm)", np.asarray(results.dy_diag, dtype=float)),
        ("dz (mm)", np.asarray(results.dz_diag, dtype=float)),
        ("rz (rad)", np.asarray(results.rz_diag, dtype=float)),
        ("ry (rad)", np.asarray(results.ry_diag, dtype=float)),
        ("σ (MPa)", np.asarray(results.sigma, dtype=float)),
        ("τt (MPa)", np.asarray(results.tau_torsion, dtype=float)),
        ("MS (-)", np.asarray(results.margin, dtype=float)),
        ("MS elastic (-)", np.asarray(results.margin_elastic, dtype=float)),
        ("MS plastic (-)", np.asarray(results.margin_plastic, dtype=float)),
    ]
    specs = [(label, arr) for label, arr in specs if arr.size == x.size]
    if not specs:
        return "<div class='muted'>无图形数据</div>"

    ncols = 3
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.2, 3.0 * nrows), dpi=140, constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(axes_arr):
        if idx >= len(specs):
            ax.axis("off")
            continue
        label, arr = specs[idx]
        if arr.size == x.size:
            ax.plot(x, arr, color="#2f6fab", linewidth=1.2)
        ax.set_title(label, fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("x (mm)")

    buff = BytesIO()
    fig.savefig(buff, format="png")
    plt.close(fig)
    return f"<img alt='结果图' src='data:image/png;base64,{b64encode(buff.getvalue()).decode('ascii')}' />"


def _draw_model_view(ax, project: Project, *, plane: str) -> None:
    load_dir = "FY" if plane == "XY" else "FZ"
    load_values = [float(ld.value) for p in project.sorted_points() for ld in p.nodal_loads if ld.direction == load_dir]
    max_abs_load = max((abs(v) for v in load_values), default=0.0)

    def _scaled_arrow_len(value: float, max_abs: float, *, min_len: float = 0.10, max_len: float = 0.24) -> float:
        if max_abs <= 0.0:
            return min_len
        ratio = abs(value) / max_abs
        return min_len + (max_len - min_len) * ratio

    points = project.sorted_points()
    for m in project.members.values():
        pi = project.points.get(m.i_uid)
        pj = project.points.get(m.j_uid)
        if pi is None or pj is None:
            continue
        ax.plot([pi.x, pj.x], [0.0, 0.0], color="#1f2d3d", linewidth=2.5)
        ax.text((pi.x + pj.x) * 0.5, 0.09, m.name or m.uid, ha="center", va="bottom", fontsize=8)

    for p in points:
        ax.scatter([p.x], [0.0], color="#2f6fab", s=25, zorder=3)
        ax.text(p.x, -0.12, p.name or p.uid, ha="center", va="top", fontsize=8)

        constrained = sorted(dof for dof, c in p.constraints.items() if c.enabled and dof in (("DY", "RZ") if plane == "XY" else ("DZ", "RY")))
        if constrained:
            ax.scatter([p.x], [-0.055], color="#27ae60", marker="^", s=40, zorder=3)
            ax.text(p.x, -0.18, "/".join(constrained), color="#27ae60", ha="center", va="top", fontsize=7)

        for ld in p.nodal_loads:
            target_dir = "FY" if plane == "XY" else "FZ"
            if ld.direction != target_dir:
                continue
            sign = -1 if ld.value < 0 else 1
            y2 = _scaled_arrow_len(float(ld.value), max_abs_load) * sign
            ax.annotate("", xy=(p.x, y2), xytext=(p.x, 0.02), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
            ax.text(
                p.x,
                y2 + (0.02 * sign),
                f"{ld.direction}={_format_fbd_value(float(ld.value))}",
                color="#c0392b",
                ha="center",
                va="bottom" if sign > 0 else "top",
                fontsize=8,
            )

    ax.set_title(f"Model view ({plane} plane)", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_yticks([])
    ax.set_ylim(-0.35, 0.35)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)


def _draw_fbd_view(ax, project: Project, results: SolveOutput, *, plane: str) -> None:
    points = project.sorted_points()
    xs = [p.x for p in points]
    ax.plot([min(xs), max(xs)], [0.0, 0.0], color="#444", linewidth=2)

    load_dir = "FY" if plane == "XY" else "FZ"
    nodal_load_values = [float(ld.value) for p in points for ld in p.nodal_loads if ld.direction == load_dir]
    udl_values = []
    if plane == "XY":
        for m in project.members.values():
            for ld in m.udl_loads:
                if ld.direction == "Fy":
                    udl_values.append(0.5 * (float(ld.w1) + float(ld.w2)))
    max_abs_load = max((abs(v) for v in (nodal_load_values + udl_values)), default=0.0)

    reactions = results.reactions or {}
    reaction_values = []
    for p in points:
        rxn = reactions.get(p.name, {}) if p.name else {}
        if not rxn:
            rxn = reactions.get(p.uid, {})
        if plane == "XY":
            reaction_values.append(float(rxn.get("FY", 0.0) or 0.0))
        else:
            reaction_values.append(float(rxn.get("FZ", 0.0) or 0.0))
    max_abs_reaction = max((abs(v) for v in reaction_values), default=0.0)

    def _scaled_arrow_len(value: float, max_abs: float, *, min_len: float = 0.10, max_len: float = 0.26) -> float:
        if max_abs <= 0.0:
            return min_len
        ratio = abs(value) / max_abs
        return min_len + (max_len - min_len) * ratio

    for m in project.members.values():
        pi = project.points.get(m.i_uid)
        pj = project.points.get(m.j_uid)
        if pi is None or pj is None:
            continue
        xm = (pi.x + pj.x) * 0.5
        for ld in m.udl_loads:
            if ld.direction != "Fy" or plane != "XY":
                continue
            w_avg = 0.5 * (float(ld.w1) + float(ld.w2))
            if abs(w_avg) <= 0.0:
                continue
            sign = -1 if w_avg < 0 else 1
            y2 = _scaled_arrow_len(w_avg, max_abs_load) * sign
            ax.annotate("", xy=(xm, y2), xytext=(xm, 0.0), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))
            ax.text(
                xm,
                y2 + (0.02 * sign),
                f"w={_format_fbd_value(w_avg)}",
                color="#c0392b",
                ha="center",
                va="bottom" if sign > 0 else "top",
                fontsize=8,
            )

    for p in points:
        for ld in p.nodal_loads:
            target_dir = "FY" if plane == "XY" else "FZ"
            if ld.direction != target_dir:
                continue
            sign = -1 if ld.value < 0 else 1
            y2 = _scaled_arrow_len(float(ld.value), max_abs_load) * sign
            ax.annotate("", xy=(p.x, y2), xytext=(p.x, 0.0), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))
            ax.text(
                p.x,
                y2 + (0.02 * sign),
                f"{ld.direction}={_format_fbd_value(float(ld.value))}",
                color="#c0392b",
                ha="center",
                va="bottom" if sign > 0 else "top",
                fontsize=8,
            )

        rxn = reactions.get(p.name, {}) if p.name else {}
        if not rxn:
            rxn = reactions.get(p.uid, {})
        for key in (("FY",) if plane == "XY" else ("FZ",)):
            val = float(rxn.get(key, 0.0) or 0.0)
            if abs(val) <= 0.0:
                continue
            sign = 1 if val > 0 else -1
            y2 = -_scaled_arrow_len(val, max_abs_reaction) * sign
            ax.annotate("", xy=(p.x, y2), xytext=(p.x, 0.0), arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))
            ax.text(
                p.x,
                y2 - (0.02 * sign),
                f"R{key}={_format_fbd_value(val)}",
                color="#27ae60",
                ha="center",
                va="top" if sign > 0 else "bottom",
                fontsize=8,
            )

        constrained = sorted(dof for dof, c in p.constraints.items() if c.enabled and dof in (("DY", "RZ") if plane == "XY" else ("DZ", "RY")))
        if constrained:
            ax.scatter([p.x], [-0.035], color="#27ae60", marker="^", s=38, zorder=3)
            ax.text(p.x, -0.09, "/".join(constrained), color="#27ae60", ha="center", va="top", fontsize=7)

    ax.set_title(f"FBD ({plane}, red: loads, green: reactions/supports)", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_yticks([])
    ax.set_ylim(-0.35, 0.35)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)


def _build_model_fbd_image_tag(project: Project, results: SolveOutput) -> str:
    points = project.sorted_points()
    if not points:
        return "<div class='muted'>无模型/FBD数据</div>"

    if project.spatial_mode == "3D":
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.2), dpi=140, constrained_layout=True)
        _draw_model_view(axes[0, 0], project, plane="XY")
        _draw_model_view(axes[0, 1], project, plane="XZ")
        _draw_fbd_view(axes[1, 0], project, results, plane="XY")
        _draw_fbd_view(axes[1, 1], project, results, plane="XZ")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), dpi=140, constrained_layout=True)
        _draw_model_view(axes[0], project, plane="XY")
        _draw_fbd_view(axes[1], project, results, plane="XY")

    buff = BytesIO()
    fig.savefig(buff, format="png")
    plt.close(fig)
    return f"<img alt='模型与FBD图' src='data:image/png;base64,{b64encode(buff.getvalue()).decode('ascii')}' />"


def _format_fbd_value(value: float) -> str:
    if abs(value) > 100000:
        return f"{value:.2e}"
    return f"{value:.0f}"


def _critical_section_detail_html(project: Project, results: SolveOutput) -> str:
    x = np.asarray(results.x_diag, dtype=float)
    ms = np.asarray(results.margin_elastic, dtype=float)
    if x.size == 0 or ms.size == 0:
        return "<div class='muted'>无关键截面详细算例数据</div>"

    idx = int(np.argmin(ms))
    sec = _first_used_section(project)
    if sec is None:
        return "<div class='muted'>无关键截面详细算例数据</div>"

    n_raw = float(np.asarray(results.N, dtype=float)[idx])
    mz_raw = float(np.asarray(results.M, dtype=float)[idx])
    my_raw = float(np.asarray(results.My, dtype=float)[idx])
    t_raw = float(np.asarray(results.T, dtype=float)[idx])
    tau_t_raw = float(np.asarray(results.tau_torsion, dtype=float)[idx])

    n_val = _arr_at(results.N, idx, digits=0)
    mz_val = _arr_at(results.M, idx, digits=0)
    my_val = _arr_at(results.My, idx, digits=0)
    t_val = _arr_at(results.T, idx, digits=0)
    tau_t_val = _arr_at(results.tau_torsion, idx, digits=0)
    ms_plastic_val = _arr_at(results.margin_plastic, idx, digits=2)
    ms_elastic_val = _arr_at(results.margin_elastic, idx, digits=2)
    mat = _first_used_material(project)
    sigma_allow = float(mat.sigma_y) if mat is not None else 0.0
    area = max(float(sec.A), 1e-12)
    k_z = max(float(getattr(sec, "shape_factor_z", getattr(sec, "shape_factor", 1.0)) or 1.0), 1.0)
    k_y = max(float(getattr(sec, "shape_factor_y", k_z) or k_z), 1.0)
    k_t = max(float(getattr(sec, "shape_factor_t", 1.0) or 1.0), 1.0)

    sigma_z_raw = mz_raw * float(sec.c_z) / max(float(sec.Iz), 1e-12)
    sigma_y_raw = my_raw * float(sec.c_y) / max(float(sec.Iy), 1e-12)
    sigma_axial_raw = n_raw / area
    is_circular = str(getattr(sec, "type", "")).startswith("Circle")
    if is_circular:
        sigma_elastic_raw = sigma_axial_raw + np.sqrt(sigma_z_raw ** 2 + sigma_y_raw ** 2)
    else:
        sigma_elastic_raw = max(
            abs(sigma_axial_raw + sigma_z_raw + sigma_y_raw),
            abs(sigma_axial_raw + sigma_z_raw - sigma_y_raw),
            abs(sigma_axial_raw - sigma_z_raw + sigma_y_raw),
            abs(sigma_axial_raw - sigma_z_raw - sigma_y_raw),
        )
    r_max = max(float(sec.c_y), float(sec.c_z), 1e-9)
    tau_t_elastic_raw = t_raw * r_max / max(float(sec.J), 1e-12)
    sigma_eq_elastic_raw = np.sqrt(sigma_elastic_raw ** 2 + 3.0 * tau_t_elastic_raw ** 2)
    sigma_eq_plastic_raw = np.sqrt(sigma_elastic_raw ** 2 + 3.0 * tau_t_raw ** 2)

    sigma_z_val = str(int(np.rint(sigma_z_raw)))
    sigma_y_val = str(int(np.rint(sigma_y_raw)))
    sigma_elastic_val = str(int(np.rint(sigma_elastic_raw)))
    tau_t_elastic_val = str(int(np.rint(tau_t_elastic_raw)))
    sigma_eq_elastic_val = str(int(np.rint(sigma_eq_elastic_raw)))
    sigma_eq_plastic_val = str(int(np.rint(sigma_eq_plastic_raw)))

    rows = [
        ["最危险位置", f"x={float(x[idx]):.3f} mm"],
        ["采用截面", f"{sec.name} ({sec.type})"],
        ["内力输入", f"N={n_val}, Mz={mz_val}, My={my_val}, T={t_val}"],
        ["截面参数", f"A={sec.A:.6g}, Iz={sec.Iz:.6g}, Iy={sec.Iy:.6g}, J={sec.J:.6g}, cz={sec.c_z:.6g}, cy={sec.c_y:.6g}"],
        ["应力结果(弹性)", f"σz={sigma_z_val}, σy={sigma_y_val}, σ={sigma_elastic_val}, τt={tau_t_elastic_val}"],
        ["应力结果(塑性修正)", f"σ(按弹性)={sigma_elastic_val}, τt_plastic={tau_t_val}"],
        ["等效应力(弹性)", f"σeq_elastic = sqrt(σ² + 3τ²) = {sigma_eq_elastic_val} MPa"],
        ["等效应力(塑性)", f"σeq_plastic = sqrt(σ(弹性)² + 3τ_plastic²) = {sigma_eq_plastic_val} MPa"],
        ["许用应力", f"σallow = fy = {sigma_allow:.6g} MPa"],
        ["安全裕度(弹性)", f"MS_elastic = σallow/|σeq_elastic|-1 = {ms_elastic_val}"],
        ["安全裕度(塑性)", f"MS_plastic = σallow/|σeq_plastic|-1 = {ms_plastic_val}"],
        ["σ说明", "圆形截面按矢量合成；非圆形截面按四个角点应力代数叠加并取最大绝对值"],
        ["形状系数", f"kz={k_z:.3f}, ky={k_y:.3f}, kt={k_t:.3f}"],
    ]
    return _table_html(["步骤", "计算说明"], rows)


def _member_strength_table_html(project: Project, results: SolveOutput) -> str:
    members = sorted(
        project.members.values(),
        key=lambda m: (
            project.points.get(m.i_uid).x if project.points.get(m.i_uid) else 0.0,
            project.points.get(m.j_uid).x if project.points.get(m.j_uid) else 0.0,
            m.name or m.uid,
        ),
    )
    if not members:
        return _table_html(["项目"], [])

    headers = ["项目", *[m.name or m.uid for m in members]]
    x = np.asarray(results.x_diag, dtype=float)
    margin_elastic = np.asarray(results.margin_elastic, dtype=float)
    margin_plastic = np.asarray(results.margin_plastic, dtype=float)
    sigma = np.asarray(results.sigma, dtype=float)
    tau_t = np.asarray(results.tau_torsion, dtype=float)

    labels_and_keys = [
        ("截面属性 / x起点 (mm)", "x_start"),
        ("截面属性 / x终点 (mm)", "x_end"),
        ("截面属性 / 材料", "material"),
        ("截面属性 / 截面", "section"),
        ("许用值 / [σ] (MPa)", "sigma_allow"),
        ("载荷 / N (N)", "N"),
        ("载荷 / Vy (N)", "V"),
        ("载荷 / Mz (N·mm)", "M"),
        ("载荷 / My (N·mm)", "My"),
        ("载荷 / T (N·mm)", "T"),
        ("应力 / σ (MPa)", "sigma"),
        ("应力 / τt (MPa)", "tau_t"),
        ("安全裕度 / MS(弹性)", "margin_elastic"),
        ("安全裕度 / MS(塑性)", "margin_plastic"),
    ]

    values_by_member: list[dict[str, str]] = []
    for m in members:
        pi = project.points.get(m.i_uid)
        pj = project.points.get(m.j_uid)
        mat = project.materials.get(m.material_uid)
        sec = project.sections.get(m.section_uid)
        x_i = float(pi.x) if pi is not None else 0.0
        x_j = float(pj.x) if pj is not None else x_i
        lo, hi = min(x_i, x_j), max(x_i, x_j)

        idx = -1
        if x.size and margin_elastic.size == x.size:
            local_idxs = np.where((x >= lo - 1e-9) & (x <= hi + 1e-9))[0]
            if local_idxs.size > 0:
                idx = int(local_idxs[np.argmin(margin_elastic[local_idxs])])

        values_by_member.append({
            "x_start": f"{x_i:.3f}",
            "x_end": f"{x_j:.3f}",
            "material": mat.name if mat else "-",
            "section": sec.name if sec else "-",
            "sigma_allow": f"{float(mat.sigma_y):.6g}" if mat else "-",
            "N": _arr_at(results.N, idx, digits=0) if idx >= 0 else "-",
            "V": _arr_at(results.V, idx, digits=0) if idx >= 0 else "-",
            "M": _arr_at(results.M, idx, digits=0) if idx >= 0 else "-",
            "My": _arr_at(results.My, idx, digits=0) if idx >= 0 else "-",
            "T": _arr_at(results.T, idx, digits=0) if idx >= 0 else "-",
            "sigma": _arr_at(sigma, idx, digits=0) if 0 <= idx < sigma.size else "-",
            "tau_t": _arr_at(tau_t, idx, digits=0) if 0 <= idx < tau_t.size else "-",
            "margin_elastic": f"{float(margin_elastic[idx]):.2f}" if 0 <= idx < margin_elastic.size else "-",
            "margin_plastic": f"{float(margin_plastic[idx]):.2f}" if 0 <= idx < margin_plastic.size else "-",
        })

    rows: list[list[str]] = []
    for label, key in labels_and_keys:
        rows.append([label, *[vals[key] for vals in values_by_member]])
    return _table_html(headers, rows)


def _first_used_section(project: Project) -> Section | None:
    for m in project.members.values():
        sec = project.sections.get(m.section_uid)
        if sec is not None:
            return sec
    return None


def _first_used_material(project: Project):
    for m in project.members.values():
        mat = project.materials.get(m.material_uid)
        if mat is not None:
            return mat
    return None


def _build_section_image_tag(project: Project) -> str:
    used_section_uids = {m.section_uid for m in project.members.values() if m.section_uid}
    secs = sorted((s for s in project.sections.values() if s.uid in used_section_uids), key=lambda x: x.name)
    if not secs:
        return "<div class='muted'>无截面数据</div>"

    n = len(secs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.6, 2.6 * nrows), dpi=140, constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(axes_arr):
        if idx >= n:
            ax.axis("off")
            continue
        _draw_section_shape(ax, secs[idx])

    buff = BytesIO()
    fig.savefig(buff, format="png")
    plt.close(fig)
    return f"<img alt='截面图' src='data:image/png;base64,{b64encode(buff.getvalue()).decode('ascii')}' />"


def _draw_section_shape(ax, sec: Section) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    typ = sec.type

    if typ == "RectSolid":
        b = max(float(sec.p1), 1.0)
        h = max(float(sec.p2), 1.0)
        ax.add_patch(Rectangle((-b / 2, -h / 2), b, h, fill=False, linewidth=2, color="#1f2d3d"))
        lim = max(b, h) * 0.65
    elif typ == "RectHollow":
        b = max(float(sec.p1), 1.0)
        h = max(float(sec.p2), 1.0)
        t = max(float(sec.p3), 0.0)
        ax.add_patch(Rectangle((-b / 2, -h / 2), b, h, fill=False, linewidth=2, color="#1f2d3d"))
        if 2 * t < min(b, h):
            ax.add_patch(Rectangle((-(b - 2 * t) / 2, -(h - 2 * t) / 2), b - 2 * t, h - 2 * t, fill=False, linewidth=1.5, color="#1f2d3d"))
        lim = max(b, h) * 0.65
    elif typ == "CircleSolid":
        d = max(float(sec.p1), 1.0)
        ax.add_patch(Circle((0, 0), d / 2, fill=False, linewidth=2, color="#1f2d3d"))
        lim = d * 0.65
    elif typ == "CircleHollow":
        d = max(float(sec.p1), 1.0)
        t = max(float(sec.p2), 0.0)
        ax.add_patch(Circle((0, 0), d / 2, fill=False, linewidth=2, color="#1f2d3d"))
        if 2 * t < d:
            ax.add_patch(Circle((0, 0), d / 2 - t, fill=False, linewidth=1.5, color="#1f2d3d"))
        lim = d * 0.65
    else:
        h = max(float(sec.p1), 1.0)
        bf = max(float(sec.p2), 1.0)
        tf = max(float(sec.p3), 1.0)
        tw = max(float(sec.p4), 1.0)
        pts = [
            (-bf / 2, h / 2), (bf / 2, h / 2), (bf / 2, h / 2 - tf), (tw / 2, h / 2 - tf),
            (tw / 2, -h / 2 + tf), (bf / 2, -h / 2 + tf), (bf / 2, -h / 2), (-bf / 2, -h / 2),
            (-bf / 2, -h / 2 + tf), (-tw / 2, -h / 2 + tf), (-tw / 2, h / 2 - tf), (-bf / 2, h / 2 - tf),
        ]
        ax.add_patch(Polygon(pts, closed=True, fill=False, linewidth=2, color="#1f2d3d"))
        lim = max(h, bf) * 0.65

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"{sec.name} ({sec.type})", fontsize=9)


def _configure_matplotlib_fonts() -> None:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = []
    for name in preferred:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            available.append(name)
        except ValueError:
            continue
    if available:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = available
    plt.rcParams["axes.unicode_minus"] = False
