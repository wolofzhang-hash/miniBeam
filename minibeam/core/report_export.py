from __future__ import annotations

from base64 import b64encode
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import Project
from .pynite_adapter import SolveOutput


def build_standard_report_html(project: Project, results: SolveOutput, *, title: str = "MiniBeam 结构验算报告") -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_rows = _project_summary_rows(project, results)
    load_rows = _load_boundary_rows(project)
    material_rows = _material_rows(project)
    section_rows = _section_rows(project)
    peak_rows = _peak_rows(results)
    critical_rows = _critical_section_rows(results)

    model_img_tag = _build_model_image_tag(project)
    fbd_img_tag = _build_fbd_image_tag(project, results)
    mat_sec_img_tag = _build_material_section_image_tag(project)
    result_img_tag = _build_plot_image_tag(results)

    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif; margin: 18px 24px; color: #222; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    h2 {{ font-size: 16px; margin: 18px 0 8px; border-left: 4px solid #2f6fab; padding-left: 8px; }}
    h3 {{ font-size: 13px; margin: 12px 0 6px; color: #334; }}
    .meta {{ color: #666; font-size: 12px; margin-bottom: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin: 6px 0 10px; }}
    th, td {{ border: 1px solid #d9d9d9; padding: 6px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    .muted {{ color: #777; font-style: italic; }}
    .plot img {{ width: 100%; border: 1px solid #ddd; }}
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
  <h3>3.1 模型截图</h3>
  <div class=\"plot\">{model_img_tag}</div>
  <h3>3.2 FBD（载荷+支反力）</h3>
  <div class=\"plot\">{fbd_img_tag}</div>

  <h2>4. 材料与截面信息</h2>
  {_table_html(['材料', 'E', 'G', 'nu', 'rho', 'σy'], material_rows)}
  {_table_html(['截面', 'type', 'A', 'Iy', 'Iz', 'J', 'cy', 'cz'], section_rows)}
  <h3>4.1 材料/截面截图</h3>
  <div class=\"plot\">{mat_sec_img_tag}</div>

  <h2>5. 峰值表</h2>
  {_table_html(['指标', '峰值', '位置 x(mm)'], peak_rows)}

  <h2>6. 关键截面验算（按 MS 从小到大）</h2>
  {_table_html(['序号', 'x(mm)', 'N(N)', 'V(N)', 'M(N·mm)', 'T(N·mm)', 'σ(N/mm²)', 'τt(N/mm²)', 'MS'], critical_rows)}

  <h2>7. 结果图形</h2>
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
        ["安全系数", f"{float(getattr(project, 'safety_factor', 1.5)):.3f}"],
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
    rows: list[list[str]] = []
    for mat in sorted(project.materials.values(), key=lambda x: x.name):
        rows.append([mat.name, f"{mat.E:.6g}", f"{mat.G:.6g}", f"{mat.nu:.6g}", f"{mat.rho:.6g}", f"{mat.sigma_y:.6g}"])
    return rows


def _section_rows(project: Project) -> list[list[str]]:
    rows: list[list[str]] = []
    for sec in sorted(project.sections.values(), key=lambda x: x.name):
        rows.append([sec.name, sec.type, f"{sec.A:.6g}", f"{sec.Iy:.6g}", f"{sec.Iz:.6g}", f"{sec.J:.6g}", f"{sec.c_y:.6g}", f"{sec.c_z:.6g}"])
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
        peak("裕度 MS", results.margin),
    ]


def _critical_section_rows(results: SolveOutput, top_n: int = 8) -> list[list[str]]:
    x = np.asarray(results.x_diag, dtype=float)
    ms = np.asarray(results.margin, dtype=float)
    if x.size == 0 or ms.size == 0:
        return []
    count = min(top_n, x.size, ms.size)
    idxs = np.argsort(ms)[:count]
    rows = []
    for k, idx in enumerate(idxs, start=1):
        rows.append([
            str(k),
            f"{float(x[idx]):.3f}",
            _arr_at(results.N, idx),
            _arr_at(results.V, idx),
            _arr_at(results.M, idx),
            _arr_at(results.T, idx),
            _arr_at(results.sigma, idx),
            _arr_at(results.tau_torsion, idx),
            _arr_at(results.margin, idx),
        ])
    return rows


def _arr_at(arr: np.ndarray, idx: int) -> str:
    a = np.asarray(arr, dtype=float)
    if idx >= a.size:
        return "-"
    return f"{float(a[idx]):.6g}"


def _build_plot_image_tag(results: SolveOutput) -> str:
    x = np.asarray(results.x_diag, dtype=float)
    if x.size == 0:
        return "<div class='muted'>无图形数据</div>"

    fig, axes = plt.subplots(3, 2, figsize=(10.2, 9.4), dpi=140, constrained_layout=True)
    specs = [
        ("N (N)", np.asarray(results.N, dtype=float)),
        ("V (N)", np.asarray(results.V, dtype=float)),
        ("M (N·mm)", np.asarray(results.M, dtype=float)),
        ("dy (mm)", np.asarray(results.dy_diag, dtype=float)),
        ("σ (N/mm²)", np.asarray(results.sigma, dtype=float)),
        ("MS (-)", np.asarray(results.margin, dtype=float)),
    ]
    for ax, (label, arr) in zip(axes.flat, specs):
        if arr.size == x.size:
            ax.plot(x, arr, color="#2f6fab", linewidth=1.2)
        ax.set_title(label, fontsize=10)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("x (mm)")

    buff = BytesIO()
    fig.savefig(buff, format="png")
    plt.close(fig)
    return f"<img alt='结果图' src='data:image/png;base64,{b64encode(buff.getvalue()).decode('ascii')}' />"


def _build_model_image_tag(project: Project) -> str:
    points = project.sorted_points()
    if not points:
        return "<div class='muted'>无模型数据</div>"

    fig, ax = plt.subplots(figsize=(10.2, 2.8), dpi=140)
    for m in project.members.values():
        pi = project.points.get(m.i_uid)
        pj = project.points.get(m.j_uid)
        if pi is None or pj is None:
            continue
        ax.plot([pi.x, pj.x], [0.0, 0.0], color="#1f2d3d", linewidth=2.5)
        ax.text((pi.x + pj.x) * 0.5, 0.06, m.name or m.uid, ha="center", va="bottom", fontsize=8)

    for p in points:
        ax.scatter([p.x], [0.0], color="#2f6fab", s=25, zorder=3)
        ax.text(p.x, -0.1, p.name or p.uid, ha="center", va="top", fontsize=8)

    ax.set_title("建模截图（梁线模型）", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    buff = BytesIO()
    fig.tight_layout()
    fig.savefig(buff, format="png")
    plt.close(fig)
    return f"<img alt='模型图' src='data:image/png;base64,{b64encode(buff.getvalue()).decode('ascii')}' />"


def _build_fbd_image_tag(project: Project, results: SolveOutput) -> str:
    points = project.sorted_points()
    if not points:
        return "<div class='muted'>无 FBD 数据</div>"

    fig, ax = plt.subplots(figsize=(10.2, 2.8), dpi=140)
    xs = [p.x for p in points]
    ax.plot([min(xs), max(xs)], [0.0, 0.0], color="#444", linewidth=2)

    for p in points:
        for ld in p.nodal_loads:
            if ld.direction not in ("FY", "FZ"):
                continue
            sign = -1 if ld.value < 0 else 1
            y2 = 0.25 * sign
            ax.annotate("", xy=(p.x, y2), xytext=(p.x, 0.0), arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))
            ax.text(p.x, y2 + (0.02 * sign), f"{ld.direction}={ld.value:.3g}", color="#c0392b", ha="center", va="bottom" if sign > 0 else "top", fontsize=8)

        rxn = (results.reactions or {}).get(p.name, {}) if p.name else {}
        for key in ("FY", "FZ"):
            val = float(rxn.get(key, 0.0) or 0.0)
            if abs(val) <= 0.0:
                continue
            sign = 1 if val > 0 else -1
            y2 = -0.25 * sign
            ax.annotate("", xy=(p.x, y2), xytext=(p.x, 0.0), arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))
            ax.text(p.x, y2 - (0.02 * sign), f"R{key}={val:.3g}", color="#27ae60", ha="center", va="top" if sign > 0 else "bottom", fontsize=8)

    ax.set_title("FBD（红: 荷载, 绿: 支反力）", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_yticks([])
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    buff = BytesIO()
    fig.tight_layout()
    fig.savefig(buff, format="png")
    plt.close(fig)
    return f"<img alt='FBD图' src='data:image/png;base64,{b64encode(buff.getvalue()).decode('ascii')}' />"


def _build_material_section_image_tag(project: Project) -> str:
    mats = sorted(project.materials.values(), key=lambda x: x.name)
    secs = sorted(project.sections.values(), key=lambda x: x.name)
    if not mats and not secs:
        return "<div class='muted'>无材料/截面数据</div>"

    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=140)
    ax.axis("off")
    y = 0.95
    ax.text(0.02, y, "材料信息", fontsize=10, fontweight="bold", transform=ax.transAxes)
    y -= 0.06
    for m in mats[:8]:
        ax.text(0.03, y, f"{m.name}: E={m.E:.3g}, G={m.G:.3g}, ν={m.nu:.3g}, σy={m.sigma_y:.3g}", fontsize=8, transform=ax.transAxes)
        y -= 0.05

    y -= 0.03
    ax.text(0.02, y, "截面信息", fontsize=10, fontweight="bold", transform=ax.transAxes)
    y -= 0.06
    for s in secs[:8]:
        ax.text(0.03, y, f"{s.name}({s.type}): A={s.A:.3g}, Iy={s.Iy:.3g}, Iz={s.Iz:.3g}, J={s.J:.3g}", fontsize=8, transform=ax.transAxes)
        y -= 0.05

    buff = BytesIO()
    fig.tight_layout()
    fig.savefig(buff, format="png")
    plt.close(fig)
    return f"<img alt='材料截面图' src='data:image/png;base64,{b64encode(buff.getvalue()).decode('ascii')}' />"
