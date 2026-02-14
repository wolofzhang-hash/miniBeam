# MiniBeam v1.0

A small 1D beam FEM GUI tool (Ribbon-like UI + right-click modeling) using **PyNiteFEA/Pynite** as the solver.

## v1.0 更新说明
- 统一版本号为 **1.0.0**（代码版本、窗口标题与文档标题保持一致）。
- 稳定提供一维梁建模、线弹性求解、结果图与 MS 安全裕度评估等 Phase-1 核心能力。
- 持续优化功能说明与用户手册内容，便于快速上手与团队交付。

## Features (Phase-1)
- Right-click add points on a 1D axis; drag points along X only; double-click to edit X.
- Auto-build members left-to-right (M1..Mn).
- Materials manager; Sections wizard (Rect solid, Circle solid, I section).
- Constraints: UX / UY / RZ (value supported; default 0).
- Loads: nodal Fy, nodal Mz, member UDL (w in Y).
- Solve (linear) using Pynite; results:
  - FBD (loads + reactions)
  - Deflection (DY along beam, sampled on unified diagram x, scaled)
  - Rotation θ (RZ) / Shear V (Fy) / Moment M (Mz) diagrams (member arrays)
  - Stress (sigma_bend from M*c/I) and Margin (sigma_y/FS / |sigma| - 1)

## Install
```bash
pip install -r requirements.txt
python run.py
```

> If PyNite is not installed, the app still runs but Solve is disabled until you install PyNiteFEA.

## Notes
- Model coordinates are in **mm** by default. Forces in **N**, moments in **N·mm**, distributed loads in **N/mm**.

## Export aligned member CSV (same x for every quantity)
```python
from minibeam.core.export_results import export_member_aligned_csv

# model: a solved PyNite FEModel3D instance
rows = export_member_aligned_csv(
    model=model,
    member_name="M1",
    csv_path="out/M1_aligned.csv",
    n_div=100,
    combo_name="Combo 1",  # optional, default is Combo 1
    shear_dir="Fy",
    moment_dir="Mz",
    disp_dirs=("UY",),
    rot_dirs=("RZ",),
)

# quick checks: aligned, monotonic, include endpoints
assert rows[0]["x_local"] == 0.0
assert rows[-1]["x_local"] > 0.0
assert all(rows[i]["x_local"] <= rows[i + 1]["x_local"] for i in range(len(rows) - 1))
```
