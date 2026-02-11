# MiniBeam v0.1.1 (Phase-1)

A small 1D beam FEM GUI tool (Ribbon-like UI + right-click modeling) using **PyNiteFEA/Pynite** as the solver.

## Features (Phase-1)
- Right-click add points on a 1D axis; drag points along X only; double-click to edit X.
- Auto-build members left-to-right (M1..Mn).
- Materials manager; Sections wizard (Rect solid, Circle solid, I section).
- Constraints: UX / UY / RZ (value supported; default 0).
- Loads: nodal Fy, nodal Mz, member UDL (w in Y).
- Solve (linear) using Pynite; results:
  - FBD (loads + reactions)
  - Deflection (DY at nodes, scaled)
  - Shear V (Fy) / Moment M (Mz) diagrams (member arrays)
  - Stress (sigma_bend from M*c/I) and Margin (sigma_y/FS / |sigma| - 1)

## Install
```bash
pip install -r requirements.txt
python run.py
```

> If PyNite is not installed, the app still runs but Solve is disabled until you install PyNiteFEA.

## Notes
- Model coordinates are in **mm** by default. Forces in **N**, moments in **N·mm**, distributed loads in **N/mm**.
