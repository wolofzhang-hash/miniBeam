from __future__ import annotations
from typing import Optional
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..core.model import Project
from ..core.pynite_adapter import SolveOutput


class ResultsView(QWidget):

    def _annotate_extrema_and_nodes(self, ax, x_plot, y_plot, x_nodes_plot=None):
        x_arr = np.asarray(x_plot, dtype=float)
        y_arr = np.asarray(y_plot, dtype=float)
        if x_arr.size == 0 or y_arr.size == 0:
            return

        idx_max = int(np.argmax(y_arr))
        idx_min = int(np.argmin(y_arr))
        ax.scatter([x_arr[idx_max], x_arr[idx_min]], [y_arr[idx_max], y_arr[idx_min]], color="#d62728", s=20, zorder=3)
        ax.annotate(f"max {y_arr[idx_max]:.3f}", (x_arr[idx_max], y_arr[idx_max]), textcoords="offset points", xytext=(6, 8), fontsize=8)
        ax.annotate(f"min {y_arr[idx_min]:.3f}", (x_arr[idx_min], y_arr[idx_min]), textcoords="offset points", xytext=(6, -12), fontsize=8)

        if x_nodes_plot is None:
            x_nodes_plot = x_arr
        nodes = np.asarray(x_nodes_plot, dtype=float)
        if nodes.size == 0:
            return
        # Interpolate values at point-node locations so every point is labeled.
        y_nodes = np.interp(nodes, x_arr, y_arr)
        ax.scatter(nodes, y_nodes, color="#1f77b4", s=14, zorder=3)
        for xn, yn in zip(nodes, y_nodes):
            ax.annotate(f"{yn:.3f}", (xn, yn), textcoords="offset points", xytext=(4, 4), fontsize=7, color="#333333")

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self.title = QLabel("Results")
        lay.addWidget(self.title)

        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas, 1)

    def set_data(self, prj: Project, out: SolveOutput, rtype: str, def_scale: float = 50.0):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.title.setText(f"Results - {rtype} ({out.combo})")

        # Normalize x-axis: leftmost point is x=0, limit plots to beam span
        try:
            xs_pts = [p.x for p in prj.points.values()]
            x0 = float(min(xs_pts)) if xs_pts else 0.0
            x1 = float(max(xs_pts)) if xs_pts else 0.0
        except Exception:
            x0, x1 = 0.0, 0.0

        def _norm(arr):
            a = np.asarray(arr, dtype=float)
            return a - x0 if a.size else a

        def _clip(xa, ya):
            xa = np.asarray(xa, dtype=float)
            ya = np.asarray(ya, dtype=float)
            if xa.size == 0:
                return xa, ya
            mask = (xa >= x0 - 1e-9) & (xa <= x1 + 1e-9)
            return xa[mask], ya[mask]

        def _draw_zero_line():
            ax.axhline(0, linewidth=1, color="#d3d3d3", linestyle="--")

        if rtype == "FBD":
            # FBD is rendered on the main UI canvas (with complete support
            # reactions). Keep this plot intentionally empty.
            ax.text(
                0.5,
                0.5,
                "FBD is shown on the model canvas\nafter Solve.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="#444444",
            )
            ax.set_axis_off()

        elif rtype == "Deflection":
            if getattr(out, "x_diag", None) is not None and np.asarray(out.x_diag).size:
                x, dy_raw = _clip(out.x_diag, out.dy_diag)
                x = _norm(x)
                dy = np.asarray(dy_raw, dtype=float) * def_scale
            else:
                x = _norm(out.x_nodes)
                dy = np.asarray(out.dy_nodes, dtype=float) * def_scale
            ax.plot(x, dy)
            _draw_zero_line()
            ax.set_xlabel("x (mm)")
            ax.set_ylabel(f"DY x{def_scale:g} (mm)")
            ax.set_title("Deflection (scaled)")

            self._annotate_extrema_and_nodes(ax, x, dy, _norm(out.x_nodes))

        elif rtype == "Rotation θ":
            xr, yr = _clip(out.x_diag, out.rz_diag)
            ax.plot(_norm(xr), yr)
            _draw_zero_line()
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("θz (rad)")
            ax.set_title("Rotation θ (RZ)")
            self._annotate_extrema_and_nodes(ax, _norm(xr), yr, _norm(out.x_nodes))

        elif rtype == "Shear V":
            xv, yv = _clip(out.x_diag, out.V)
            ax.plot(_norm(xv), yv)
            _draw_zero_line()
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("V (N)")
            ax.set_title("Shear V (Fy)")
            self._annotate_extrema_and_nodes(ax, _norm(xv), yv, _norm(out.x_nodes))

        elif rtype == "Axial N":
            xn, yn = _clip(out.x_diag, getattr(out, "N", np.zeros_like(out.x_diag)))
            ax.plot(_norm(xn), yn)
            _draw_zero_line()
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("N (N)")
            ax.set_title("Axial Force N (Fx)")
            self._annotate_extrema_and_nodes(ax, _norm(xn), yn, _norm(out.x_nodes))

        elif rtype == "Moment M":
            xm, ym = _clip(out.x_diag, out.M)
            ax.plot(_norm(xm), ym)
            _draw_zero_line()
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("M (N·mm)")
            ax.set_title("Moment M (Mz)")
            self._annotate_extrema_and_nodes(ax, _norm(xm), ym, _norm(out.x_nodes))

        elif rtype == "Torsion T":
            xt, yt = _clip(out.x_diag, getattr(out, "T", np.zeros_like(out.x_diag)))
            ax.plot(_norm(xt), yt)
            _draw_zero_line()
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("T (N·mm)")
            ax.set_title("Torsion / Torque (about X)")
            self._annotate_extrema_and_nodes(ax, _norm(xt), yt, _norm(out.x_nodes))

        elif rtype == "Torsion τ":
            xtau, ytau = _clip(out.x_diag, getattr(out, "tau_torsion", np.zeros_like(out.x_diag)))
            ax.plot(_norm(xtau), ytau)
            _draw_zero_line()
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("tau (MPa)")
            ax.set_title("Torsion Shear Stress (simplified)")
            self._annotate_extrema_and_nodes(ax, _norm(xtau), ytau, _norm(out.x_nodes))

        elif rtype == "Stress σ":
            xs, ys = _clip(out.x_diag, out.sigma)
            ax.plot(_norm(xs), ys)
            _draw_zero_line()
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("sigma (MPa)")
            ax.set_title("Bending Stress sigma = M*c/I")
            self._annotate_extrema_and_nodes(ax, _norm(xs), ys, _norm(out.x_nodes))

        elif rtype == "Margin MS":
            xm2, ym2 = _clip(out.x_diag, out.margin)
            ax.plot(_norm(xm2), ym2)
            _draw_zero_line()
            ax.set_ylim(-1, 2)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("MS")
            ax.set_title("Margin of Safety (allow/|sigma|-1)")
            self._annotate_extrema_and_nodes(ax, _norm(xm2), ym2, _norm(out.x_nodes))

        self.fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90)
        
        try:
            ax.set_xlim(0, max(0.0, x1 - x0))
        except Exception:
            pass
        self.canvas.draw()
