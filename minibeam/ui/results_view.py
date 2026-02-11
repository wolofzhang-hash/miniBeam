from __future__ import annotations
from typing import Optional
import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..core.model import Project
from ..core.pynite_adapter import SolveOutput


class ResultsView(QWidget):
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

        if rtype == "FBD":
            # Plot baseline and show loads/reactions at nodes
            x = _norm(out.x_nodes)
            y0 = np.zeros_like(x)
            ax.plot(x, y0)

            # reactions
            for i, px in enumerate(x):
                name = f"P{i+1}"
                r = out.reactions.get(name, {})
                fy = r.get("FY", 0.0)
                if abs(fy) > 1e-9:
                    ax.annotate(f"R={fy:.1f}", (px, 0), xytext=(px, 30), textcoords="offset points",
                                arrowprops=dict(arrowstyle="->"))

            ax.set_xlabel("x (mm)")
            ax.set_ylabel("FBD")
            ax.set_title("Free Body Diagram (Reactions shown)")

        elif rtype == "Deflection":
            x = _norm(out.x_nodes)
            dy = np.array(out.dy_nodes) * def_scale
            ax.plot(x, dy)
            ax.axhline(0, linewidth=1)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel(f"DY x{def_scale:g} (mm)")
            ax.set_title("Deflection (scaled)")

            idx = int(np.argmin(dy))
            ax.annotate(f"min {dy[idx]:.3f}", (x[idx], dy[idx]))

        elif rtype == "Shear V":
            xv, yv = _clip(out.x_diag, out.V)
            ax.plot(_norm(xv), yv)
            ax.axhline(0, linewidth=1)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("V (N)")
            ax.set_title("Shear V (Fy)")

        elif rtype == "Moment M":
            xm, ym = _clip(out.x_diag, out.M)
            ax.plot(_norm(xm), ym)
            ax.axhline(0, linewidth=1)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("M (N·mm)")
            ax.set_title("Moment M (Mz)")

        elif rtype == "Stress σ":
            xs, ys = _clip(out.x_diag, out.sigma)
            ax.plot(_norm(xs), ys)
            ax.axhline(0, linewidth=1)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("sigma (N/mm²)")
            ax.set_title("Bending Stress sigma = M*c/I")

        elif rtype == "Margin MS":
            xm2, ym2 = _clip(out.x_diag, out.margin)
            ax.plot(_norm(xm2), ym2)
            ax.axhline(0, linewidth=1)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("MS")
            ax.set_title("Margin of Safety (allow/|sigma|-1)")
            idx = int(np.argmin(out.margin))
            ax.annotate(f"min {out.margin[idx]:.3f}", (out.x_diag[idx], out.margin[idx]))

        self.fig.tight_layout()
        
        try:
            ax.set_xlim(0, max(0.0, x1 - x0))
        except Exception:
            pass
        self.canvas.draw()
