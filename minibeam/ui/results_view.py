from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..core.model import Project
from ..core.pynite_adapter import SolveOutput


class ResultsView(QWidget):

    @staticmethod
    def _annotate_extrema_and_nodes(ax, x_plot, y_plot, x_nodes_plot=None, text_scale: float = 1.0):
        x_arr = np.asarray(x_plot, dtype=float)
        y_arr = np.asarray(y_plot, dtype=float)
        if x_arr.size == 0 or y_arr.size == 0:
            return

        idx_max = int(np.argmax(y_arr))
        idx_min = int(np.argmin(y_arr))
        ax.scatter([x_arr[idx_max], x_arr[idx_min]], [y_arr[idx_max], y_arr[idx_min]], color="#d62728", s=20, zorder=3)
        ext_fontsize = max(7, int(round(8 * text_scale)))
        node_fontsize = max(6, int(round(7 * text_scale)))
        ax.annotate(
            f"max {y_arr[idx_max]:.3f}",
            (x_arr[idx_max], y_arr[idx_max]),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=ext_fontsize,
        )
        ax.annotate(
            f"min {y_arr[idx_min]:.3f}",
            (x_arr[idx_min], y_arr[idx_min]),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=ext_fontsize,
        )

        if x_nodes_plot is None:
            x_nodes_plot = x_arr
        nodes = np.asarray(x_nodes_plot, dtype=float)
        if nodes.size == 0:
            return
        # Interpolate values at point-node locations so every point is labeled.
        y_nodes = np.interp(nodes, x_arr, y_arr)
        ax.scatter(nodes, y_nodes, color="#1f77b4", s=14, zorder=3)
        for xn, yn in zip(nodes, y_nodes):
            ax.annotate(
                f"{yn:.3f}",
                (xn, yn),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=node_fontsize,
                color="#333333",
            )

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

    @staticmethod
    def _plot_result_type(
        ax,
        prj: Project,
        out: SolveOutput,
        rtype: str,
        def_scale: float = 50.0,
        include_title: bool = True,
        text_scale: float = 1.0,
    ):

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

        label_fontsize = max(8, int(round(10 * text_scale)))
        tick_fontsize = max(7, int(round(9 * text_scale)))
        title_fontsize = max(10, int(round(14 * text_scale)))

        def _set_labels(xlabel: str, ylabel: str, title: str):
            ax.set_xlabel(xlabel, fontsize=label_fontsize)
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            if include_title:
                ax.set_title(title, fontsize=title_fontsize, pad=8)

        x_for_click = np.array([], dtype=float)
        y_for_click = np.array([], dtype=float)

        if rtype == "Deflection":
            if getattr(out, "x_diag", None) is not None and np.asarray(out.x_diag).size:
                x, dy_raw = _clip(out.x_diag, out.dy_diag)
                x = _norm(x)
                dy = np.asarray(dy_raw, dtype=float) * def_scale
            else:
                x = _norm(out.x_nodes)
                dy = np.asarray(out.dy_nodes, dtype=float) * def_scale
            ax.plot(x, dy)
            _draw_zero_line()
            _set_labels("x (mm)", f"DY x{def_scale:g} (mm)", "Deflection (scaled)")

            ResultsView._annotate_extrema_and_nodes(ax, x, dy, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = np.asarray(x, dtype=float), np.asarray(dy, dtype=float)

        elif rtype == "Rotation θ":
            xr, yr = _clip(out.x_diag, out.rz_diag)
            ax.plot(_norm(xr), yr)
            _draw_zero_line()
            _set_labels("x (mm)", "θz (rad)", "Rotation θ (RZ)")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xr), yr, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xr), np.asarray(yr, dtype=float)

        elif rtype == "Deflection Z":
            x, dz_raw = _clip(out.x_diag, getattr(out, "dz_diag", np.zeros_like(out.x_diag)))
            x = _norm(x)
            dz = np.asarray(dz_raw, dtype=float) * def_scale
            ax.plot(x, dz)
            _draw_zero_line()
            _set_labels("x (mm)", f"DZ x{def_scale:g} (mm)", "Deflection Z (scaled)")
            ResultsView._annotate_extrema_and_nodes(ax, x, dz, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = np.asarray(x, dtype=float), np.asarray(dz, dtype=float)

        elif rtype == "Rotation Y":
            xr, yr = _clip(out.x_diag, getattr(out, "ry_diag", np.zeros_like(out.x_diag)))
            ax.plot(_norm(xr), yr)
            _draw_zero_line()
            _set_labels("x (mm)", "θy (rad)", "Rotation θ (RY)")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xr), yr, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xr), np.asarray(yr, dtype=float)

        elif rtype == "Shear V":
            xv, yv = _clip(out.x_diag, out.V)
            ax.plot(_norm(xv), yv)
            _draw_zero_line()
            _set_labels("x (mm)", "V (N)", "Shear V (Fy)")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xv), yv, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xv), np.asarray(yv, dtype=float)

        elif rtype == "Axial N":
            xn, yn = _clip(out.x_diag, getattr(out, "N", np.zeros_like(out.x_diag)))
            ax.plot(_norm(xn), yn)
            _draw_zero_line()
            _set_labels("x (mm)", "N (N)", "Axial Force N (Fx)")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xn), yn, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xn), np.asarray(yn, dtype=float)

        elif rtype == "Shear Vz":
            xv, yv = _clip(out.x_diag, getattr(out, "Vz", np.zeros_like(out.x_diag)))
            ax.plot(_norm(xv), yv)
            _draw_zero_line()
            _set_labels("x (mm)", "Vz (N)", "Shear Vz (Fz)")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xv), yv, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xv), np.asarray(yv, dtype=float)

        elif rtype == "Moment M":
            xm, ym = _clip(out.x_diag, out.M)
            ax.plot(_norm(xm), ym)
            _draw_zero_line()
            _set_labels("x (mm)", "M (N·mm)", "Moment M (Mz)")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xm), ym, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xm), np.asarray(ym, dtype=float)

        elif rtype == "Moment My":
            xm, ym = _clip(out.x_diag, getattr(out, "My", np.zeros_like(out.x_diag)))
            ax.plot(_norm(xm), ym)
            _draw_zero_line()
            _set_labels("x (mm)", "My (N·mm)", "Moment My")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xm), ym, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xm), np.asarray(ym, dtype=float)

        elif rtype == "Torsion T":
            xt, yt = _clip(out.x_diag, getattr(out, "T", np.zeros_like(out.x_diag)))
            ax.plot(_norm(xt), yt)
            _draw_zero_line()
            _set_labels("x (mm)", "T (N·mm)", "Torsion / Torque (about X)")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xt), yt, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xt), np.asarray(yt, dtype=float)

        elif rtype == "Torsion τ":
            xtau, ytau = _clip(out.x_diag, getattr(out, "tau_torsion", np.zeros_like(out.x_diag)))
            ax.plot(_norm(xtau), ytau)
            _draw_zero_line()
            _set_labels("x (mm)", "tau (MPa)", "Torsion Shear Stress (simplified)")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xtau), ytau, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xtau), np.asarray(ytau, dtype=float)

        elif rtype == "Stress σ":
            xs, ys = _clip(out.x_diag, out.sigma)
            ax.plot(_norm(xs), ys)
            _draw_zero_line()
            _set_labels("x (mm)", "sigma (MPa)", "Normal Stress sigma = N/A + M*c/(I*k)")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xs), ys, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xs), np.asarray(ys, dtype=float)

        elif rtype == "Margin MS":
            xm2, ym2 = _clip(out.x_diag, out.margin)
            ax.plot(_norm(xm2), ym2)
            _draw_zero_line()
            ax.set_ylim(-1, 2)
            _set_labels("x (mm)", "MS", "Margin of Safety (plastic-corrected: allow/|sigma|-1)")
            ResultsView._annotate_extrema_and_nodes(ax, _norm(xm2), ym2, _norm(out.x_nodes), text_scale=text_scale)
            x_for_click, y_for_click = _norm(xm2), np.asarray(ym2, dtype=float)

        try:
            ax.set_xlim(0, max(0.0, x1 - x0))
        except Exception:
            pass
        ax.tick_params(axis="both", labelsize=tick_fontsize)

        return x_for_click, y_for_click

    def set_data(self, prj: Project, out: SolveOutput, rtype: str, def_scale: float = 50.0):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.title.setText(f"Results - {rtype} ({out.combo})")
        self._plot_result_type(ax, prj, out, rtype, def_scale=def_scale, include_title=False)
        self.fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.96)
        self.canvas.draw()


@dataclass
class PlotSlot:
    container: QWidget
    combo: QComboBox
    fig: Figure
    canvas: FigureCanvas
    coord_label: QLabel
    annotation: object = None
    x_data: np.ndarray | None = None
    y_data: np.ndarray | None = None


class ResultsGridDialog(QDialog):
    RESULT_TYPES = [
        "Deflection",
        "Rotation θ",
        "Deflection Z",
        "Rotation Y",
        "Axial N",
        "Shear V",
        "Shear Vz",
        "Moment M",
        "Moment My",
        "Torsion T",
        "Torsion τ",
        "Stress σ",
        "Margin MS",
    ]

    LAYOUT_MODES = {
        "1 x 1": (1, 1),
        "1 x 2": (1, 2),
        "2 x 2": (2, 2),
        "2 x 3": (2, 3),
        "3 x 3": (3, 3),
        "4 x 1": (4, 1),
        "4 x 2": (4, 2),
        "4 x 3": (4, 3),
    }

    def __init__(self, prj: Project, out: SolveOutput, def_scale: float = 1.0, parent=None):
        super().__init__(parent)
        self.project = prj
        self.out = out
        self.def_scale = def_scale
        self.setWindowTitle(f"Results Multi-View ({out.combo})")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.resize(1400, 900)

        root = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        root.addLayout(toolbar)

        self.layout_combo = QComboBox()
        self.layout_combo.addItems(list(self.LAYOUT_MODES.keys()))
        self.layout_combo.setCurrentText("2 x 2")
        toolbar.addWidget(QLabel("Layout"))
        toolbar.addWidget(self.layout_combo)

        self.btn_export_all = QPushButton("Export All Images")
        toolbar.addWidget(self.btn_export_all)
        toolbar.addStretch(1)

        self.grid_host = QWidget()
        self.grid_layout = QGridLayout(self.grid_host)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setHorizontalSpacing(8)
        self.grid_layout.setVerticalSpacing(8)
        root.addWidget(self.grid_host, 1)

        self.slots: list[PlotSlot] = []
        self.layout_combo.currentTextChanged.connect(self._rebuild_grid)
        self.btn_export_all.clicked.connect(self.export_all_images)
        self._rebuild_grid()

    def _clear_grid(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.slots = []

    def _rebuild_grid(self):
        self._clear_grid()
        rows, cols = self.LAYOUT_MODES[self.layout_combo.currentText()]
        total = rows * cols

        for idx in range(total):
            panel = QWidget()
            vbox = QVBoxLayout(panel)

            cmb = QComboBox()
            cmb.addItem("(Empty)")
            cmb.addItems(self.RESULT_TYPES)
            if idx < len(self.RESULT_TYPES):
                cmb.setCurrentIndex(idx + 1)
            else:
                cmb.setCurrentIndex(0)
            vbox.addWidget(cmb)

            fig = Figure(figsize=(5, 3), dpi=100)
            canvas = FigureCanvas(fig)
            vbox.addWidget(canvas, 1)

            coord_label = QLabel("Click curve point to inspect coordinates")
            vbox.addWidget(coord_label)

            slot = PlotSlot(panel, cmb, fig, canvas, coord_label)
            cmb.currentTextChanged.connect(lambda _=None, s=slot: self._draw_slot(s))
            canvas.mpl_connect("button_press_event", lambda event, s=slot: self._on_plot_click(event, s))
            self.slots.append(slot)
            self.grid_layout.addWidget(panel, idx // cols, idx % cols)

        for row in range(rows):
            self.grid_layout.setRowStretch(row, 1)
        for col in range(cols):
            self.grid_layout.setColumnStretch(col, 1)

        for slot in self.slots:
            self._draw_slot(slot)

    def _draw_slot(self, slot: PlotSlot):
        slot.fig.clear()
        ax = slot.fig.add_subplot(111)
        rtype = slot.combo.currentText()
        if rtype == "(Empty)":
            ax.set_axis_off()
            slot.x_data = np.array([], dtype=float)
            slot.y_data = np.array([], dtype=float)
        else:
            canvas_width = max(420, slot.canvas.width())
            text_scale = max(0.9, min(1.25, canvas_width / 760.0))
            x_data, y_data = ResultsView._plot_result_type(
                ax,
                self.project,
                self.out,
                rtype,
                self.def_scale,
                include_title=False,
                text_scale=text_scale,
            )
            slot.x_data = np.asarray(x_data, dtype=float)
            slot.y_data = np.asarray(y_data, dtype=float)
        slot.coord_label.setText("Click curve point to inspect coordinates")
        slot.annotation = None
        slot.fig.subplots_adjust(left=0.12, right=0.98, bottom=0.20, top=0.96)
        slot.canvas.draw_idle()

    def _on_plot_click(self, event, slot: PlotSlot):
        if event.inaxes is None or slot.x_data is None or slot.y_data is None or slot.x_data.size == 0:
            return
        x = float(event.xdata)
        idx = int(np.argmin(np.abs(slot.x_data - x)))
        xp = float(slot.x_data[idx])
        yp = float(slot.y_data[idx])
        slot.coord_label.setText(f"x={xp:.6g}, y={yp:.6g}")

        if slot.annotation is not None:
            try:
                slot.annotation.remove()
            except Exception:
                pass
        slot.annotation = event.inaxes.annotate(
            f"({xp:.4g}, {yp:.4g})",
            (xp, yp),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=8,
            color="#111111",
            bbox={"boxstyle": "round,pad=0.2", "fc": "#fff8dc", "ec": "#888888", "alpha": 0.9},
        )
        event.inaxes.scatter([xp], [yp], color="#ff0000", s=20, zorder=4)
        slot.canvas.draw_idle()

    def export_all_images(self):
        out_dir = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not out_dir:
            return
        import re
        from pathlib import Path

        saved = 0
        for idx, slot in enumerate(self.slots, start=1):
            rtype = slot.combo.currentText()
            if rtype == "(Empty)":
                continue
            safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", rtype).strip("_")
            fn = Path(out_dir) / f"result_{idx:02d}_{safe}.png"
            export_fig = Figure(figsize=(7.2, 4.0), dpi=180)
            export_ax = export_fig.add_subplot(111)
            ResultsView._plot_result_type(
                export_ax,
                self.project,
                self.out,
                rtype,
                self.def_scale,
                include_title=True,
                text_scale=1.1,
            )
            export_fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.90)
            export_fig.savefig(fn, dpi=180)
            saved += 1

        if saved == 0:
            QMessageBox.information(self, "Export", "No non-empty plots to export.")
        else:
            QMessageBox.information(self, "Export", f"Exported {saved} image(s) to\n{out_dir}")
