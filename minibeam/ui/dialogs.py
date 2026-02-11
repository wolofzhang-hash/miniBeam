from __future__ import annotations
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QFormLayout, QLineEdit,
    QDoubleSpinBox, QLabel, QMessageBox, QComboBox, QGroupBox
)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QPolygonF

from ..core.model import Project, Material, Section
from ..core.section_props import rect_solid, circle_solid, i_section, rect_hollow, circle_hollow
from ..core.library_store import save_material_library, save_section_library


class SectionPreview(QLabel):
    """A lightweight 2D sketch of the current section (for wizard preview)."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._typ = "RectSolid"
        self._p = (100.0, 10.0, 0.0, 0.0)

    def _draw_dimension(self, qp: QPainter, x1: float, y1: float, x2: float, y2: float, text: str):
        """Draw simple dimension line with arrow heads and centered label."""
        qp.setPen(QPen(Qt.GlobalColor.darkGray, 1))
        qp.drawLine(int(x1), int(y1), int(x2), int(y2))

        dx = x2 - x1
        dy = y2 - y1
        ln = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / ln, dy / ln
        px, py = -uy, ux
        ah = 8.0
        aw = 4.0

        head1 = QPolygonF([
            QPointF(x1, y1),
            QPointF(x1 + ux * ah + px * aw, y1 + uy * ah + py * aw),
            QPointF(x1 + ux * ah - px * aw, y1 + uy * ah - py * aw),
        ])
        head2 = QPolygonF([
            QPointF(x2, y2),
            QPointF(x2 - ux * ah + px * aw, y2 - uy * ah + py * aw),
            QPointF(x2 - ux * ah - px * aw, y2 - uy * ah - py * aw),
        ])
        qp.setBrush(QBrush(Qt.GlobalColor.darkGray))
        qp.drawPolygon(head1)
        qp.drawPolygon(head2)

        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        qp.setPen(QPen(Qt.GlobalColor.black, 1))
        qp.drawText(int(mx + px * 12 - 20), int(my + py * 12 + 4), text)

    def set_section(self, typ: str, p1: float, p2: float, p3: float, p4: float):
        self._typ = typ
        self._p = (p1, p2, p3, p4)
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        w = self.width()
        h = self.height()
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)

        # frame
        qp.setPen(QPen(Qt.GlobalColor.lightGray, 1))
        qp.drawRect(4, 4, w - 8, h - 8)

        p1, p2, p3, p4 = self._p
        # Choose a rough drawing scale.
        box = min(w, h) * 0.75
        cx, cy = w / 2, h / 2

        qp.setPen(QPen(Qt.GlobalColor.black, 2))
        qp.setBrush(QBrush(Qt.GlobalColor.white))

        def draw_rect(b, hh):
            s = box / max(b, hh, 1e-6)
            bw = b * s
            bh = hh * s
            x0, y0 = cx - bw / 2, cy - bh / 2
            qp.drawRect(int(x0), int(y0), int(bw), int(bh))
            self._draw_dimension(qp, x0, y0 - 14, x0 + bw, y0 - 14, f"b={b:g}")
            self._draw_dimension(qp, x0 - 14, y0, x0 - 14, y0 + bh, f"h={hh:g}")

        if self._typ == "RectSolid":
            draw_rect(p1, p2)
        elif self._typ == "RectHollow":
            # p1=b, p2=h, p3=t
            b, hh, t = float(p1), float(p2), float(p3)
            s = box / max(b, hh, 1e-6)
            bw = b * s
            bh = hh * s
            qp.drawRect(int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh))
            if t > 0 and 2*t < min(b, hh):
                bi = (b - 2*t) * s
                hi = (hh - 2*t) * s
                qp.setBrush(QBrush(Qt.GlobalColor.white))
                qp.drawRect(int(cx - bi / 2), int(cy - hi / 2), int(bi), int(hi))
            x0, y0 = cx - bw / 2, cy - bh / 2
            self._draw_dimension(qp, x0, y0 - 14, x0 + bw, y0 - 14, f"b={b:g}")
            self._draw_dimension(qp, x0 - 14, y0, x0 - 14, y0 + bh, f"h={hh:g}")
            self._draw_dimension(qp, x0 + bw + 12, y0, x0 + bw + 12, y0 + t * s, f"t={t:g}")
        elif self._typ == "CircleSolid":
            d = max(p1, 1e-6)
            s = box / d
            rr = d * s / 2
            qp.drawEllipse(int(cx - rr), int(cy - rr), int(2 * rr), int(2 * rr))
            self._draw_dimension(qp, cx - rr, cy - rr - 14, cx + rr, cy - rr - 14, f"d={d:g}")
        elif self._typ == "CircleHollow":
            D = float(p1)
            t = float(p2)
            D = max(D, 1e-6)
            s = box / D
            R = D * s / 2
            qp.drawEllipse(int(cx - R), int(cy - R), int(2 * R), int(2 * R))
            if t > 0 and 2*t < D:
                r = (D/2 - t) * s
                qp.setBrush(QBrush(Qt.GlobalColor.white))
                qp.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
            self._draw_dimension(qp, cx - R, cy - R - 14, cx + R, cy - R - 14, f"D={D:g}")
            self._draw_dimension(qp, cx + R + 12, cy - R, cx + R + 12, cy - R + t * s, f"t={t:g}")
        else:
            # I section: h, bf, tf, tw
            h0, bf, tf, tw = p1, p2, p3, p4
            s = box / max(h0, bf, 1e-6)
            H = h0 * s
            B = bf * s
            TF = tf * s
            TW = tw * s
            # top flange
            qp.drawRect(int(cx - B / 2), int(cy - H / 2), int(B), int(TF))
            # bottom flange
            qp.drawRect(int(cx - B / 2), int(cy + H / 2 - TF), int(B), int(TF))
            # web
            qp.drawRect(int(cx - TW / 2), int(cy - H / 2 + TF), int(TW), int(H - 2 * TF))
            x0, y0 = cx - B / 2, cy - H / 2
            self._draw_dimension(qp, x0, y0 - 14, x0 + B, y0 - 14, f"bf={bf:g}")
            self._draw_dimension(qp, x0 - 14, y0, x0 - 14, y0 + H, f"h={h0:g}")
            self._draw_dimension(qp, x0 + B + 12, y0, x0 + B + 12, y0 + TF, f"tf={tf:g}")
            self._draw_dimension(qp, cx - TW / 2, y0 + H + 14, cx + TW / 2, y0 + H + 14, f"tw={tw:g}")

        qp.end()


class MaterialManagerDialog(QDialog):
    def __init__(self, prj: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Materials")
        self.resize(640, 420)
        self.prj = prj

        lay = QHBoxLayout(self)
        self.list = QListWidget()
        lay.addWidget(self.list, 1)

        right = QVBoxLayout()
        lay.addLayout(right, 2)

        form = QFormLayout()
        right.addLayout(form)

        self.ed_name = QLineEdit()
        self.ed_E = QDoubleSpinBox(); self.ed_E.setRange(0, 1e12); self.ed_E.setDecimals(3)
        self.ed_G = QDoubleSpinBox(); self.ed_G.setRange(0, 1e12); self.ed_G.setDecimals(3)
        class NuSpinBox(QDoubleSpinBox):
            def textFromValue(self, val: float) -> str:  # noqa: N802 (Qt API)
                v = float(val)
                if abs(v) < 1.0:
                    return f"{v:.2f}"
                return f"{v:.0f}"

        self.ed_nu = NuSpinBox(); self.ed_nu.setRange(0.0, 0.49); self.ed_nu.setDecimals(2)
        self.ed_fy = QDoubleSpinBox(); self.ed_fy.setRange(0, 1e6); self.ed_fy.setDecimals(3)

        form.addRow("Name", self.ed_name)
        form.addRow("E (N/mm²)", self.ed_E)
        form.addRow("G (N/mm²)", self.ed_G)
        form.addRow("nu", self.ed_nu)
        form.addRow("sigma_y (N/mm²)", self.ed_fy)

        btns = QHBoxLayout()
        right.addLayout(btns)
        self.btn_add = QPushButton("Add")
        self.btn_del = QPushButton("Delete")
        self.btn_active = QPushButton("Set Active ★")
        self.btn_save_lib = QPushButton("Save to Library")
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_del)
        btns.addWidget(self.btn_active)
        btns.addWidget(self.btn_save_lib)
        btns.addStretch(1)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_add.clicked.connect(self.add_material)
        self.btn_del.clicked.connect(self.delete_material)
        self.btn_active.clicked.connect(self.set_active)
        self.btn_save_lib.clicked.connect(self.save_library)
        self.list.currentRowChanged.connect(self.load_selected)
        # live update
        for w in [self.ed_name, self.ed_E, self.ed_G, self.ed_nu, self.ed_fy]:
            if hasattr(w, "textChanged"):
                w.textChanged.connect(self.update_selected)
            else:
                w.valueChanged.connect(self.update_selected)

        self._loading_selection = False
        self.refresh()

    def refresh(self, selected_uid: str | None = None):
        if selected_uid is None:
            selected_uid = self._current_uid()
        self.list.clear()
        selected_row = -1
        for mat in self.prj.materials.values():
            star = " ★" if mat.uid == self.prj.active_material_uid else ""
            self.list.addItem(f"{mat.name}{star} [{mat.uid}]")
            if mat.uid == selected_uid:
                selected_row = self.list.count() - 1
        if self.list.count() > 0:
            self.list.setCurrentRow(selected_row if selected_row >= 0 else 0)

    def _current_uid(self):
        row = self.list.currentRow()
        if row < 0:
            return None
        txt = self.list.item(row).text()
        uid = txt.split("[")[-1].split("]")[0]
        return uid

    def load_selected(self, _):
        uid = self._current_uid()
        if not uid or uid not in self.prj.materials:
            return
        m = self.prj.materials[uid]
        self._loading_selection = True
        self.ed_name.setText(m.name)
        self.ed_E.setValue(m.E)
        self.ed_G.setValue(m.G)
        self.ed_nu.setValue(m.nu)
        self.ed_fy.setValue(m.sigma_y)
        self._loading_selection = False

    def update_selected(self, *_):
        if self._loading_selection:
            return
        uid = self._current_uid()
        if not uid or uid not in self.prj.materials:
            return
        m = self.prj.materials[uid]
        m.name = self.ed_name.text().strip() or m.name
        m.E = float(self.ed_E.value())
        m.G = float(self.ed_G.value())
        m.nu = float(self.ed_nu.value())
        m.sigma_y = float(self.ed_fy.value())
        self.refresh(selected_uid=uid)

    def add_material(self):
        m = Material(name="Material", E=210000.0, G=81000.0, nu=0.3, rho=7.85e-6, sigma_y=355.0)
        self.prj.materials[m.uid] = m
        self.refresh()

    def delete_material(self):
        uid = self._current_uid()
        if not uid:
            return
        if uid == self.prj.active_material_uid:
            QMessageBox.warning(self, "Delete", "不能删除 Active material。请先切换 Active。")
            return
        self.prj.materials.pop(uid, None)
        self.refresh()

    def set_active(self):
        uid = self._current_uid()
        if not uid:
            return
        self.prj.active_material_uid = uid
        self.refresh(selected_uid=uid)

    def save_library(self):
        try:
            save_material_library(self.prj.materials)
            QMessageBox.information(self, "Materials", "材料已保存到材料库（~/.minibeam/materials.json）。")
        except Exception as e:
            QMessageBox.critical(self, "Materials", f"保存材料库失败：{e}")


class SectionManagerDialog(QDialog):
    def __init__(self, prj: Project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sections")
        self.resize(760, 520)
        self.prj = prj

        lay = QHBoxLayout(self)
        self.list = QListWidget()
        lay.addWidget(self.list, 1)

        right = QVBoxLayout()
        lay.addLayout(right, 2)

        # wizard
        gb = QGroupBox("Section Wizard (Phase-1)")
        right.addWidget(gb)
        self.form = QFormLayout(gb)

        self.cmb_type = QComboBox()
        self.cmb_type.addItems(["RectSolid", "RectHollow", "CircleSolid", "CircleHollow", "ISection"])
        self.ed_name = QLineEdit()

        self.sp1 = QDoubleSpinBox(); self.sp1.setRange(0.001, 1e9); self.sp1.setDecimals(3)
        self.sp2 = QDoubleSpinBox(); self.sp2.setRange(0.001, 1e9); self.sp2.setDecimals(3)
        self.sp3 = QDoubleSpinBox(); self.sp3.setRange(0.001, 1e9); self.sp3.setDecimals(3)
        self.sp4 = QDoubleSpinBox(); self.sp4.setRange(0.001, 1e9); self.sp4.setDecimals(3)
        self.sp5 = QDoubleSpinBox(); self.sp5.setRange(0.001, 1e9); self.sp5.setDecimals(3)

        self.lbl_p1 = QLabel("Param1")
        self.lbl_p2 = QLabel("Param2")
        self.lbl_p3 = QLabel("Param3")
        self.lbl_p4 = QLabel("Param4")
        self.lbl_p5 = QLabel("Param5")

        self.form.addRow("Type", self.cmb_type)
        self.form.addRow("Name", self.ed_name)
        self.form.addRow(self.lbl_p1, self.sp1)
        self.form.addRow(self.lbl_p2, self.sp2)
        self.form.addRow(self.lbl_p3, self.sp3)
        self.form.addRow(self.lbl_p4, self.sp4)
        self.form.addRow(self.lbl_p5, self.sp5)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setWordWrap(True)
        right.addWidget(self.lbl_hint)

        # preview + computed props
        self.preview = SectionPreview()
        right.addWidget(self.preview)
        self.lbl_props = QLabel("A= -   Iy= -   Iz= -   J= -   Zp= -   k= -")
        self.lbl_props.setWordWrap(True)
        right.addWidget(self.lbl_props)

        btns = QHBoxLayout()
        right.addLayout(btns)
        self.btn_add = QPushButton("Add / Update")
        self.btn_del = QPushButton("Delete")
        self.btn_active = QPushButton("Set Active ★")
        self.btn_save_lib = QPushButton("Save to Library")
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_del)
        btns.addWidget(self.btn_active)
        btns.addWidget(self.btn_save_lib)
        btns.addStretch(1)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_add.clicked.connect(self.add_or_update)
        self.btn_del.clicked.connect(self.delete_section)
        self.btn_active.clicked.connect(self.set_active)
        self.cmb_type.currentTextChanged.connect(self._update_hint)
        self.list.currentRowChanged.connect(self._load_selected)
        self.btn_save_lib.clicked.connect(self.save_library)
        # live preview
        for sp in [self.sp1, self.sp2, self.sp3, self.sp4, self.sp5]:
            sp.valueChanged.connect(self._update_preview)
        self.cmb_type.currentTextChanged.connect(lambda _: self._update_preview())

        self._loading_selection = False
        self._update_hint(self.cmb_type.currentText())
        self.refresh()

    def _update_preview(self):
        typ = self.cmb_type.currentText()
        p1, p2, p3, p4 = float(self.sp1.value()), float(self.sp2.value()), float(self.sp3.value()), float(self.sp4.value())
        self.preview.set_section(typ, p1, p2, p3, p4)
        # compute properties
        try:
            if typ == "RectSolid":
                props = rect_solid(p1, p2)
            elif typ == "RectHollow":
                props = rect_hollow(p1, p2, p3)
            elif typ == "CircleSolid":
                props = circle_solid(p1)
            elif typ == "CircleHollow":
                props = circle_hollow(p1, p2)
            else:
                props = i_section(p1, p2, p3, p4)
            self.lbl_props.setText(f"A={props.A:.3f} mm²   Iy={props.Iy:.3f} mm⁴   Iz={props.Iz:.3f} mm⁴   J={props.J:.3f} mm⁴   Zp={props.Zp:.3f} mm³   k={props.shape_factor:.3f}")
        except Exception:
            self.lbl_props.setText("A= -   Iy= -   Iz= -   J= -   Zp= -   k= -")

    def _update_hint(self, typ: str):
        param_defs = {
            "RectSolid": [("b(mm)", True), ("h(mm)", True), ("(unused)", False), ("(unused)", False), ("(unused)", False)],
            "RectHollow": [("b(mm)", True), ("h(mm)", True), ("t(mm)", True), ("(unused)", False), ("(unused)", False)],
            "CircleSolid": [("d(mm)", True), ("(unused)", False), ("(unused)", False), ("(unused)", False), ("(unused)", False)],
            "CircleHollow": [("D(mm)", True), ("t(mm)", True), ("(unused)", False), ("(unused)", False), ("(unused)", False)],
            "ISection": [("h(mm)", True), ("bf(mm)", True), ("tf(mm)", True), ("tw(mm)", True), ("(unused)", False)],
        }
        labels = [self.lbl_p1, self.lbl_p2, self.lbl_p3, self.lbl_p4, self.lbl_p5]
        spins = [self.sp1, self.sp2, self.sp3, self.sp4, self.sp5]
        for (title, visible), lbl, sp in zip(param_defs.get(typ, []), labels, spins):
            lbl.setText(title)
            lbl.setVisible(visible)
            sp.setVisible(visible)

        if typ == "RectSolid":
            self.lbl_hint.setText("RectSolid: Param1=b(mm), Param2=h(mm). Param3/4 ignored.")
            self.ed_name.setText("Rect100x10")
            self.sp1.setValue(100.0); self.sp2.setValue(10.0)
        elif typ == "RectHollow":
            self.lbl_hint.setText("RectHollow: Param1=b(mm), Param2=h(mm), Param3=t(mm). Param4 ignored.")
            self.ed_name.setText("RectTube100x50x5")
            self.sp1.setValue(100.0); self.sp2.setValue(50.0); self.sp3.setValue(5.0); self.sp4.setValue(0.0)
        elif typ == "CircleSolid":
            self.lbl_hint.setText("CircleSolid: Param1=d(mm). Others ignored.")
            self.ed_name.setText("Circle20")
            self.sp1.setValue(20.0); self.sp2.setValue(0.0); self.sp3.setValue(0.0); self.sp4.setValue(0.0)
        elif typ == "CircleHollow":
            self.lbl_hint.setText("CircleHollow: Param1=D(mm), Param2=t(mm). Others ignored.")
            self.ed_name.setText("Pipe60x4")
            self.sp1.setValue(60.0); self.sp2.setValue(4.0); self.sp3.setValue(0.0); self.sp4.setValue(0.0)
        else:
            self.lbl_hint.setText("ISection: Param1=h(mm), Param2=bf(mm), Param3=tf(mm), Param4=tw(mm)")
            self.ed_name.setText("I200")
            self.sp1.setValue(200.0); self.sp2.setValue(100.0); self.sp3.setValue(10.0); self.sp4.setValue(6.0)

    def refresh(self, selected_uid: str | None = None):
        if selected_uid is None:
            selected_uid = self._current_uid()
        self.list.clear()
        selected_row = -1
        for sec in self.prj.sections.values():
            star = " ★" if sec.uid == self.prj.active_section_uid else ""
            self.list.addItem(f"{sec.name} ({sec.type}){star} [{sec.uid}]")
            if sec.uid == selected_uid:
                selected_row = self.list.count() - 1
        if self.list.count() > 0:
            self.list.setCurrentRow(selected_row if selected_row >= 0 else 0)

    def _current_uid(self):
        row = self.list.currentRow()
        if row < 0:
            return None
        txt = self.list.item(row).text()
        uid = txt.split("[")[-1].split("]")[0]
        return uid

    def _load_selected(self, _):
        uid = self._current_uid()
        if not uid or uid not in self.prj.sections:
            return
        s = self.prj.sections[uid]
        self._loading_selection = True
        self.cmb_type.setCurrentText(s.type)
        self.ed_name.setText(s.name)
        self._loading_selection = False

    def add_or_update(self):
        typ = self.cmb_type.currentText()
        name = self.ed_name.text().strip() or "Section"
        if typ == "RectSolid":
            b = float(self.sp1.value()); h = float(self.sp2.value())
            props = rect_solid(b, h)
        elif typ == "RectHollow":
            b = float(self.sp1.value()); h = float(self.sp2.value()); t = float(self.sp3.value())
            props = rect_hollow(b, h, t)
        elif typ == "CircleSolid":
            d = float(self.sp1.value())
            props = circle_solid(d)
        elif typ == "CircleHollow":
            d = float(self.sp1.value()); t = float(self.sp2.value())
            props = circle_hollow(d, t)
        else:
            h = float(self.sp1.value()); bf = float(self.sp2.value()); tf = float(self.sp3.value()); tw = float(self.sp4.value())
            props = i_section(h, bf, tf, tw)

        uid = self._current_uid()
        if uid and uid in self.prj.sections and self.prj.sections[uid].name == name:
            s = self.prj.sections[uid]
            s.type = typ
            s.A, s.Iy, s.Iz, s.J, s.c_z, s.c_t, s.Zp = props.A, props.Iy, props.Iz, props.J, props.c_z, props.c_t, props.Zp
            s.p1, s.p2, s.p3, s.p4 = float(self.sp1.value()), float(self.sp2.value()), float(self.sp3.value()), float(self.sp4.value())
        else:
            s = Section(
                name=name,
                type=typ,
                A=props.A,
                Iy=props.Iy,
                Iz=props.Iz,
                J=props.J,
                c_z=props.c_z,
                c_t=props.c_t,
                Zp=props.Zp,
                p1=float(self.sp1.value()),
                p2=float(self.sp2.value()),
                p3=float(self.sp3.value()),
                p4=float(self.sp4.value()),
            )
            self.prj.sections[s.uid] = s
        self.refresh(selected_uid=s.uid)

    def save_library(self):
        try:
            save_section_library(self.prj.sections)
            QMessageBox.information(self, "Sections", "截面已保存到截面库（~/.minibeam/sections.json）。")
        except Exception as e:
            QMessageBox.critical(self, "Sections", f"保存截面库失败：{e}")

    def delete_section(self):
        uid = self._current_uid()
        if not uid:
            return
        if uid == self.prj.active_section_uid:
            QMessageBox.warning(self, "Delete", "不能删除 Active section。请先切换 Active。")
            return
        self.prj.sections.pop(uid, None)
        self.refresh()

    def set_active(self):
        uid = self._current_uid()
        if not uid:
            return
        self.prj.active_section_uid = uid
        self.refresh(selected_uid=uid)
