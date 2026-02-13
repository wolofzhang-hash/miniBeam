from __future__ import annotations
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFormLayout, QLineEdit,
    QDoubleSpinBox, QLabel, QMessageBox, QComboBox, QGroupBox, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QPolygonF

from ..core.model import Project, Material, Section
from ..core.section_props import rect_solid, circle_solid, i_section, rect_hollow, circle_hollow
from .i18n import LANG_ZH
from ..core.library_store import (
    save_material_library,
    save_section_library,
    load_builtin_material_library,
    load_material_library,
    load_section_library,
)


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
    def __init__(self, prj: Project, parent=None, lang: str = LANG_ZH):
        super().__init__(parent)
        self._is_zh = lang == LANG_ZH
        self._txt = lambda zh, en: zh if self._is_zh else en
        self.setWindowTitle(self._txt("材料", "Materials"))
        self.resize(900, 520)
        self.prj = prj
        self.model_materials = {uid: Material(**vars(mat)) for uid, mat in prj.materials.items()}
        self.library_materials = self._load_library_materials()
        self._library_edit_enabled = False

        lay = QHBoxLayout(self)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([self._txt("材料", "Materials")])
        lay.addWidget(self.tree, 1)

        right = QVBoxLayout()
        lay.addLayout(right, 2)

        form = QFormLayout()
        right.addLayout(form)

        self.ed_name = QLineEdit()
        self.ed_E = QDoubleSpinBox(); self.ed_E.setRange(0, 1e12); self.ed_E.setDecimals(0)
        self.ed_G = QDoubleSpinBox(); self.ed_G.setRange(0, 1e12); self.ed_G.setDecimals(0)
        class NuSpinBox(QDoubleSpinBox):
            def textFromValue(self, val: float) -> str:  # noqa: N802 (Qt API)
                v = float(val)
                if abs(v) < 1.0:
                    return f"{v:.2f}"
                return f"{v:.0f}"

        self.ed_nu = NuSpinBox(); self.ed_nu.setRange(0.0, 0.49); self.ed_nu.setDecimals(2)
        self.ed_fy = QDoubleSpinBox(); self.ed_fy.setRange(0, 1e6); self.ed_fy.setDecimals(0)

        form.addRow(self._txt("名称", "Name"), self.ed_name)
        form.addRow("E (MPa)", self.ed_E)
        form.addRow("G (MPa)", self.ed_G)
        form.addRow("nu", self.ed_nu)
        form.addRow("sigma_y (MPa)", self.ed_fy)

        self.lbl_scope_hint = QLabel("")
        self.lbl_scope_hint.setWordWrap(True)
        right.addWidget(self.lbl_scope_hint)

        btns = QHBoxLayout()
        right.addLayout(btns)
        self.btn_add = QPushButton(self._txt("新增", "Add"))
        self.btn_del = QPushButton(self._txt("删除", "Delete"))
        self.btn_model_to_library = QPushButton(self._txt("模型 -> 材料库", "Model -> Library"))
        self.btn_add_from_library = QPushButton(self._txt("材料库 -> 模型", "Library -> Model"))
        self.btn_edit_lib = QPushButton(self._txt("编辑材料库", "Edit Library"))
        self.btn_save_lib = QPushButton(self._txt("保存材料库", "Save Library"))
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_del)
        btns.addWidget(self.btn_model_to_library)
        btns.addWidget(self.btn_add_from_library)
        btns.addWidget(self.btn_edit_lib)
        btns.addWidget(self.btn_save_lib)
        btns.addStretch(1)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_add.clicked.connect(self.add_material)
        self.btn_model_to_library.clicked.connect(self.copy_selected_model_to_library)
        self.btn_add_from_library.clicked.connect(self.add_selected_library_to_model)
        self.btn_del.clicked.connect(self.delete_material)
        self.btn_edit_lib.clicked.connect(self.enable_library_edit)
        self.btn_save_lib.clicked.connect(self.save_library)
        self.tree.currentItemChanged.connect(self.load_selected)
        # live update
        for w in [self.ed_name, self.ed_E, self.ed_G, self.ed_nu, self.ed_fy]:
            if hasattr(w, "textChanged"):
                w.textChanged.connect(self.update_selected)
            else:
                w.valueChanged.connect(self.update_selected)

        self._loading_selection = False
        self.refresh()

    def _load_library_materials(self) -> dict[str, Material]:
        mats: dict[str, Material] = {}
        by_name: set[str] = set()
        for src in (load_builtin_material_library(), load_material_library()):
            for m in src:
                if m.name in by_name:
                    continue
                mats[m.uid] = m
                by_name.add(m.name)
        return mats

    def _current_selection(self) -> tuple[str, str] | tuple[None, None]:
        it = self.tree.currentItem()
        if it is None:
            return (None, None)
        data = it.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return (None, None)
        return data

    def _selected_material(self) -> Material | None:
        scope, uid = self._current_selection()
        if scope == "model":
            return self.model_materials.get(uid)
        if scope == "library":
            return self.library_materials.get(uid)
        return None

    def _sync_current_item_label(self, scope: str, uid: str, name: str):
        item = self.tree.currentItem()
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data != (scope, uid):
            return
        item.setText(0, f"{name} [{uid}]")

    def _set_editor_enabled(self, enabled: bool):
        for w in [self.ed_name, self.ed_E, self.ed_G, self.ed_nu, self.ed_fy]:
            w.setEnabled(enabled)

    def refresh(self, selected: tuple[str, str] | None = None):
        if selected is None:
            selected = self._current_selection()
        self.tree.clear()

        root_model = QTreeWidgetItem([self._txt("当前模型", "Current Model")])
        root_model.setData(0, Qt.ItemDataRole.UserRole, None)
        self.tree.addTopLevelItem(root_model)
        for mat in self.model_materials.values():
            item = QTreeWidgetItem([f"{mat.name} [{mat.uid}]"])
            item.setData(0, Qt.ItemDataRole.UserRole, ("model", mat.uid))
            root_model.addChild(item)

        root_lib = QTreeWidgetItem([self._txt("材料库", "Library")])
        root_lib.setData(0, Qt.ItemDataRole.UserRole, None)
        self.tree.addTopLevelItem(root_lib)
        for mat in self.library_materials.values():
            item = QTreeWidgetItem([f"{mat.name} [{mat.uid}]"])
            item.setData(0, Qt.ItemDataRole.UserRole, ("library", mat.uid))
            root_lib.addChild(item)

        root_model.setExpanded(True)
        root_lib.setExpanded(True)

        if selected and all(selected):
            for i in range(self.tree.topLevelItemCount()):
                root = self.tree.topLevelItem(i)
                for c in range(root.childCount()):
                    child = root.child(c)
                    if child.data(0, Qt.ItemDataRole.UserRole) == selected:
                        self.tree.setCurrentItem(child)
                        return

    def load_selected(self, current, _previous):
        _ = current
        mat = self._selected_material()
        if mat is None:
            self._set_editor_enabled(False)
            self.btn_model_to_library.setEnabled(False)
            self.btn_add_from_library.setEnabled(False)
            return

        self._loading_selection = True
        self.ed_name.setText(mat.name)
        self.ed_E.setValue(mat.E)
        self.ed_G.setValue(mat.G)
        self.ed_nu.setValue(mat.nu)
        self.ed_fy.setValue(mat.sigma_y)
        self._loading_selection = False

        scope, _uid = self._current_selection()
        editable = (scope == "model") or (scope == "library" and self._library_edit_enabled)
        self._set_editor_enabled(editable)
        self.btn_model_to_library.setEnabled(scope == "model")
        self.btn_add_from_library.setEnabled(scope == "library")
        if scope == "library" and not self._library_edit_enabled:
            self.lbl_scope_hint.setText(self._txt("材料库默认只读。请使用“材料库 -> 模型”复制到当前模型后再修改。", "Library is read-only by default. Use 'Library -> Model' before editing."))
        elif scope == "library":
            self.lbl_scope_hint.setText(self._txt("⚠ 当前正在直接编辑材料库，可能影响后续所有新模型，请谨慎操作。", "⚠ You are editing the library directly; this may affect future models."))
        else:
            self.lbl_scope_hint.setText("正在编辑当前模型中的材料。")

    def update_selected(self, *_):
        if self._loading_selection:
            return
        scope, uid = self._current_selection()
        if scope == "model":
            m = self.model_materials.get(uid)
        elif scope == "library" and self._library_edit_enabled:
            m = self.library_materials.get(uid)
        else:
            return
        if m is None:
            return

        m.name = self.ed_name.text().strip() or m.name
        m.E = float(self.ed_E.value())
        m.G = float(self.ed_G.value())
        m.nu = float(self.ed_nu.value())
        m.sigma_y = float(self.ed_fy.value())
        # Avoid rebuilding the tree while user is typing. Rebuild triggers
        # currentItemChanged -> load_selected(), which steals editor focus and
        # causes the cursor to "jump" after each keystroke.
        self._sync_current_item_label(scope, uid, m.name)

    def add_material(self):
        m = Material(name=self._txt("新材料", "Material"), E=210000.0, G=81000.0, nu=0.3, rho=7.85e-6, sigma_y=355.0)
        self.model_materials[m.uid] = m
        self.refresh(selected=("model", m.uid))

    def add_selected_library_to_model(self):
        scope, uid = self._current_selection()
        if scope != "library" or uid not in self.library_materials:
            return
        src = self.library_materials[uid]
        same_uid = self._find_same_material(self.model_materials, src)
        if same_uid:
            QMessageBox.warning(self, self._txt("材料", "Materials"), self._txt("模型中已存在属性相同的材料，无需重复添加。", "The same material already exists in the model."))
            self.refresh(selected=("model", same_uid))
            return
        m = Material(name=src.name, E=src.E, G=src.G, nu=src.nu, rho=src.rho, sigma_y=src.sigma_y)
        self.model_materials[m.uid] = m
        self.refresh(selected=("model", m.uid))

    def copy_selected_model_to_library(self):
        scope, uid = self._current_selection()
        if scope != "model" or uid not in self.model_materials:
            return

        src = self.model_materials[uid]
        same_uid = self._find_same_material(self.library_materials, src)
        if same_uid:
            QMessageBox.warning(self, self._txt("材料", "Materials"), self._txt("材料库中已存在属性相同的材料，无需重复添加。", "The same material already exists in the library."))
            self.refresh(selected=("library", same_uid))
            return

        lib_item = Material(name=src.name, E=src.E, G=src.G, nu=src.nu, rho=src.rho, sigma_y=src.sigma_y)
        self.library_materials[lib_item.uid] = lib_item
        self.refresh(selected=("library", lib_item.uid))

    def _find_same_material(self, mats: dict[str, Material], target: Material) -> str | None:
        for uid, mat in mats.items():
            if (
                mat.name == target.name
                and mat.E == target.E
                and mat.G == target.G
                and mat.nu == target.nu
                and mat.rho == target.rho
                and mat.sigma_y == target.sigma_y
            ):
                return uid
        return None

    def delete_material(self):
        scope, uid = self._current_selection()
        if not uid:
            return
        if scope == "model":
            self.model_materials.pop(uid, None)
        elif scope == "library":
            if not self._library_edit_enabled:
                QMessageBox.warning(self, self._txt("材料", "Materials"), self._txt("材料库默认不可删除。请先启用高风险编辑模式。", "Library deletion is locked. Enable high-risk edit mode first."))
                return
            self.library_materials.pop(uid, None)
        self.refresh()

    def accept(self):
        self.prj.materials = {uid: Material(**vars(mat)) for uid, mat in self.model_materials.items()}
        super().accept()

    def enable_library_edit(self):
        if self._library_edit_enabled:
            return
        reply = QMessageBox.warning(
            self,
            self._txt("风险提示", "Risk Warning"),
            self._txt("直接修改材料库会影响所有后续新建模型，存在风险。是否继续启用材料库编辑？", "Directly editing the library impacts future models. Continue?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._library_edit_enabled = True
            self.load_selected(self.tree.currentItem(), None)

    def save_library(self):
        if not self._library_edit_enabled:
            QMessageBox.warning(self, self._txt("材料", "Materials"), self._txt("材料库编辑未启用。请先点击“编辑材料库”。", "Library editing is not enabled. Click 'Edit Library' first."))
            return
        try:
            save_material_library(self.library_materials)
            QMessageBox.information(self, self._txt("材料", "Materials"), self._txt("材料库已保存（~/.minibeam/materials.json）。", "Material library saved (~/.minibeam/materials.json)."))
        except Exception as e:
            QMessageBox.critical(self, self._txt("材料", "Materials"), self._txt(f"保存材料库失败：{e}", f"Failed to save material library: {e}"))


class SectionManagerDialog(QDialog):
    def __init__(self, prj: Project, parent=None, lang: str = LANG_ZH):
        super().__init__(parent)
        self._is_zh = lang == LANG_ZH
        self._txt = lambda zh, en: zh if self._is_zh else en
        self.setWindowTitle(self._txt("截面", "Sections"))
        self.resize(900, 520)
        self.prj = prj
        self.model_sections = {uid: Section(**vars(sec)) for uid, sec in prj.sections.items()}
        self.library_sections = self._load_library_sections()
        self._library_edit_enabled = False

        lay = QHBoxLayout(self)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([self._txt("截面", "Sections")])
        lay.addWidget(self.tree, 1)

        right = QVBoxLayout()
        lay.addLayout(right, 2)

        # wizard
        gb = QGroupBox(self._txt("截面向导", "Section Wizard"))
        right.addWidget(gb)
        self.form = QFormLayout(gb)

        self.cmb_type = QComboBox()
        self.cmb_type.addItems(["RectSolid", "RectHollow", "CircleSolid", "CircleHollow", "ISection"])
        self.ed_name = QLineEdit()

        self.sp1 = QDoubleSpinBox(); self.sp1.setRange(0.001, 1e9); self.sp1.setDecimals(1)
        self.sp2 = QDoubleSpinBox(); self.sp2.setRange(0.001, 1e9); self.sp2.setDecimals(1)
        self.sp3 = QDoubleSpinBox(); self.sp3.setRange(0.001, 1e9); self.sp3.setDecimals(1)
        self.sp4 = QDoubleSpinBox(); self.sp4.setRange(0.001, 1e9); self.sp4.setDecimals(1)
        self.sp5 = QDoubleSpinBox(); self.sp5.setRange(0.001, 1e9); self.sp5.setDecimals(1)

        self.lbl_p1 = QLabel("参数1")
        self.lbl_p2 = QLabel("参数2")
        self.lbl_p3 = QLabel("参数3")
        self.lbl_p4 = QLabel("参数4")
        self.lbl_p5 = QLabel("参数5")

        self.form.addRow(self._txt("类型", "Type"), self.cmb_type)
        self.form.addRow(self._txt("名称", "Name"), self.ed_name)
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
        self.lbl_props = QLabel("A= -   Iy= -   Iz= -   J= -   Zpy= -   Zpz= -   ky= -   kz= -   kt= -")
        self.lbl_props.setWordWrap(True)
        right.addWidget(self.lbl_props)

        btns = QHBoxLayout()
        right.addLayout(btns)
        self.btn_add = QPushButton(self._txt("新增/更新", "Add/Update"))
        self.btn_del = QPushButton(self._txt("删除", "Delete"))
        self.btn_model_to_library = QPushButton(self._txt("模型 -> 截面库", "Model -> Library"))
        self.btn_add_from_library = QPushButton(self._txt("截面库 -> 模型", "Library -> Model"))
        self.btn_edit_lib = QPushButton(self._txt("编辑截面库", "Edit Library"))
        self.btn_save_lib = QPushButton(self._txt("保存截面库", "Save Library"))
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_del)
        btns.addWidget(self.btn_model_to_library)
        btns.addWidget(self.btn_add_from_library)
        btns.addWidget(self.btn_edit_lib)
        btns.addWidget(self.btn_save_lib)
        btns.addStretch(1)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)

        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_add.clicked.connect(self.add_or_update)
        self.btn_model_to_library.clicked.connect(self.copy_selected_model_to_library)
        self.btn_add_from_library.clicked.connect(self.add_selected_library_to_model)
        self.btn_del.clicked.connect(self.delete_section)
        self.btn_edit_lib.clicked.connect(self.enable_library_edit)
        self.cmb_type.currentTextChanged.connect(self._update_hint)
        self.tree.currentItemChanged.connect(self._load_selected)
        self.btn_save_lib.clicked.connect(self.save_library)
        # live preview
        for sp in [self.sp1, self.sp2, self.sp3, self.sp4, self.sp5]:
            sp.valueChanged.connect(self._update_preview)
        self.cmb_type.currentTextChanged.connect(lambda _: self._update_preview())

        self._loading_selection = False
        self._update_hint(self.cmb_type.currentText())
        self.refresh()

    def _load_library_sections(self) -> dict[str, Section]:
        sections: dict[str, Section] = {}
        by_name: set[str] = set()
        for sec in load_section_library():
            if sec.name in by_name:
                continue
            sections[sec.uid] = sec
            by_name.add(sec.name)
        return sections

    def _current_selection(self) -> tuple[str, str] | tuple[None, None]:
        it = self.tree.currentItem()
        if it is None:
            return (None, None)
        data = it.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return (None, None)
        return data

    def _selected_section(self) -> Section | None:
        scope, uid = self._current_selection()
        if scope == "model":
            return self.model_sections.get(uid)
        if scope == "library":
            return self.library_sections.get(uid)
        return None

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
            self.lbl_props.setText(f"A={props.A:.0f} mm²   Iy={props.Iy:.0f} mm⁴   Iz={props.Iz:.0f} mm⁴   J={props.J:.0f} mm⁴   Zpy={props.Zp_y:.0f} mm³   Zpz={props.Zp_z:.0f} mm³   ky={props.shape_factor_y:.3f}   kz={props.shape_factor_z:.3f}   kt={props.shape_factor_t:.3f}")
        except Exception:
            self.lbl_props.setText("A= -   Iy= -   Iz= -   J= -   Zpy= -   Zpz= -   ky= -   kz= -   kt= -")

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

        if self._loading_selection:
            return

        if typ == "RectSolid":
            self.lbl_hint.setText(self._txt("实心矩形：参数1=b(mm)，参数2=h(mm)。参数3/4忽略。", "RectSolid: Param1=b(mm), Param2=h(mm). Param3/4 ignored."))
            self.ed_name.setText("Rect100x10")
            self.sp1.setValue(100.0); self.sp2.setValue(10.0)
        elif typ == "RectHollow":
            self.lbl_hint.setText(self._txt("空心矩形：参数1=b(mm)，参数2=h(mm)，参数3=t(mm)。参数4忽略。", "RectHollow: Param1=b(mm), Param2=h(mm), Param3=t(mm). Param4 ignored."))
            self.ed_name.setText("RectTube100x50x5")
            self.sp1.setValue(100.0); self.sp2.setValue(50.0); self.sp3.setValue(5.0); self.sp4.setValue(0.0)
        elif typ == "CircleSolid":
            self.lbl_hint.setText(self._txt("实心圆：参数1=d(mm)。其余忽略。", "CircleSolid: Param1=d(mm). Others ignored."))
            self.ed_name.setText("Circle20")
            self.sp1.setValue(20.0); self.sp2.setValue(0.0); self.sp3.setValue(0.0); self.sp4.setValue(0.0)
        elif typ == "CircleHollow":
            self.lbl_hint.setText(self._txt("空心圆：参数1=D(mm)，参数2=t(mm)。其余忽略。", "CircleHollow: Param1=D(mm), Param2=t(mm). Others ignored."))
            self.ed_name.setText("Pipe60x4")
            self.sp1.setValue(60.0); self.sp2.setValue(4.0); self.sp3.setValue(0.0); self.sp4.setValue(0.0)
        else:
            self.lbl_hint.setText(self._txt("工字形：参数1=h(mm)，参数2=bf(mm)，参数3=tf(mm)，参数4=tw(mm)。", "ISection: Param1=h(mm), Param2=bf(mm), Param3=tf(mm), Param4=tw(mm)"))
            self.ed_name.setText("I200")
            self.sp1.setValue(200.0); self.sp2.setValue(100.0); self.sp3.setValue(10.0); self.sp4.setValue(6.0)

    def refresh(self, selected: tuple[str, str] | None = None):
        if selected is None:
            selected = self._current_selection()
        self.tree.clear()

        root_model = QTreeWidgetItem([self._txt("当前模型", "Current Model")])
        root_model.setData(0, Qt.ItemDataRole.UserRole, None)
        self.tree.addTopLevelItem(root_model)
        for sec in self.model_sections.values():
            it = QTreeWidgetItem([f"{sec.name} ({sec.type}) [{sec.uid}]"])
            it.setData(0, Qt.ItemDataRole.UserRole, ("model", sec.uid))
            root_model.addChild(it)

        root_lib = QTreeWidgetItem([self._txt("截面库", "Library")])
        root_lib.setData(0, Qt.ItemDataRole.UserRole, None)
        self.tree.addTopLevelItem(root_lib)
        for sec in self.library_sections.values():
            it = QTreeWidgetItem([f"{sec.name} ({sec.type}) [{sec.uid}]"])
            it.setData(0, Qt.ItemDataRole.UserRole, ("library", sec.uid))
            root_lib.addChild(it)

        root_model.setExpanded(True)
        root_lib.setExpanded(True)

        if selected and all(selected):
            for i in range(self.tree.topLevelItemCount()):
                root = self.tree.topLevelItem(i)
                for c in range(root.childCount()):
                    child = root.child(c)
                    if child.data(0, Qt.ItemDataRole.UserRole) == selected:
                        self.tree.setCurrentItem(child)
                        return

    def _load_selected(self, current, _previous):
        _ = current
        s = self._selected_section()
        if s is None:
            self.btn_model_to_library.setEnabled(False)
            self.btn_add_from_library.setEnabled(False)
            return

        self._loading_selection = True
        self.cmb_type.setCurrentText(s.type)
        self.ed_name.setText(s.name)
        self.sp1.setValue(float(s.p1))
        self.sp2.setValue(float(s.p2))
        self.sp3.setValue(float(s.p3))
        self.sp4.setValue(float(s.p4))
        self._loading_selection = False
        self._update_preview()

        scope, _uid = self._current_selection()
        editable = (scope == "model") or (scope == "library" and self._library_edit_enabled)
        for w in [self.cmb_type, self.ed_name, self.sp1, self.sp2, self.sp3, self.sp4, self.sp5]:
            w.setEnabled(editable)
        self.btn_model_to_library.setEnabled(scope == "model")
        self.btn_add_from_library.setEnabled(scope == "library")
        if scope == "library" and not self._library_edit_enabled:
            self.lbl_hint.setText(self._txt("截面库默认只读。请使用“截面库 -> 模型”复制后再编辑。", "Section library is read-only by default. Use 'Library -> Model' before editing."))
        elif scope == "library":
            self.lbl_hint.setText(self._txt("⚠ 当前正在直接编辑截面库，可能影响后续所有模型。", "⚠ You are editing the section library directly; this may affect future models."))

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

        scope, uid = self._current_selection()
        target_scope = "model"
        target_sections = self.model_sections
        if scope == "library" and self._library_edit_enabled:
            target_scope = "library"
            target_sections = self.library_sections

        if uid and uid in target_sections:
            s = target_sections[uid]
            s.type = typ
            s.name = name
            s.A, s.Iy, s.Iz, s.J = props.A, props.Iy, props.Iz, props.J
            s.c_y, s.c_z = props.c_y, props.c_z
            s.Zp_y, s.Zp_z = props.Zp_y, props.Zp_z
            s.shape_factor_y, s.shape_factor_z, s.shape_factor_t = props.shape_factor_y, props.shape_factor_z, props.shape_factor_t
            s.Zp, s.shape_factor = props.Zp_z, props.shape_factor_z
            s.p1, s.p2, s.p3, s.p4 = float(self.sp1.value()), float(self.sp2.value()), float(self.sp3.value()), float(self.sp4.value())
        else:
            s = Section(
                name=name,
                type=typ,
                A=props.A,
                Iy=props.Iy,
                Iz=props.Iz,
                J=props.J,
                c_y=props.c_y,
                c_z=props.c_z,
                Zp_y=props.Zp_y,
                Zp_z=props.Zp_z,
                shape_factor_y=props.shape_factor_y,
                shape_factor_z=props.shape_factor_z,
                shape_factor_t=props.shape_factor_t,
                Zp=props.Zp_z,
                shape_factor=props.shape_factor_z,
                p1=float(self.sp1.value()),
                p2=float(self.sp2.value()),
                p3=float(self.sp3.value()),
                p4=float(self.sp4.value()),
            )
            target_sections[s.uid] = s
        self.refresh(selected=(target_scope, s.uid))

    def add_selected_library_to_model(self):
        scope, uid = self._current_selection()
        if scope != "library" or uid not in self.library_sections:
            return
        src = self.library_sections[uid]
        same_uid = self._find_same_section(self.model_sections, src)
        if same_uid:
            QMessageBox.warning(self, self._txt("截面", "Sections"), self._txt("模型中已存在属性相同的截面，无需重复添加。", "The same section already exists in the model."))
            self.refresh(selected=("model", same_uid))
            return
        sec = Section(
            name=src.name,
            type=src.type,
            A=src.A,
            Iy=src.Iy,
            Iz=src.Iz,
            J=src.J,
            c_y=getattr(src, "c_y", src.c_z),
            c_z=src.c_z,
            Zp_y=getattr(src, "Zp_y", src.Zp),
            Zp_z=getattr(src, "Zp_z", src.Zp),
            shape_factor_y=getattr(src, "shape_factor_y", src.shape_factor),
            shape_factor_z=getattr(src, "shape_factor_z", src.shape_factor),
            shape_factor_t=getattr(src, "shape_factor_t", 1.0),
            Zp=getattr(src, "Zp_z", src.Zp),
            shape_factor=getattr(src, "shape_factor_z", src.shape_factor),
            p1=src.p1,
            p2=src.p2,
            p3=src.p3,
            p4=src.p4,
        )
        self.model_sections[sec.uid] = sec
        self.refresh(selected=("model", sec.uid))

    def copy_selected_model_to_library(self):
        scope, uid = self._current_selection()
        if scope != "model" or uid not in self.model_sections:
            return

        src = self.model_sections[uid]
        same_uid = self._find_same_section(self.library_sections, src)
        if same_uid:
            QMessageBox.warning(self, self._txt("截面", "Sections"), self._txt("截面库中已存在属性相同的截面，无需重复添加。", "The same section already exists in the library."))
            self.refresh(selected=("library", same_uid))
            return

        sec = Section(
            name=src.name,
            type=src.type,
            A=src.A,
            Iy=src.Iy,
            Iz=src.Iz,
            J=src.J,
            c_y=getattr(src, "c_y", src.c_z),
            c_z=src.c_z,
            Zp_y=getattr(src, "Zp_y", src.Zp),
            Zp_z=getattr(src, "Zp_z", src.Zp),
            shape_factor_y=getattr(src, "shape_factor_y", src.shape_factor),
            shape_factor_z=getattr(src, "shape_factor_z", src.shape_factor),
            shape_factor_t=getattr(src, "shape_factor_t", 1.0),
            Zp=getattr(src, "Zp_z", src.Zp),
            shape_factor=getattr(src, "shape_factor_z", src.shape_factor),
            p1=src.p1,
            p2=src.p2,
            p3=src.p3,
            p4=src.p4,
        )
        self.library_sections[sec.uid] = sec
        self.refresh(selected=("library", sec.uid))

    def _find_same_section(self, sections: dict[str, Section], target: Section) -> str | None:
        for uid, sec in sections.items():
            if (
                sec.name == target.name
                and sec.type == target.type
                and sec.p1 == target.p1
                and sec.p2 == target.p2
                and sec.p3 == target.p3
                and sec.p4 == target.p4
            ):
                return uid
        return None

    def enable_library_edit(self):
        if self._library_edit_enabled:
            return
        reply = QMessageBox.warning(
            self,
            self._txt("风险提示", "Risk Warning"),
            self._txt("直接修改截面库会影响所有后续新建模型，存在风险。是否继续启用编辑？", "Directly editing the section library impacts future models. Continue?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._library_edit_enabled = True
            self._load_selected(self.tree.currentItem(), None)

    def save_library(self):
        if not self._library_edit_enabled:
            QMessageBox.warning(self, self._txt("截面", "Sections"), self._txt("截面库编辑未启用。请先点击“编辑截面库”。", "Library editing is not enabled. Click 'Edit Library' first."))
            return
        try:
            save_section_library(self.library_sections)
            QMessageBox.information(self, self._txt("截面", "Sections"), self._txt("截面库已保存（~/.minibeam/sections.json）。", "Section library saved (~/.minibeam/sections.json)."))
        except Exception as e:
            QMessageBox.critical(self, self._txt("截面", "Sections"), self._txt(f"保存截面库失败：{e}", f"Failed to save section library: {e}"))

    def delete_section(self):
        scope, uid = self._current_selection()
        if not uid:
            return
        if scope == "model":
            self.model_sections.pop(uid, None)
        elif scope == "library":
            if not self._library_edit_enabled:
                QMessageBox.warning(self, self._txt("截面", "Sections"), self._txt("截面库默认不可删除，请先启用高风险编辑模式。", "Library deletion is locked. Enable high-risk edit mode first."))
                return
            self.library_sections.pop(uid, None)
        self.refresh()

    def accept(self):
        scope, uid = self._current_selection()
        if uid and scope in {"model", "library"}:
            try:
                self.add_or_update()
            except Exception as e:
                QMessageBox.warning(self, self._txt("截面", "Sections"), self._txt(f"参数无效，无法保存：{e}", f"Invalid section parameters: {e}"))
                return
        self.prj.sections = {uid: Section(**vars(sec)) for uid, sec in self.model_sections.items()}
        super().accept()
