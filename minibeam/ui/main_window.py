from __future__ import annotations
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
    QLabel, QMessageBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QStackedWidget, QInputDialog
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QPixmap
from PyQt6.QtWidgets import QStyle, QFileDialog

import sys
import traceback
import json
import numpy as np

from ..core.model import Project, Material, Section, Constraint, NodalLoad
from ..core.section_props import rect_solid, circle_solid, i_section
from ..core.validation import validate_project
from ..core.pynite_adapter import solve_with_pynite, PyniteSolverError, SolveOutput

from .canvas_view import BeamCanvas
from .dialogs import MaterialManagerDialog, SectionManagerDialog
from .results_view import ResultsView
from ..core.undo import UndoStack
from ..core.library_store import load_material_library, load_section_library, merge_by_name


class MainWindow(QMainWindow):
    def _std_icon(self, primary: str, fallback: str):
        """Return a Qt standard icon with a safe fallback.

        PyQt6's QStyle.StandardPixmap enum can differ across Qt builds.
        Some members (e.g. SP_ArrowCursor) may not exist on certain
        installations. This helper keeps the app from crashing at startup.
        """
        sp = getattr(QStyle.StandardPixmap, primary, None)
        if sp is None:
            sp = getattr(QStyle.StandardPixmap, fallback)
        return self.style().standardIcon(sp)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiniBeam v0.2.0 (Phase-1)")
        self.resize(1400, 850)

        # ---------------- Crash resilience ----------------
        # Hard Qt C++ crashes (0xC0000409) are not catchable from Python.
        # But all Python exceptions in slots/callbacks should be shown in a
        # dialog so you can report them and we can fix quickly.
        def _excepthook(exc_type, exc, tb):
            try:
                msg = "".join(traceback.format_exception(exc_type, exc, tb))
                QMessageBox.critical(None, "MiniBeam Error", msg)
            finally:
                # Also print to stderr for IDE consoles
                sys.__excepthook__(exc_type, exc, tb)

        sys.excepthook = _excepthook

        # Coalesce expensive redraw operations. We always defer redraw/rebuild
        # to the next Qt event loop tick to prevent crashes caused by clearing
        # a QGraphicsScene in the middle of a mouse event (e.g. drag or double
        # click).
        self._refresh_pending = False
        self._rebuild_pending = False
        self._reselect_point_uid: str | None = None

        self.project = Project()
        self.undo_stack = UndoStack()
        self._last_model_mode: str = "add_point"  # Phase-1: only point modeling
        # seed one material & one section
        mat = Material(name="Steel", E=210000.0, G=81000.0, nu=0.3, rho=7.85e-6, sigma_y=355.0)
        self.project.materials[mat.uid] = mat
        self.project.active_material_uid = mat.uid

        sp = rect_solid(100.0, 10.0)
        sec = Section(name="Rect100x10", type="RectSolid", A=sp.A, Iy=sp.Iy, Iz=sp.Iz, J=sp.J, c_z=sp.c_z)
        self.project.sections[sec.uid] = sec
        self.project.active_section_uid = sec.uid

        # Load user libraries (materials/sections) and merge by name.
        # This lets you build a reusable library across projects.
        try:
            merge_by_name(self.project.materials, load_material_library(), name_attr="name")
            merge_by_name(self.project.sections, load_section_library(), name_attr="name")
        except Exception:
            pass

        self._build_ui()
        self._connect()
        self._schedule_refresh()

        self.last_results: SolveOutput | None = None

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        
        # ---------------- Ribbon (Word-like tabs with large icons) ----------------
        icon_size = QSize(48, 48)

        # Actions (wired in _connect)
        self.act_new = QAction(self._std_icon("SP_FileIcon", "SP_DirIcon"), "New", self)
        self.act_open = QAction(self._std_icon("SP_DialogOpenButton", "SP_DirOpenIcon"), "Open", self)
        self.act_save = QAction(self._std_icon("SP_DialogSaveButton", "SP_DriveHDIcon"), "Save", self)

        self.act_select = QAction(self._std_icon("SP_ArrowCursor", "SP_ArrowForward"), "Select", self)
        self.act_add_point = QAction(self._std_icon("SP_FileDialogNewFolder", "SP_DirIcon"), "Add Point", self)
        self.act_auto_members = QAction(self._std_icon("SP_BrowserReload", "SP_BrowserStop"), "Auto Members", self)
        self.act_delete = QAction(self._std_icon("SP_TrashIcon", "SP_DialogCloseButton"), "Delete", self)

        self.act_materials = QAction(self._std_icon("SP_DriveFDIcon", "SP_DriveHDIcon"), "Materials…", self)
        self.act_sections = QAction(self._std_icon("SP_DriveDVDIcon", "SP_DriveHDIcon"), "Sections…", self)
        self.act_assign_mat = QAction(self._std_icon("SP_ArrowDown", "SP_ArrowForward"), "Assign Material", self)
        self.act_assign_sec = QAction(self._std_icon("SP_ArrowDown", "SP_ArrowForward"), "Assign Section", self)

        self.act_add_dx = QAction(self._std_icon("SP_DialogYesButton", "SP_DialogApplyButton"), "Constraint", self)
        self.act_add_dy = QAction(self._std_icon("SP_DialogYesButton", "SP_DialogApplyButton"), "Add DY", self)
        self.act_add_rz = QAction(self._std_icon("SP_DialogYesButton", "SP_DialogApplyButton"), "Add RZ", self)

        self.act_add_fy = QAction(self._std_icon("SP_ArrowUp", "SP_ArrowForward"), "Load", self)
        self.act_add_mz = QAction(self._std_icon("SP_ArrowUp", "SP_ArrowForward"), "Add MZ", self)
        self.act_add_udl = QAction(self._std_icon("SP_ArrowDown", "SP_ArrowForward"), "UDL", self)

        # Background
        self.act_bg_import = QAction(self._std_icon("SP_DialogOpenButton", "SP_DirOpenIcon"), "Import", self)
        self.act_bg_calibrate = QAction(self._std_icon("SP_BrowserReload", "SP_DialogResetButton"), "Calibrate", self)
        self.act_bg_opacity = QAction(self._std_icon("SP_DialogApplyButton", "SP_DialogOkButton"), "Opacity", self)
        self.act_bg_bw = QAction(self._std_icon("SP_FileDialogDetailedView", "SP_FileDialogListView"), "B/W", self)
        self.act_bg_bw.setCheckable(True)
        self.act_bg_clear = QAction(self._std_icon("SP_TrashIcon", "SP_DialogCancelButton"), "Clear", self)

        self.act_validate = QAction(self._std_icon("SP_DialogApplyButton", "SP_DialogOkButton"), "Validate", self)
        self.act_solve = QAction(self._std_icon("SP_MediaPlay", "SP_DialogOkButton"), "Solve", self)

        self.act_show_results = QAction(self._std_icon("SP_ComputerIcon", "SP_DesktopIcon"), "Results", self)
        self.act_back_to_model = QAction(self._std_icon("SP_ArrowBack", "SP_ArrowLeft"), "Back", self)
        self.act_export_csv = QAction(self._std_icon("SP_DialogSaveButton", "SP_DriveHDIcon"), "Export CSV", self)

        # Shortcuts
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_new.setShortcut(QKeySequence.StandardKey.New)

        # Ribbon widget
        from PyQt6.QtWidgets import QTabWidget, QToolButton, QGridLayout, QSizePolicy

        self.ribbon = QTabWidget()
        self.ribbon.setDocumentMode(True)
        root.addWidget(self.ribbon, 0)

        def mk_group(title: str, items: list[object]) -> QWidget:
            gb = QGroupBox(title)
            lay = QGridLayout(gb)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setHorizontalSpacing(8)
            lay.setVerticalSpacing(8)
            col = 0
            for it in items:
                if isinstance(it, QAction):
                    btn = QToolButton()
                    btn.setDefaultAction(it)
                    btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
                    btn.setIconSize(icon_size)
                    btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
                    lay.addWidget(btn, 0, col)
                    col += 1
                elif isinstance(it, QWidget):
                    lay.addWidget(it, 0, col)
                    col += 1
                else:
                    # ignore unknown item types
                    pass
            return gb

        def mk_tab(name: str, groups: list[QWidget]) -> QWidget:
            w = QWidget()
            hl = QHBoxLayout(w)
            hl.setContentsMargins(6, 6, 6, 6)
            hl.setSpacing(10)
            for g in groups:
                hl.addWidget(g)
            hl.addStretch(1)
            self.ribbon.addTab(w, name)
            return w

        mk_tab("Home", [
            mk_group("File", [self.act_new, self.act_open, self.act_save]),
            mk_group("Model", [self.act_select, self.act_add_point, self.act_auto_members, self.act_delete]),
        ])

        mk_tab("Properties", [
            mk_group("Libraries", [self.act_materials, self.act_sections]),
            mk_group("Assign", [self.act_assign_mat, self.act_assign_sec]),
        ])

        mk_tab("Constraints", [
            mk_group("Constraints", [self.act_add_dx]),
        ])

        mk_tab("Loads", [
            mk_group("Loads", [self.act_add_fy, self.act_add_udl]),
        ])

        mk_tab("Background", [
            mk_group("Background", [self.act_bg_import, self.act_bg_calibrate, self.act_bg_opacity, self.act_bg_bw, self.act_bg_clear]),
        ])

        mk_tab("Solve", [
            mk_group("Solve", [self.act_validate, self.act_solve]),
        ])

        # Results: dropdown + plot
        from PyQt6.QtWidgets import QComboBox
        self.cmb_result_type = QComboBox()
        self.cmb_result_type.addItems([
            "Deflection",
            "FBD",
            "Shear V",
            "Moment M",
            "Stress σ",
            "Margin MS",
        ])
        self.cmb_result_type.setMinimumWidth(160)
        self.cmb_result_type.setToolTip("Select result type")

        mk_tab("Results", [
            mk_group("Results", [self.cmb_result_type, self.act_show_results, self.act_export_csv, self.act_back_to_model]),
        ])

        # main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # left tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Objects"])
        splitter.addWidget(self.tree)
        self.tree.setMinimumWidth(240)

        # center: canvas + results (stack)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)
        self.canvas = BeamCanvas()
        self.results_view = ResultsView()
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self.canvas)
        self.center_stack.addWidget(self.results_view)
        center_layout.addWidget(self.center_stack, 1)
        splitter.addWidget(center)

        # right: property panel
        self.prop = QWidget()
        self.prop.setMinimumWidth(320)
        splitter.addWidget(self.prop)
        pr = QVBoxLayout(self.prop)
        pr.setContentsMargins(6, 6, 6, 6)
        pr.setSpacing(8)

        self.lbl_sel = QLabel("Selection: (none)")
        self.lbl_sel.setWordWrap(True)
        pr.addWidget(self.lbl_sel)

        gb = QGroupBox("Point Properties")
        pr.addWidget(gb)
        form = QFormLayout(gb)
        self.ed_x = QDoubleSpinBox()
        self.ed_x.setRange(-1e9, 1e9)
        self.ed_x.setDecimals(6)
        # Commit on Enter (editingFinished) instead of moving geometry on every
        # keystroke while typing.
        self.ed_x.setKeyboardTracking(False)
        self.ed_x.setEnabled(False)
        form.addRow("X (mm)", self.ed_x)

        self.lbl_len = QLabel("-")
        form.addRow("Member length", self.lbl_len)

        pr.addStretch(1)

        splitter.setStretchFactor(1, 1)

    def _connect(self):
        # --- File ---
        self.act_new.triggered.connect(self.new_project)
        self.act_open.triggered.connect(self.open_project)
        self.act_save.triggered.connect(self.save_project)

        # --- Property panel ---
        # QDoubleSpinBox with keyboardTracking(False) only emits editingFinished
        # when the user presses Enter or the control loses focus.
        self.ed_x.editingFinished.connect(lambda: self._edit_selected_point_x(float(self.ed_x.value())))

        # --- Model ---
        self.act_select.triggered.connect(lambda: self.set_model_mode("select"))
        self.act_add_point.triggered.connect(lambda: self.set_model_mode("add_point"))
        self.act_auto_members.setCheckable(True)
        self.act_auto_members.setChecked(bool(self.project.auto_members))
        self.act_auto_members.toggled.connect(self._toggle_auto_members)
        self.act_delete.triggered.connect(self.delete_selected_points)

        # --- Libraries ---
        self.act_materials.triggered.connect(self.open_materials)
        self.act_sections.triggered.connect(self.open_sections)
        self.act_assign_mat.triggered.connect(self.assign_material_to_selected_members)
        self.act_assign_sec.triggered.connect(self.assign_section_to_selected_members)

        # --- Constraints / Loads (quick add) ---
        self.act_add_dx.triggered.connect(self.edit_constraints_selected)
        self.act_add_dy.triggered.connect(self.edit_constraints_selected)
        self.act_add_rz.triggered.connect(self.edit_constraints_selected)

        self.act_add_fy.triggered.connect(self.edit_nodal_loads_selected)
        self.act_add_mz.triggered.connect(self.edit_nodal_loads_selected)
        self.act_add_udl.triggered.connect(self.add_udl_to_selected_members)

        # --- Background ---
        self.act_bg_import.triggered.connect(self.import_background)
        self.act_bg_calibrate.triggered.connect(self.calibrate_background)
        self.act_bg_opacity.triggered.connect(self.set_background_opacity)
        self.act_bg_bw.toggled.connect(self.toggle_background_bw)
        self.act_bg_clear.triggered.connect(self.clear_background)

        # --- Solve / Results ---
        self.act_validate.triggered.connect(self.validate_only)
        self.act_solve.triggered.connect(self.solve_active)
        self.act_show_results.triggered.connect(self.show_results)
        self.act_back_to_model.triggered.connect(self.back_to_model)
        self.act_export_csv.triggered.connect(self.export_results_csv)
        # Changing the dropdown should redraw if we are already in results view
        if hasattr(self, 'cmb_result_type'):
            self.cmb_result_type.currentTextChanged.connect(lambda _=None: (
                self.show_results() if self.center_stack.currentWidget() == self.results_view else None
            ))

        # --- Canvas signals ---
        self.canvas.selection_changed.connect(self.on_selection_changed)
        self.canvas.point_added.connect(self.on_point_added)
        self.canvas.point_moved.connect(self.on_point_moved)
        self.canvas.background_calibration_ready.connect(self._on_bg_calib_ready)
        self.canvas.request_edit_constraints.connect(self.edit_constraints_selected)
        self.canvas.request_edit_nodal_loads.connect(self.edit_nodal_loads_selected)
        self.canvas.request_delete_selected_points.connect(self.delete_selected_points)


    def set_model_mode(self, mode: str):
        """Set active canvas mode.

        Phase-1 supports 'select' and 'add_point'. We also remember the last
        modeling mode for repeated modeling actions (Planar Sketch style).
        """
        self.canvas.set_mode(mode)
        if mode != "select":
            self._last_model_mode = mode

    # ---------------- Undo/Redo ----------------
    def _push_snapshot(self, label: str, before: dict, after: dict):
        # Avoid no-op snapshots
        if before == after:
            return
        self.undo_stack.push(label, before, after)

    def undo(self):
        d = self.undo_stack.undo()
        if d is None:
            return
        self.project = Project.from_dict(d)
        # Keep toolbar state in sync
        self.act_auto_members.blockSignals(True)
        self.act_auto_members.setChecked(self.project.auto_members)
        self.act_auto_members.blockSignals(False)
        self._schedule_refresh()

    def redo(self):
        d = self.undo_stack.redo()
        if d is None:
            return
        self.project = Project.from_dict(d)
        self.act_auto_members.blockSignals(True)
        self.act_auto_members.setChecked(self.project.auto_members)
        self.act_auto_members.blockSignals(False)
        self._schedule_refresh()

    def repeat_last_model_action(self):
        # Phase-1: switches to continuous Add Point
        self.set_model_mode(self._last_model_mode)

    # ---------------- Model actions ----------------
    def _schedule_refresh(self, full: bool = False):
        # If the canvas is in an active interaction (dragging, context menu
        # nested event loop, etc.), defer refresh a bit longer. Clearing and
        # rebuilding the QGraphicsScene during those interactions can hard
        # crash Qt on Windows (0xC0000409).
        if self._refresh_pending:
            return
        self._refresh_pending = True

        def _try():
            # Keep waiting until interaction is done.
            if getattr(self.canvas, "is_interacting", None) and self.canvas.is_interacting():
                QTimer.singleShot(25, _try)
                return
            self._do_refresh()

        QTimer.singleShot(0, _try)

    def _do_refresh(self):
        self._refresh_pending = False
        # Rebuild all UI views from current project state
        self.refresh_all()
        if self._reselect_point_uid:
            self.canvas.select_point(self._reselect_point_uid)
            self._reselect_point_uid = None

    def _schedule_rebuild(self):
        if self._rebuild_pending:
            return
        self._rebuild_pending = True

        def _try():
            if getattr(self.canvas, "is_interacting", None) and self.canvas.is_interacting():
                QTimer.singleShot(25, _try)
                return
            self._do_rebuild()

        QTimer.singleShot(0, _try)

    def _do_rebuild(self):
        self._rebuild_pending = False
        self._rebuild_model_members()
        self._schedule_refresh()

    def _rebuild_model_members(self):
        """Rebuild members in the data model while preserving identities.

        IMPORTANT:
        - If we clear & recreate members on every point move, their UIDs
          change, which forces a full scene rebuild (QGraphicsScene.clear).
          That combination is a common cause of hard Qt crashes on Windows.
        - We therefore preserve existing members wherever possible and only
          add/remove the deltas.
        """
        pts = self.project.sorted_points()
        desired_pairs: list[tuple[str, str]] = []
        for i in range(len(pts) - 1):
            desired_pairs.append((pts[i].uid, pts[i + 1].uid))

        # Index existing members by (i_uid, j_uid)
        existing_by_pair: dict[tuple[str, str], str] = {
            (m.i_uid, m.j_uid): mid for mid, m in self.project.members.items()
        }

        from ..core.model import Member

        new_members: dict[str, Member] = {}

        # Keep or create members in left->right order
        for (i_uid, j_uid) in desired_pairs:
            if (i_uid, j_uid) in existing_by_pair:
                mid = existing_by_pair[(i_uid, j_uid)]
                new_members[mid] = self.project.members[mid]
            else:
                m = Member(
                    i_uid=i_uid,
                    j_uid=j_uid,
                    material_uid=self.project.active_material_uid or "",
                    section_uid=self.project.active_section_uid or "",
                )
                # Deterministic uid helps keep the canvas incremental even
                # if you Undo/Redo or reload a file.
                m.uid = f"M_{i_uid[:8]}_{j_uid[:8]}"
                new_members[m.uid] = m

        self.project.members = new_members
        self.project.rebuild_names()

    def delete_selected_points(self):
        pids = self.canvas.selected_point_uids()
        if not pids:
            return
        before = self.project.to_dict()
        # Delete members connected to these points
        to_delete_members = [
            mid for mid, m in self.project.members.items()
            if m.i_uid in pids or m.j_uid in pids
        ]
        for mid in to_delete_members:
            self.project.members.pop(mid, None)
        for pid in pids:
            self.project.points.pop(pid, None)
        if self.project.auto_members:
            self._rebuild_model_members()
        else:
            self.project.rebuild_names()
        after = self.project.to_dict()
        self._push_snapshot("Delete Point", before, after)
        self._schedule_refresh()

    def _context_add_constraint(self, dof: str):
        # Right-click helper: apply a constraint to the currently selected point
        map_dof_to_ui = {"DX": "DX (UX)", "DY": "DY (UY)", "RZ": "RZ"}
        label = map_dof_to_ui.get(dof)
        if label:
            self.cmb_constraint_type.setCurrentText(label)
        self.apply_constraint_to_selected_points()

    def _context_add_nodal_load(self, load_type: str):
        # Right-click helper: apply nodal load (FY or MZ) to selected point
        if load_type == "FY":
            self.cmb_load_type.setCurrentText("Nodal FY")
        elif load_type == "MZ":
            self.cmb_load_type.setCurrentText("Nodal MZ")
        self.apply_load_to_selection()

    def _toggle_auto_members(self, checked: bool):
        """Toggle auto member rebuilding.

        Bound to the toolbar check action.
        """
        self.project.auto_members = bool(checked)
        if self.project.auto_members:
            self.rebuild_members_now()
        else:
            self._schedule_refresh()

    def rebuild_members_now(self):
        # Defer rebuild to avoid QGraphicsScene clear while mouse events are active
        self._schedule_rebuild()

    def open_materials(self):
        before = self.project.to_dict()
        dlg = MaterialManagerDialog(self.project, self)
        if dlg.exec():
            after = self.project.to_dict()
            self._push_snapshot("Edit Materials", before, after)
            self._schedule_refresh()

    def open_sections(self):
        before = self.project.to_dict()
        dlg = SectionManagerDialog(self.project, self)
        if dlg.exec():
            after = self.project.to_dict()
            self._push_snapshot("Edit Sections", before, after)
            self._schedule_refresh()

    def assign_material_to_selected_members(self):
        mids = self.canvas.selected_member_uids()
        if not mids:
            return
        if not self.project.active_material_uid:
            QMessageBox.warning(self, "No material", "请先在 Materials 中选择一个 Active material。")
            return
        before = self.project.to_dict()
        for uid in mids:
            self.project.members[uid].material_uid = self.project.active_material_uid
        after = self.project.to_dict()
        self._push_snapshot("Assign Material", before, after)
        self._schedule_refresh()

    def assign_section_to_selected_members(self):
        mids = self.canvas.selected_member_uids()
        if not mids:
            return
        if not self.project.active_section_uid:
            QMessageBox.warning(self, "No section", "请先在 Sections 中选择一个 Active section。")
            return
        before = self.project.to_dict()
        for uid in mids:
            self.project.members[uid].section_uid = self.project.active_section_uid
        after = self.project.to_dict()
        self._push_snapshot("Assign Section", before, after)
        self._schedule_refresh()

    def apply_constraint_to_selected_points(self):
        pids = self.canvas.selected_point_uids()
        if not pids:
            return
        before = self.project.to_dict()
        typ = self.cmb_constraint_type.currentText()
        val = float(self.sp_constraint_value.value())
        dof = {"DY (UY)": "DY", "RZ": "RZ", "DX (UX)": "DX"}[typ]
        from ..core.model import Constraint
        for uid in pids:
            p = self.project.points[uid]
            p.constraints[dof] = Constraint(dof=dof, value=val, enabled=True)
        after = self.project.to_dict()
        self._push_snapshot("Apply Constraint", before, after)
        self._schedule_refresh()

    def clear_constraints_on_selected_points(self):
        pids = self.canvas.selected_point_uids()
        if not pids:
            return
        before = self.project.to_dict()
        for uid in pids:
            self.project.points[uid].constraints.clear()
        after = self.project.to_dict()
        self._push_snapshot("Clear Constraints", before, after)
        self._schedule_refresh()

    def _set_active_load_case(self, case: str):
        self.project.active_load_case = case

    def apply_load_to_selection(self):
        case = self.project.active_load_case
        typ = self.cmb_load_type.currentText()
        val = float(self.sp_load_value.value())
        before = self.project.to_dict()
        if typ.startswith("Nodal"):
            pids = self.canvas.selected_point_uids()
            if not pids:
                return
            from ..core.model import NodalLoad
            direction = "FY" if "FY" in typ else "MZ"
            for uid in pids:
                self.project.points[uid].nodal_loads.append(NodalLoad(direction=direction, value=val, case=case))
        else:
            mids = self.canvas.selected_member_uids()
            if not mids:
                return
            from ..core.model import MemberLoadUDL
            for uid in mids:
                self.project.members[uid].udl_loads.append(MemberLoadUDL(direction="Fy", w=val, case=case))
        after = self.project.to_dict()
        self._push_snapshot("Apply Load", before, after)
        self._schedule_refresh()

    def clear_loads_on_selection(self):
        case = self.project.active_load_case
        pids = self.canvas.selected_point_uids()
        mids = self.canvas.selected_member_uids()
        before = self.project.to_dict()
        for uid in pids:
            p = self.project.points[uid]
            p.nodal_loads = [ld for ld in p.nodal_loads if ld.case != case]
        for uid in mids:
            m = self.project.members[uid]
            m.udl_loads = [ld for ld in m.udl_loads if ld.case != case]
        after = self.project.to_dict()
        self._push_snapshot("Clear Loads", before, after)
        self._schedule_refresh()

    def validate_only(self):
        sfw = getattr(self, "sp_safety_factor", None)
        if sfw is not None:
            self.project.safety_factor = float(sfw.value())
        msgs = validate_project(self.project)
        if not msgs:
            QMessageBox.information(self, "Validate", "OK：模型检查通过。")
            return
        text = "\n".join([f"[{m.level}] {m.text}" for m in msgs])
        if any(m.level == "ERROR" for m in msgs):
            QMessageBox.critical(self, "Validate", text)
        else:
            QMessageBox.warning(self, "Validate", text)

    def solve_model(self):
        sfw = getattr(self, 'sp_safety_factor', None)
        if sfw is not None:
            self.project.safety_factor = float(sfw.value())
        msgs = validate_project(self.project)
        if any(m.level == "ERROR" for m in msgs):
            text = "\n".join([f"[{m.level}] {m.text}" for m in msgs])
            QMessageBox.critical(self, "Cannot Solve", text)
            return
        try:
            self.project.rebuild_names()
            out = solve_with_pynite(self.project, "COMB1", n_samples_per_member=int(getattr(getattr(self, "sp_samples", None), "value", lambda: 50)()))
            self.last_results = out
            QMessageBox.information(self, "Solve", "求解成功。请到 Results 里查看。")
        except PyniteSolverError as e:
            QMessageBox.critical(self, "PyNite Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Solve Error", f"{type(e).__name__}: {e}")

    def show_results(self):
        if self.last_results is None:
            QMessageBox.information(self, "Results", "还没有结果。请先 Solve。")
            return
        cmb = getattr(self, "cmb_result_type", None)
        rtype = cmb.currentText() if cmb is not None else "Deflection"
        sp = getattr(self, "sp_def_scale", None)
        def_scale = float(sp.value()) if sp is not None else 1.0
        self.results_view.set_data(self.project, self.last_results, rtype, def_scale=def_scale)
        self.center_stack.setCurrentWidget(self.results_view)

    def back_to_model(self):
        # Return from results to interactive modeling canvas
        self.center_stack.setCurrentWidget(self.canvas)

    # ---------------- Canvas callbacks ----------------

    def export_results_csv(self):
        if self.last_results is None:
            QMessageBox.information(self, "Export CSV", "还没有结果。请先 Solve。")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export Results CSV", "results.csv", "CSV Files (*.csv)")
        if not fn:
            return

        prj = self.project
        out = self.last_results

        # Beam span and normalization (leftmost point is x=0)
        try:
            xs_pts = [p.x for p in prj.points.values()]
            x0 = float(min(xs_pts)) if xs_pts else 0.0
            x1 = float(max(xs_pts)) if xs_pts else 0.0
        except Exception:
            x0, x1 = 0.0, 0.0

        # Nodal table (always exported)
        pts_sorted = prj.sorted_points()
        x_nodes = [p.x - x0 for p in pts_sorted]
        try:
            dy_nodes = list(out.dy_nodes) if out.dy_nodes is not None else [0.0] * len(pts_sorted)
        except Exception:
            dy_nodes = [0.0] * len(pts_sorted)

        # Diagram table (may be empty)
        xg = np.asarray(getattr(out, "x_diag", []), dtype=float)
        has_diag = xg.size > 0
        if has_diag:
            mask = (xg >= x0 - 1e-9) & (xg <= x1 + 1e-9)
            xg = xg[mask]
            x = xg - x0
            V = np.asarray(getattr(out, "V", []), dtype=float)[mask]
            M = np.asarray(getattr(out, "M", []), dtype=float)[mask]
            sigma = np.asarray(getattr(out, "sigma", []), dtype=float)[mask]
            margin = np.asarray(getattr(out, "margin", []), dtype=float)[mask]

            # Deflection on diagram x: interpolate nodal DY
            try:
                xn_abs = np.asarray(getattr(out, "x_nodes", []), dtype=float)
                dyn = np.asarray(getattr(out, "dy_nodes", []), dtype=float)
                if xn_abs.size >= 2:
                    dy = np.interp(xg, xn_abs, dyn)
                elif xn_abs.size == 1:
                    dy = np.full_like(xg, dyn[0])
                else:
                    dy = np.zeros_like(xg)
            except Exception:
                dy = np.zeros_like(xg)
        else:
            x = np.array([], dtype=float)
            dy = np.array([], dtype=float)
            V = M = sigma = margin = np.array([], dtype=float)

        import csv
        try:
            with open(fn, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)

                # Unified table header (avoid duplicated header blocks)
                w.writerow(["TYPE", "name", "combo", "x_mm", "dy_mm", "Rxn_FX_N", "Rxn_FY_N", "Rxn_MZ_Nmm", "V_N", "M_Nmm", "sigma_Nmm2", "MS"])

                node_rows = []
                for i, p in enumerate(pts_sorted, start=1):
                    r = out.reactions.get(p.name, {}) if getattr(out, "reactions", None) else {}
                    node_rows.append([
                        "NODE",
                        p.name,
                        "",
                        f"{x_nodes[i-1]:.6f}",
                        f"{dy_nodes[i-1]:.9g}",
                        f"{float(r.get('FX', 0.0)):.9g}",
                        f"{float(r.get('FY', 0.0)):.9g}",
                        f"{float(r.get('MZ', 0.0)):.9g}",
                        "",
                        "",
                        "",
                        "",
                    ])

                if len(x) == 0:
                    for row in node_rows:
                        w.writerow(row)
                else:
                    diag_rows = []
                    for i in range(len(x)):
                        diag_rows.append([
                            "DIAG",
                            "",
                            out.combo,
                            f"{x[i]:.6f}",
                            f"{dy[i]:.9g}",
                            "",
                            "",
                            "",
                            f"{V[i]:.9g}" if i < len(V) else "",
                            f"{M[i]:.9g}" if i < len(M) else "",
                            f"{sigma[i]:.9g}" if i < len(sigma) else "",
                            f"{margin[i]:.9g}" if i < len(margin) else "",
                        ])

                    x_vals = np.asarray(x, dtype=float)
                    for row in node_rows:
                        x_node = float(row[3])
                        idx = int(np.argmin(np.abs(x_vals - x_node))) if x_vals.size else -1
                        if idx >= 0 and abs(float(x_vals[idx]) - x_node) <= 1e-6:
                            # Keep diagram values (V/M/sigma/MS) at node positions while
                            # injecting node identity and reactions into the same row.
                            merged_row = list(diag_rows[idx])
                            merged_row[0] = row[0]
                            merged_row[1] = row[1]
                            merged_row[2] = row[2]
                            merged_row[3] = row[3]
                            merged_row[4] = row[4]
                            merged_row[5] = row[5]
                            merged_row[6] = row[6]
                            merged_row[7] = row[7]
                            diag_rows[idx] = merged_row
                        else:
                            diag_rows.append(row)

                    for row in sorted(diag_rows, key=lambda r: (float(r[3]), 0 if r[0] == "NODE" else 1)):
                        w.writerow(row)
        except Exception as e:
            QMessageBox.critical(self, "Export CSV", f"导出失败：{e}")
            return

        QMessageBox.information(self, "Export CSV", f"已导出：\n{fn}")

    def on_point_added(self, x: float):
        before = self.project.to_dict()
        from ..core.model import Point
        p = Point(x=x)
        self.project.points[p.uid] = p
        self._reselect_point_uid = p.uid
        if self.project.auto_members:
            self._rebuild_model_members()
        else:
            self.project.rebuild_names()
        after = self.project.to_dict()
        self._push_snapshot("Add Point", before, after)
        self._schedule_refresh()

    def on_point_moved(self, uid: str, new_x: float):
        if uid not in self.project.points:
            return
        before = self.project.to_dict()
        self.project.points[uid].x = new_x
        self._reselect_point_uid = uid
        if self.project.auto_members:
            self._rebuild_model_members()
        else:
            self.project.rebuild_names()
        after = self.project.to_dict()
        self._push_snapshot("Move Point", before, after)
        self._schedule_refresh()

    def on_selection_changed(self):
        pids = self.canvas.selected_point_uids()
        mids = self.canvas.selected_member_uids()
        parts = []
        if pids:
            parts.append("Points: " + ", ".join(self.project.points[uid].name for uid in pids))
        if mids:
            parts.append("Members: " + ", ".join(self.project.members[uid].name for uid in mids))
        self.lbl_sel.setText("Selection: " + ("; ".join(parts) if parts else "(none)"))

        # property panel
        if len(pids) == 1 and not mids:
            p = self.project.points[pids[0]]
            self.ed_x.blockSignals(True)
            self.ed_x.setEnabled(True)
            self.ed_x.setValue(p.x)
            self.ed_x.blockSignals(False)
            self.lbl_len.setText("-")
        else:
            self.ed_x.setEnabled(False)
            self.lbl_len.setText("-")

        if len(mids) == 1 and not pids:
            m = self.project.members[mids[0]]
            xi = self.project.points[m.i_uid].x
            xj = self.project.points[m.j_uid].x
            self.lbl_len.setText(f"{abs(xj-xi):.3f} mm")

    def _edit_selected_point_x(self, x: float):
        pids = self.canvas.selected_point_uids()
        if len(pids) != 1:
            return
        self.on_point_moved(pids[0], float(x))

    # ---------------- Refresh ----------------
    def refresh_all(self):
        self.project.rebuild_names()
        self.refresh_tree()
        # Full sync only when topology changes. For point moves/attribute edits
        # we prefer incremental updates to avoid QGraphicsScene.clear during
        # interactive operations (can hard-crash Qt on Windows).
        try:
            self.canvas.sync(self.project, full=False)
        except Exception:
            # Fallback to a full rebuild if something unexpected happens.
            self.canvas.sync(self.project, full=True)

    def refresh_tree(self):
        self.tree.clear()
        root_pts = QTreeWidgetItem(["Points"])
        root_mem = QTreeWidgetItem(["Members"])
        root_mat = QTreeWidgetItem(["Materials"])
        root_sec = QTreeWidgetItem(["Sections"])
        self.tree.addTopLevelItem(root_pts)
        self.tree.addTopLevelItem(root_mem)
        self.tree.addTopLevelItem(root_mat)
        self.tree.addTopLevelItem(root_sec)

        for p in self.project.sorted_points():
            it = QTreeWidgetItem([f"{p.name}  x={p.x:.3f}"])
            root_pts.addChild(it)

        mems = sorted(self.project.members.values(), key=lambda m: m.name)
        for m in mems:
            xi = self.project.points[m.i_uid].x
            xj = self.project.points[m.j_uid].x
            it = QTreeWidgetItem([f"{m.name}  L={abs(xj-xi):.3f}"])
            root_mem.addChild(it)

        for mat in self.project.materials.values():
            mark = " ★" if mat.uid == self.project.active_material_uid else ""
            it = QTreeWidgetItem([f"{mat.name}{mark}"])
            root_mat.addChild(it)

        for sec in self.project.sections.values():
            mark = " ★" if sec.uid == self.project.active_section_uid else ""
            it = QTreeWidgetItem([f"{sec.name}{mark}"])
            root_sec.addChild(it)

        self.tree.expandAll()


    # ---------------- File: New/Open/Save (JSON model) ----------------
    def new_project(self):
        # Keep libraries (materials/sections) already loaded; clear model topology.
        self.project.points.clear()
        self.project.members.clear()
        self.project.rebuild_names()
        self.last_results = None
        self.undo_stack.clear()
        self._schedule_refresh()

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "MiniBeam Model (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.project.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Model", "", "MiniBeam Model (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.project = Project.from_dict(d)
            # Re-bind project to canvas
            self.canvas.project = self.project
            self.last_results = None
            self.undo_stack.clear()
            self._schedule_refresh(full=True)
        except Exception as e:
            QMessageBox.critical(self, "Open Failed", str(e))

    # ---------------- Quick add constraints / loads ----------------

    # ---------------- Constraint / Load editors ----------------
    def edit_constraints_selected(self):
        """Open a single dialog to edit DX/DY/RZ constraints for selected point(s).
        Existing constraints are overwritten (no duplicates)."""
        pids = self.canvas.selected_point_uids()
        if not pids:
            return

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QCheckBox, QWidget, QHBoxLayout, QDoubleSpinBox

        dlg = QDialog(self)
        dlg.setWindowTitle("Constraint (DX, DY, RZ)")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        lay.addLayout(form)

        def mk_row(label, enabled_default=False, value_default=0.0):
            cb = QCheckBox("Enable")
            cb.setChecked(bool(enabled_default))
            sb = QDoubleSpinBox()
            sb.setRange(-1e12, 1e12)
            sb.setDecimals(6)
            sb.setValue(float(value_default))
            roww = QWidget()
            hl = QHBoxLayout(roww)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(cb)
            hl.addWidget(sb, 1)
            form.addRow(label, roww)
            return cb, sb

        p0 = self.project.points.get(pids[0])
        dx_en, dx_val = (False, 0.0)
        dy_en, dy_val = (False, 0.0)
        rz_en, rz_val = (False, 0.0)
        if p0:
            if "DX" in p0.constraints:
                dx_en, dx_val = (p0.constraints["DX"].enabled, p0.constraints["DX"].value)
            if "DY" in p0.constraints:
                dy_en, dy_val = (p0.constraints["DY"].enabled, p0.constraints["DY"].value)
            if "RZ" in p0.constraints:
                rz_en, rz_val = (p0.constraints["RZ"].enabled, p0.constraints["RZ"].value)

        cb_dx, sb_dx = mk_row("UX / DX", dx_en, dx_val)
        cb_dy, sb_dy = mk_row("UY / DY", dy_en, dy_val)
        cb_rz, sb_rz = mk_row("RZ", rz_en, rz_val)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        before = self.project.to_dict()
        for uid in pids:
            p = self.project.points.get(uid)
            if not p:
                continue
            # overwrite/clear
            if cb_dx.isChecked():
                p.constraints["DX"] = Constraint(dof="DX", value=float(sb_dx.value()), enabled=True)
            else:
                p.constraints.pop("DX", None)

            if cb_dy.isChecked():
                p.constraints["DY"] = Constraint(dof="DY", value=float(sb_dy.value()), enabled=True)
            else:
                p.constraints.pop("DY", None)

            if cb_rz.isChecked():
                p.constraints["RZ"] = Constraint(dof="RZ", value=float(sb_rz.value()), enabled=True)
            else:
                p.constraints.pop("RZ", None)

        after = self.project.to_dict()
        self._push_snapshot("Edit constraints", before, after)
        self._schedule_refresh()

    def edit_nodal_loads_selected(self):
        """Open a dialog to edit FY and MZ nodal loads for selected point(s) in active load case.
        Loads are unique per (direction, loadcase) and will be overwritten (no duplicates)."""
        pids = self.canvas.selected_point_uids()
        if not pids:
            return

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QCheckBox, QWidget, QHBoxLayout, QDoubleSpinBox

        dlg = QDialog(self)
        dlg.setWindowTitle("Load (FY, MZ)")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        lay.addLayout(form)

        def mk_row(label, enabled_default=False, value_default=0.0):
            cb = QCheckBox("Enable")
            cb.setChecked(bool(enabled_default))
            sb = QDoubleSpinBox()
            sb.setRange(-1e12, 1e12)
            sb.setDecimals(6)
            sb.setValue(float(value_default))
            roww = QWidget()
            hl = QHBoxLayout(roww)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(cb)
            hl.addWidget(sb, 1)
            form.addRow(label, roww)
            return cb, sb

        lc = self.project.active_load_case
        p0 = self.project.points.get(pids[0])

        fy_en, fy_val = (False, 0.0)
        mz_en, mz_val = (False, 0.0)
        if p0:
            for ld in p0.nodal_loads:
                if ld.case == lc and ld.direction == "FY":
                    fy_en, fy_val = (True, ld.value)
                if ld.case == lc and ld.direction == "MZ":
                    mz_en, mz_val = (True, ld.value)

        cb_fy, sb_fy = mk_row("FY (N)", fy_en, fy_val)
        cb_mz, sb_mz = mk_row("MZ (N·mm)", mz_en, mz_val)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        before = self.project.to_dict()
        for uid in pids:
            p = self.project.points.get(uid)
            if not p:
                continue

            # remove existing for this LC
            p.nodal_loads = [ld for ld in p.nodal_loads if ld.case != lc or ld.direction not in ("FY", "MZ")]

            if cb_fy.isChecked():
                p.nodal_loads.append(NodalLoad(direction="FY", value=float(sb_fy.value()), case=lc))
            if cb_mz.isChecked():
                p.nodal_loads.append(NodalLoad(direction="MZ", value=float(sb_mz.value()), case=lc))

        after = self.project.to_dict()
        self._push_snapshot("Edit loads", before, after)
        self._schedule_refresh()
    def add_udl_to_selected_members(self):
        from PyQt6.QtWidgets import QInputDialog
        mids = self.canvas.selected_member_uids()
        if not mids:
            return
        w, ok = QInputDialog.getDouble(self, "UDL", "w (N/mm, +Y upward):", 0.0, -1e12, 1e12, 6)
        if not ok:
            return
        from ..core.model import MemberLoadUDL
        lc = self.project.active_load_case
        for mid in mids:
            m = self.project.members.get(mid)
            if m is None:
                continue
            m.udl_loads.append(MemberLoadUDL(direction="Fy", w=float(w), case=lc))
        self._schedule_refresh()

    # ---------------- Solve / results wrappers ----------------
    def solve_active(self):
        self.solve_model()



    # ---------------- Background ----------------
    def import_background(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Import Background Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not fn:
            return
        pix = QPixmap(fn)
        if pix.isNull():
            QMessageBox.warning(self, "Background", "Failed to load image.")
            return
        self.canvas.set_background(pix)
        self._schedule_refresh()

    def calibrate_background(self):
        if getattr(self.canvas, "_bg_item", None) is None:
            QMessageBox.information(self, "Background", "Please import a background image first.")
            return
        QMessageBox.information(self, "Background Calibration", "Click two points on the background image to define a known distance.")
        self.canvas.start_background_calibration()

    def _on_bg_calib_ready(self, p1, p2):
        dist, ok = QInputDialog.getDouble(self, "Background Calibration", "Enter real distance between the two clicked points (mm):", 100.0, 1e-6, 1e12, 3)
        if not ok:
            return
        try:
            self.canvas.apply_background_calibration(p1, p2, float(dist))
        except Exception as e:
            QMessageBox.critical(self, "Background Calibration", f"Failed to apply calibration: {e}")
        self._schedule_refresh()

    def set_background_opacity(self):
        alpha, ok = QInputDialog.getDouble(self, "Background Opacity", "Opacity (0~1):", float(getattr(self.canvas, "_bg_opacity", 0.35)), 0.0, 1.0, 2)
        if not ok:
            return
        self.canvas.set_background_opacity(float(alpha))

    def toggle_background_bw(self, on: bool):
        try:
            self.canvas.set_background_grayscale(bool(on))
        except Exception as e:
            QMessageBox.critical(self, "Background", f"Failed to set B/W: {e}")

    def clear_background(self):
        self.canvas.clear_background()
        self._schedule_refresh()

    # ---------------- Constraints dialog ----------------
