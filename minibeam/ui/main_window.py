from __future__ import annotations
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget, QTreeWidgetItem,
    QLabel, QMessageBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QStackedWidget, QInputDialog, QTableWidget, QTableWidgetItem, QDialog,
    QDialogButtonBox, QRadioButton, QTextBrowser
)
from PyQt6.QtCore import Qt, QTimer, QSize, QUrl
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QPixmap, QColor, QDesktopServices, QFontMetrics, QTextDocument, QPageSize
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import QStyle, QFileDialog, QHeaderView

import sys
import traceback
import json
import logging
import os
from pathlib import Path
import numpy as np

from ..core.model import Project, Material, Section, Constraint, Bush, NodalLoad
from ..core.section_props import rect_solid, circle_solid, i_section
from ..core.validation import validate_project
from ..core.pynite_adapter import solve_with_pynite, PyniteSolverError, SolveOutput
from ..core.report_export import build_standard_report_html

from .canvas_view import BeamCanvas
from .dialogs import MaterialManagerDialog, SectionManagerDialog
from .results_view import ResultsView, ResultsGridDialog
from .i18n import LANG_ZH, tr
from ..core.undo import UndoStack
from ..core.library_store import (
    load_builtin_material_library,
    load_material_library,
    load_section_library,
    merge_by_name,
)


class MainWindow(QMainWindow):
    def _resource_base_dir(self) -> Path:
        """Return runtime directory for adjacent resources (PyInstaller-safe)."""
        if getattr(sys, "frozen", False):
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                return Path(meipass).resolve()
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parents[2]

    def _resource_path(self, relative_path: str) -> Path:
        return self._resource_base_dir() / relative_path

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
        app_icon = self._resource_path("minibeam/assets/app_icon.svg")
        if app_icon.exists():
            self.setWindowIcon(QIcon(str(app_icon)))
        self.setWindowTitle("MiniBeam v0.2.0")
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
        self._syncing_canvas_view = False
        self._rebuild_pending = False
        self._reselect_point_uid: str | None = None

        self.current_lang = LANG_ZH
        self._tr = lambda key, **kwargs: tr(self.current_lang, key, **kwargs)

        self.project = Project()
        self.project.spatial_mode = self._choose_spatial_mode(default_mode="2D")
        self.undo_stack = UndoStack()
        self._last_model_mode: str = "add_point"  # Phase-1: only point modeling
        # seed one material & one section
        mat = Material(name="steel", E=206000.0, G=79300.0, nu=0.3, rho=7.85e-6, sigma_y=235.0)
        self.project.materials[mat.uid] = mat

        sp = rect_solid(100.0, 10.0)
        sec = Section(name="Rect100x10", type="RectSolid", A=sp.A, Iy=sp.Iy, Iz=sp.Iz, J=sp.J, c_y=sp.c_y, c_z=sp.c_z, Zp_y=sp.Zp_y, Zp_z=sp.Zp_z, shape_factor_y=sp.shape_factor_y, shape_factor_z=sp.shape_factor_z, shape_factor_t=sp.shape_factor_t, Zp=sp.Zp_z, shape_factor=sp.shape_factor_z)
        self.project.sections[sec.uid] = sec

        self._merge_libraries_into_project()

        self._build_ui()
        self._connect()
        self._sync_background_visibility_action()
        self._schedule_refresh()

        self.last_results: SolveOutput | None = None

    def _merge_libraries_into_project(self):
        # Keep material/section library data available across open/save cycles.
        try:
            merge_by_name(self.project.materials, load_builtin_material_library(), name_attr="name")
            merge_by_name(self.project.materials, load_material_library(), name_attr="name")
            merge_by_name(self.project.sections, load_section_library(), name_attr="name")
        except Exception as exc:
            logging.exception("Failed to merge material/section libraries into project")
            if os.environ.get("MINIBEAM_DEV") == "1":
                QMessageBox.critical(self, "Library Load Error", "".join(traceback.format_exception(exc)))
            else:
                self.statusBar().showMessage(self._tr("msg.library_load_failed"), 6000)

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        
        # ---------------- Ribbon (Word-like tabs with large icons) ----------------
        icon_size = QSize(40, 40)

        # Actions (wired in _connect)
        self.act_new = QAction(self._std_icon("SP_FileIcon", "SP_DirIcon"), self._tr("action.new"), self)
        self.act_open = QAction(self._std_icon("SP_DialogOpenButton", "SP_DirOpenIcon"), self._tr("action.open"), self)
        self.act_save = QAction(self._std_icon("SP_DialogSaveButton", "SP_DriveHDIcon"), self._tr("action.save"), self)

        self.act_select = QAction(self._std_icon("SP_ArrowCursor", "SP_ArrowForward"), self._tr("action.select"), self)
        self.act_add_point = QAction(self._std_icon("SP_FileDialogNewFolder", "SP_DirIcon"), self._tr("action.add_point"), self)
        self.act_delete = QAction(self._std_icon("SP_TrashIcon", "SP_DialogCloseButton"), self._tr("action.delete"), self)

        self.act_materials = QAction(self._std_icon("SP_DriveFDIcon", "SP_DriveHDIcon"), self._tr("action.materials"), self)
        self.act_sections = QAction(self._std_icon("SP_DriveDVDIcon", "SP_DriveHDIcon"), self._tr("action.sections"), self)
        self.act_assign_prop = QAction(self._std_icon("SP_ArrowDown", "SP_ArrowForward"), self._tr("action.assign_property"), self)

        self.act_add_dx = QAction(self._std_icon("SP_DialogYesButton", "SP_DialogApplyButton"), self._tr("action.constraint"), self)
        self.act_add_bush = QAction(self._std_icon("SP_DialogYesButton", "SP_DialogApplyButton"), self._tr("action.bush"), self)
        self.act_add_dy = QAction(self._std_icon("SP_DialogYesButton", "SP_DialogApplyButton"), "Add DY", self)
        self.act_add_rz = QAction(self._std_icon("SP_DialogYesButton", "SP_DialogApplyButton"), "Add RZ", self)

        self.act_add_fy = QAction(self._std_icon("SP_ArrowUp", "SP_ArrowForward"), self._tr("action.load"), self)
        self.act_add_mz = QAction(self._std_icon("SP_ArrowUp", "SP_ArrowForward"), "Add MZ", self)
        self.act_add_udl = QAction(self._std_icon("SP_ArrowDown", "SP_ArrowForward"), self._tr("action.udl"), self)

        # Background
        self.act_bg_import = QAction(self._std_icon("SP_DialogOpenButton", "SP_DirOpenIcon"), self._tr("action.import"), self)
        self.act_bg_calibrate = QAction(self._std_icon("SP_BrowserReload", "SP_DialogResetButton"), self._tr("action.calibrate"), self)
        self.act_bg_opacity = QAction(self._std_icon("SP_DialogApplyButton", "SP_DialogOkButton"), self._tr("action.opacity"), self)
        self.act_bg_bw = QAction(self._std_icon("SP_FileDialogDetailedView", "SP_FileDialogListView"), self._tr("action.bw"), self)
        self.act_bg_bw.setCheckable(True)
        self.act_bg_visible = QAction(self._std_icon("SP_DialogYesButton", "SP_DialogApplyButton"), self._tr("action.show_bg"), self)
        self.act_bg_visible.setCheckable(True)
        self.act_bg_visible.setChecked(True)
        self.act_bg_clear = QAction(self._std_icon("SP_TrashIcon", "SP_DialogCancelButton"), self._tr("action.clear"), self)

        self.act_validate = QAction(self._std_icon("SP_DialogApplyButton", "SP_DialogOkButton"), self._tr("action.validate"), self)
        self.act_solve = QAction(self._std_icon("SP_MediaPlay", "SP_DialogOkButton"), self._tr("action.solve"), self)

        self.act_show_results = QAction(self._std_icon("SP_ComputerIcon", "SP_DesktopIcon"), self._tr("action.results"), self)
        self.act_export_csv = QAction(self._std_icon("SP_DialogSaveButton", "SP_DriveHDIcon"), self._tr("action.export_csv"), self)
        self.act_export_report = QAction(self._std_icon("SP_DialogSaveButton", "SP_DriveHDIcon"), self._tr("action.export_report"), self)
        self.act_help_pdf = QAction(self._std_icon("SP_DialogHelpButton", "SP_MessageBoxInformation"), self._tr("action.help"), self)
        self.act_about = QAction(self._std_icon("SP_MessageBoxInformation", "SP_DialogHelpButton"), self._tr("action.copyright"), self)

        # Shortcuts
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_new.setShortcut(QKeySequence.StandardKey.New)
        self.act_delete.setShortcuts([
            QKeySequence(Qt.Key.Key_Delete),
            QKeySequence(Qt.Key.Key_Backspace),
        ])

        self.act_undo = QAction(self._tr("action.undo"), self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_redo = QAction(self._tr("action.redo"), self)
        self.act_redo.setShortcuts([
            QKeySequence.StandardKey.Redo,
            QKeySequence("Ctrl+Y"),
        ])
        self.addAction(self.act_undo)
        self.addAction(self.act_redo)
        self.addAction(self.act_delete)

        # Ribbon widget
        from PyQt6.QtWidgets import QTabWidget, QToolButton, QGridLayout, QSizePolicy

        self._ribbon_tab_keys: list[str] = []
        self._ribbon_buttons: list[QToolButton] = []

        self.ribbon = QTabWidget()
        self.ribbon.setDocumentMode(True)
        root.addWidget(self.ribbon, 0)

        self.cmb_language = QComboBox()
        self.cmb_language.addItem(self._tr("language.zh"), "zh")
        self.cmb_language.addItem(self._tr("language.en"), "en")
        self.cmb_language.setCurrentIndex(0)

        def mk_group(title_key: str, items: list[object]) -> QWidget:
            gb = QGroupBox(self._tr(title_key))
            gb.setProperty("i18n_key", title_key)
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
                    self._ribbon_buttons.append(btn)
                    col += 1
                elif isinstance(it, QWidget):
                    lay.addWidget(it, 0, col)
                    col += 1
                else:
                    # ignore unknown item types
                    pass
            return gb

        def mk_tab(tab_key: str, groups: list[QWidget]) -> QWidget:
            w = QWidget()
            hl = QHBoxLayout(w)
            hl.setContentsMargins(6, 6, 6, 6)
            hl.setSpacing(10)
            for g in groups:
                hl.addWidget(g)
            hl.addStretch(1)
            self.ribbon.addTab(w, self._tr(tab_key))
            self._ribbon_tab_keys.append(tab_key)
            return w

        mk_tab("tab.home", [
            mk_group("group.file", [self.act_new, self.act_open, self.act_save]),
            mk_group("group.model", [self.act_select, self.act_add_point, self.act_delete]),
            mk_group("group.language", [self.cmb_language]),
        ])

        mk_tab("tab.properties", [
            mk_group("group.libraries", [self.act_materials, self.act_sections]),
            mk_group("group.assign", [self.act_assign_prop]),
        ])

        mk_tab("tab.constraints_loads", [
            mk_group("group.constraints", [self.act_add_dx, self.act_add_bush]),
            mk_group("group.loads", [self.act_add_fy, self.act_add_udl]),
        ])

        mk_tab("tab.background", [
            mk_group("group.background", [self.act_bg_import, self.act_bg_calibrate, self.act_bg_opacity, self.act_bg_bw, self.act_bg_visible, self.act_bg_clear]),
        ])

        mk_tab("tab.solve_results", [
            mk_group("group.solve", [self.act_validate, self.act_solve]),
            mk_group("group.results", [self.act_show_results, self.act_export_csv, self.act_export_report]),
        ])

        mk_tab("tab.help", [
            mk_group("group.support", [self.act_help_pdf, self.act_about]),
        ])

        # main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # left tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([self._tr("objects")])
        splitter.addWidget(self.tree)
        self.tree.setMinimumWidth(180)

        # center: canvas + results (stack)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)
        self.canvas = BeamCanvas(view_plane="XY")
        self.canvas.set_translator(self._tr)
        self.canvas_xz: BeamCanvas | None = None
        if self.project.spatial_mode == "3D":
            self.canvas_xz = BeamCanvas(view_plane="XZ")
            self.canvas_xz.set_translator(self._tr)
            self.canvas_xz.set_mode("readonly")
            self.canvas_xz.set_read_only_display(True)
            self.canvas_xz.setDragMode(self.canvas_xz.DragMode.NoDrag)
            self.canvas_xz.setInteractive(False)
        self.results_view = ResultsView(self._tr)
        self.center_stack = QStackedWidget()
        self.canvas_page = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_page)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_layout.setSpacing(4)
        self.xy_label = QLabel(self._tr("xy_view"))
        self.canvas_layout.addWidget(self.xy_label)
        self.canvas_layout.addWidget(self.canvas, 1)
        self.xz_label: QLabel | None = None
        if self.canvas_xz is not None:
            self.xz_label = QLabel(self._tr("xz_view"))
            self.xz_label.setStyleSheet("color:#666;")
            self.canvas_layout.addWidget(self.xz_label)
            self.canvas_layout.addWidget(self.canvas_xz, 1)
        self.center_stack.addWidget(self.canvas_page)
        self.center_stack.addWidget(self.results_view)
        center_layout.addWidget(self.center_stack, 1)
        splitter.addWidget(center)

        # right: property panel
        self.prop = QWidget()
        self.prop.setMinimumWidth(420)
        splitter.addWidget(self.prop)
        pr = QVBoxLayout(self.prop)
        pr.setContentsMargins(6, 6, 6, 6)
        pr.setSpacing(8)

        self.lbl_sel = QLabel(self._tr("selection.none"))
        self.lbl_sel.setWordWrap(True)
        pr.addWidget(self.lbl_sel)

        self.gb_point_props = QGroupBox(self._tr("point_props"))
        pr.addWidget(self.gb_point_props)
        form = QFormLayout(self.gb_point_props)
        self.ed_x = QDoubleSpinBox()
        self.ed_x.setRange(-1e9, 1e9)
        self.ed_x.setDecimals(1)
        self.ed_x.setSingleStep(0.1)
        # Commit on Enter (editingFinished) instead of moving geometry on every
        # keystroke while typing.
        self.ed_x.setKeyboardTracking(False)
        self.ed_x.setEnabled(False)
        self.lbl_x_mm = QLabel(self._tr("point.x"))
        form.addRow(self.lbl_x_mm, self.ed_x)

        self.lbl_len = QLabel("-")
        self.lbl_member_len = QLabel(self._tr("member_length"))
        form.addRow(self.lbl_member_len, self.lbl_len)

        self.gb_assign = QGroupBox(self._tr("assign_prop"))
        pr.addWidget(self.gb_assign)
        lay_assign = QVBoxLayout(self.gb_assign)
        self.tbl_assign = QTableWidget(0, 5)
        self.tbl_assign.setHorizontalHeaderLabels([
            self._tr("assign.table.id"),
            self._tr("assign.table.len"),
            self._tr("assign.table.mat"),
            self._tr("assign.table.sec"),
            self._tr("assign.table.clr"),
        ])
        self.tbl_assign.verticalHeader().setVisible(False)
        self.tbl_assign.setAlternatingRowColors(True)
        self.tbl_assign.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_assign.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header = self.tbl_assign.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        lay_assign.addWidget(self.tbl_assign)
        self.gb_assign.setVisible(False)

        pr.addStretch(1)

        splitter.setStretchFactor(1, 1)
        splitter.setSizes([180, 1160, 420])
        self._sync_ribbon_button_widths()

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
        self.act_delete.triggered.connect(self.delete_selected_points)
        self.act_undo.triggered.connect(self.undo)
        self.act_redo.triggered.connect(self.redo)

        # --- Libraries ---
        self.act_materials.triggered.connect(self.open_materials)
        self.act_sections.triggered.connect(self.open_sections)
        self.act_assign_prop.triggered.connect(self.show_assign_property_panel)

        # --- Constraints / Loads (quick add) ---
        self.act_add_dx.triggered.connect(self.edit_constraints_selected)
        self.act_add_dy.triggered.connect(self.edit_constraints_selected)
        self.act_add_rz.triggered.connect(self.edit_constraints_selected)
        self.act_add_bush.triggered.connect(self.edit_bushes_selected)

        self.act_add_fy.triggered.connect(self.edit_nodal_loads_selected)
        self.act_add_mz.triggered.connect(self.edit_nodal_loads_selected)
        self.act_add_udl.triggered.connect(self.add_udl_to_selected_members)

        # --- Background ---
        self.act_bg_import.triggered.connect(self.import_background)
        self.act_bg_calibrate.triggered.connect(self.calibrate_background)
        self.act_bg_opacity.triggered.connect(self.set_background_opacity)
        self.act_bg_bw.toggled.connect(self.toggle_background_bw)
        self.act_bg_visible.toggled.connect(self.toggle_background_visible)
        self.act_bg_clear.triggered.connect(self.clear_background)

        # --- Solve / Results ---
        self.act_validate.triggered.connect(self.validate_only)
        self.act_solve.triggered.connect(self.solve_active)
        self.act_show_results.triggered.connect(self.show_results)
        self.act_export_csv.triggered.connect(self.export_results_csv)
        self.act_export_report.triggered.connect(self.export_standard_report)
        self.act_help_pdf.triggered.connect(self.open_help_pdf)
        self.act_about.triggered.connect(self.show_about_dialog)
        self.cmb_language.currentIndexChanged.connect(self._on_language_changed)
        # --- Canvas signals ---
        self.canvas.selection_changed.connect(self.on_selection_changed)
        self.canvas.point_added.connect(self.on_point_added)
        self.canvas.point_moved.connect(self.on_point_moved)
        self.canvas.background_calibration_ready.connect(self._on_bg_calib_ready)
        self.canvas.request_edit_constraints.connect(self.edit_constraints_selected)
        self.canvas.request_edit_bushes.connect(self.edit_bushes_selected)
        self.canvas.request_edit_nodal_loads.connect(self.edit_nodal_loads_selected)
        self.canvas.request_edit_member_udl.connect(self.add_udl_to_selected_members)
        self.canvas.request_delete_selected_points.connect(self.delete_selected_points)
        if self.canvas_xz is not None:
            self.canvas_xz.selection_changed.connect(lambda: None)
            self.canvas.view_state_changed.connect(lambda st: self._sync_canvas_view(self.canvas, self.canvas_xz, st))
            self.canvas_xz.view_state_changed.connect(lambda st: self._sync_canvas_view(self.canvas_xz, self.canvas, st))
        self.tbl_assign.itemSelectionChanged.connect(self._on_assign_table_selection_changed)


    def _on_language_changed(self, _idx: int):
        lang = self.cmb_language.currentData()
        if lang:
            self.current_lang = lang
            self._apply_language()

    def _apply_language(self):
        self._tr = lambda key, **kwargs: tr(self.current_lang, key, **kwargs)
        self.act_new.setText(self._tr("action.new"))
        self.act_open.setText(self._tr("action.open"))
        self.act_save.setText(self._tr("action.save"))
        self.act_select.setText(self._tr("action.select"))
        self.act_add_point.setText(self._tr("action.add_point"))
        self.act_delete.setText(self._tr("action.delete"))
        self.act_materials.setText(self._tr("action.materials"))
        self.act_sections.setText(self._tr("action.sections"))
        self.act_assign_prop.setText(self._tr("action.assign_property"))
        self.act_add_dx.setText(self._tr("action.constraint"))
        self.act_add_bush.setText(self._tr("action.bush"))
        self.act_add_fy.setText(self._tr("action.load"))
        self.act_add_udl.setText(self._tr("action.udl"))
        self.act_bg_import.setText(self._tr("action.import"))
        self.act_bg_calibrate.setText(self._tr("action.calibrate"))
        self.act_bg_opacity.setText(self._tr("action.opacity"))
        self.act_bg_bw.setText(self._tr("action.bw"))
        self._sync_background_visibility_action()
        self.act_bg_clear.setText(self._tr("action.clear"))
        self.act_validate.setText(self._tr("action.validate"))
        self.act_solve.setText(self._tr("action.solve"))
        self.act_show_results.setText(self._tr("action.results"))
        self.act_export_csv.setText(self._tr("action.export_csv"))
        self.act_export_report.setText(self._tr("action.export_report"))
        self.act_help_pdf.setText(self._tr("action.help"))
        self.act_about.setText(self._tr("action.copyright"))
        self.act_undo.setText(self._tr("action.undo"))
        self.act_redo.setText(self._tr("action.redo"))
        self.cmb_language.setItemText(0, self._tr("language.zh"))
        self.cmb_language.setItemText(1, self._tr("language.en"))
        for idx, tab_key in enumerate(self._ribbon_tab_keys):
            self.ribbon.setTabText(idx, self._tr(tab_key))
        for gb in self.ribbon.findChildren(QGroupBox):
            key = gb.property("i18n_key")
            if isinstance(key, str):
                gb.setTitle(self._tr(key))
        self.tree.setHeaderLabels([self._tr("objects")])
        self.xy_label.setText(self._tr("xy_view"))
        if self.xz_label is not None:
            self.xz_label.setText(self._tr("xz_view"))
        self.lbl_sel.setText(self._tr("selection.none"))
        self.gb_point_props.setTitle(self._tr("point_props"))
        self.lbl_x_mm.setText(self._tr("point.x"))
        self.lbl_member_len.setText(self._tr("member_length"))
        self.gb_assign.setTitle(self._tr("assign_prop"))
        self.tbl_assign.setHorizontalHeaderLabels([
            self._tr("assign.table.id"),
            self._tr("assign.table.len"),
            self._tr("assign.table.mat"),
            self._tr("assign.table.sec"),
            self._tr("assign.table.clr"),
        ])
        self.canvas.set_translator(self._tr)
        if self.canvas_xz is not None:
            self.canvas_xz.set_translator(self._tr)
        self.results_view.set_translator(self._tr)
        self._sync_ribbon_button_widths()

    def _sync_ribbon_button_widths(self):
        if not self._ribbon_buttons:
            return
        fm = QFontMetrics(self.font())
        button_keys = [
            "action.new", "action.open", "action.save", "action.select", "action.add_point", "action.delete",
            "action.materials", "action.sections", "action.assign_property", "action.constraint", "action.bush",
            "action.load", "action.udl", "action.import", "action.calibrate", "action.opacity", "action.bw",
            "action.show_bg", "action.hide_bg", "action.clear", "action.validate", "action.solve",
            "action.results", "action.export_csv", "action.export_report", "action.help", "action.copyright",
        ]
        max_text_w = 0
        for key in button_keys:
            for lang in ("zh", "en"):
                max_text_w = max(max_text_w, fm.horizontalAdvance(tr(lang, key)))
        # Keep ribbon buttons more compact: roughly half of previous baseline width.
        fixed_w = max(37, (max_text_w + 28) // 2)
        for btn in self._ribbon_buttons:
            btn.setMinimumWidth(fixed_w)

    def _ensure_spatial_views(self):
        wants_3d = getattr(self.project, "spatial_mode", "2D") == "3D"
        if wants_3d and self.canvas_xz is None:
            self.canvas_xz = BeamCanvas(view_plane="XZ")
            self.canvas_xz.set_translator(self._tr)
            self.canvas_xz.set_mode("readonly")
            self.canvas_xz.set_read_only_display(True)
            self.canvas_xz.setDragMode(self.canvas_xz.DragMode.NoDrag)
            self.canvas_xz.setInteractive(False)
            self.xz_label = QLabel(self._tr("xz_view"))
            self.xz_label.setStyleSheet("color:#666;")
            self.canvas_layout.addWidget(self.xz_label)
            self.canvas_layout.addWidget(self.canvas_xz, 1)
            self.canvas_xz.selection_changed.connect(lambda: None)
            self.canvas.view_state_changed.connect(lambda st: self._sync_canvas_view(self.canvas, self.canvas_xz, st))
            self.canvas_xz.view_state_changed.connect(lambda st: self._sync_canvas_view(self.canvas_xz, self.canvas, st))
        elif (not wants_3d) and self.canvas_xz is not None:
            self.canvas_xz.setParent(None)
            self.canvas_xz.deleteLater()
            self.canvas_xz = None
            if self.xz_label is not None:
                self.xz_label.setParent(None)
                self.xz_label.deleteLater()
                self.xz_label = None

    def _sync_canvas_view(self, source: BeamCanvas, target: BeamCanvas | None, state: dict):
        if target is None:
            return
        if getattr(self, "_syncing_canvas_view", False):
            return
        self._syncing_canvas_view = True
        try:
            target.apply_view_state(state)
        finally:
            self._syncing_canvas_view = False

    def _choose_spatial_mode(self, default_mode: str = "2D") -> str:
        dlg = QDialog(self)
        dlg.setWindowTitle(self._tr("dialog.spatial_mode.title"))
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(self._tr("dialog.spatial_mode.prompt")))
        rb_2d = QRadioButton(self._tr("dialog.spatial_mode.2d"))
        rb_3d = QRadioButton(self._tr("dialog.spatial_mode.3d"))
        rb_2d.setChecked(default_mode != "3D")
        rb_3d.setChecked(default_mode == "3D")
        lay.addWidget(rb_2d)
        lay.addWidget(rb_3d)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return default_mode
        return "3D" if rb_3d.isChecked() else "2D"


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
        self.last_results = None
        self.canvas.clear_support_reactions()
        if self.canvas_xz is not None:
            self.canvas_xz.clear_support_reactions()

    def undo(self):
        d = self.undo_stack.undo()
        if d is None:
            return
        self.project = Project.from_dict(d)
        self.last_results = None
        self.canvas.clear_support_reactions()
        if self.canvas_xz is not None:
            self.canvas_xz.clear_support_reactions()
        self._schedule_refresh()

    def redo(self):
        d = self.undo_stack.redo()
        if d is None:
            return
        self.project = Project.from_dict(d)
        self.last_results = None
        self.canvas.clear_support_reactions()
        if self.canvas_xz is not None:
            self.canvas_xz.clear_support_reactions()
        self._schedule_refresh()

    def repeat_last_model_action(self):
        # Phase-1: switches to continuous Add Point
        self.set_model_mode(self._last_model_mode)

    @staticmethod
    def _round_model_value(value: float) -> float:
        return round(float(value), 1)

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
        self._syncing_canvas_view = False
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
        default_mat_uid = next(iter(self.project.materials.keys()), "")
        default_sec_uid = next(iter(self.project.sections.keys()), "")

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
                    material_uid=default_mat_uid,
                    section_uid=default_sec_uid,
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

    def open_help_pdf(self):
        help_pdf = self._resource_path("help.pdf")
        if not help_pdf.exists():
            self._show_help_text_dialog(f"help.pdf not found\n{help_pdf}")
            return
        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(help_pdf))):
            return
        self._show_help_text_dialog(
            self._tr("msg.help_pdf_fallback")
        )

    def _show_help_text_dialog(self, prefix_message: str = ""):
        help_md = self._resource_path("docs/help.md")
        if not help_md.exists():
            QMessageBox.warning(
                self,
                "Help",
                f"{prefix_message}\n\nhelp.md not found\n{help_md}".strip(),
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("MiniBeam Help")
        dlg.resize(880, 680)

        lay = QVBoxLayout(dlg)
        if prefix_message:
            lay.addWidget(QLabel(prefix_message))
        viewer = QTextBrowser(dlg)
        viewer.setMarkdown(help_md.read_text(encoding="utf-8"))
        lay.addWidget(viewer, 1)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dlg)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec()

    def show_about_dialog(self):
        QMessageBox.information(
            self,
            "Copyright",
            "copyright at 2026 by zwb and jcc",
        )

    def rebuild_members_now(self):
        # Defer rebuild to avoid QGraphicsScene clear while mouse events are active
        self._schedule_rebuild()

    def open_materials(self):
        before = self.project.to_dict()
        dlg = MaterialManagerDialog(self.project, self, lang=self.current_lang)
        if dlg.exec():
            after = self.project.to_dict()
            self._push_snapshot("Edit Materials", before, after)
            self._schedule_refresh()

    def open_sections(self):
        before = self.project.to_dict()
        dlg = SectionManagerDialog(self.project, self, lang=self.current_lang)
        if dlg.exec():
            after = self.project.to_dict()
            self._push_snapshot("Edit Sections", before, after)
            self._schedule_refresh()

    def _assignable_materials(self):
        return list(self.project.materials.values())

    def _assignable_sections(self):
        return list(self.project.sections.values())

    def show_assign_property_panel(self):
        self.gb_assign.setVisible(True)
        self.populate_assign_property_table()

    def populate_assign_property_table(self):
        materials = self._assignable_materials()
        sections = self._assignable_sections()
        mat_items = [(m.uid, m.name) for m in materials]
        sec_items = [(s.uid, s.name) for s in sections]
        color_items = ["#000000", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

        mems = sorted(self.project.members.values(), key=lambda m: m.name)
        selected_member_uid = next(iter(self.canvas.selected_member_uids()), None)
        self.tbl_assign.blockSignals(True)
        self.tbl_assign.setRowCount(len(mems))
        for row, m in enumerate(mems):
            id_item = QTableWidgetItem(m.name or m.uid)
            id_item.setData(Qt.ItemDataRole.UserRole, m.uid)
            self.tbl_assign.setItem(row, 0, id_item)
            xi = self.project.points[m.i_uid].x
            xj = self.project.points[m.j_uid].x
            self.tbl_assign.setItem(row, 1, QTableWidgetItem(f"{abs(xj-xi):.1f}"))

            cmb_mat = QComboBox()
            for uid, name in mat_items:
                cmb_mat.addItem(name, uid)
            idx_mat = max(0, cmb_mat.findData(m.material_uid))
            cmb_mat.setCurrentIndex(idx_mat)
            cmb_mat.currentIndexChanged.connect(lambda _=None, mid=m.uid, r=row: self._on_assign_row_changed(mid, r))
            self.tbl_assign.setCellWidget(row, 2, cmb_mat)

            cmb_sec = QComboBox()
            for uid, name in sec_items:
                cmb_sec.addItem(name, uid)
            idx_sec = max(0, cmb_sec.findData(m.section_uid))
            cmb_sec.setCurrentIndex(idx_sec)
            cmb_sec.currentIndexChanged.connect(lambda _=None, mid=m.uid, r=row: self._on_assign_row_changed(mid, r))
            self.tbl_assign.setCellWidget(row, 3, cmb_sec)

            cmb_color = QComboBox()
            for c in color_items:
                swatch = QPixmap(18, 18)
                swatch.fill(QColor(c))
                cmb_color.addItem(QIcon(swatch), "", c)
            cval = m.color if getattr(m, "color", "") else color_items[0]
            idx_color = max(0, cmb_color.findData(cval))
            cmb_color.setCurrentIndex(idx_color)
            cmb_color.currentIndexChanged.connect(lambda _=None, mid=m.uid, r=row: self._on_assign_row_changed(mid, r))
            self.tbl_assign.setCellWidget(row, 4, cmb_color)
            if selected_member_uid == m.uid:
                self.tbl_assign.selectRow(row)
        self.tbl_assign.setColumnWidth(0, 60)
        self.tbl_assign.setColumnWidth(1, 64)
        self.tbl_assign.setColumnWidth(4, 48)
        self.tbl_assign.blockSignals(False)

    def _on_assign_row_changed(self, member_uid: str, row: int):
        m = self.project.members.get(member_uid)
        if m is None:
            return
        cmb_mat = self.tbl_assign.cellWidget(row, 2)
        cmb_sec = self.tbl_assign.cellWidget(row, 3)
        cmb_color = self.tbl_assign.cellWidget(row, 4)
        before = self.project.to_dict()
        if isinstance(cmb_mat, QComboBox):
            m.material_uid = cmb_mat.currentData() or ""
        if isinstance(cmb_sec, QComboBox):
            m.section_uid = cmb_sec.currentData() or ""
        if isinstance(cmb_color, QComboBox):
            m.color = cmb_color.currentData() or "#000000"
        after = self.project.to_dict()
        self._push_snapshot("Assign Property", before, after)
        self._schedule_refresh()

    def _on_assign_table_selection_changed(self):
        row = self.tbl_assign.currentRow()
        if row < 0:
            return
        id_item = self.tbl_assign.item(row, 0)
        if id_item is None:
            return
        member_uid = id_item.data(Qt.ItemDataRole.UserRole)
        if isinstance(member_uid, str) and member_uid in self.project.members:
            self.canvas.select_member(member_uid)

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
            if "FX" in typ:
                direction = "FX"
            elif "FY" in typ:
                direction = "FY"
            elif "MX" in typ:
                direction = "MX"
            else:
                direction = "MZ"
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
            QMessageBox.information(self, "Validate", self._tr("msg.validate_ok"))
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
            self.canvas.set_support_reactions(out.reactions)
            if self.canvas_xz is not None:
                self.canvas_xz.set_support_reactions(out.reactions)
            QMessageBox.information(self, "Solve", self._tr("msg.solve_ok"))
        except PyniteSolverError as e:
            QMessageBox.critical(self, "PyNite Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Solve Error", f"{type(e).__name__}: {e}")

    def show_results(self):
        if self.last_results is None:
            QMessageBox.information(self, "Results", self._tr("msg.no_results"))
            return
        sp = getattr(self, "sp_def_scale", None)
        def_scale = float(sp.value()) if sp is not None else 1.0
        dlg = ResultsGridDialog(self.project, self.last_results, def_scale=def_scale, parent=self, translator=self._tr)
        dlg.exec()

    # ---------------- Canvas callbacks ----------------

    def export_results_csv(self):
        if self.last_results is None:
            QMessageBox.information(self, "Export CSV", self._tr("msg.no_results"))
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
            N = np.asarray(getattr(out, "N", np.zeros_like(xg)), dtype=float)[mask]
            V = np.asarray(getattr(out, "V", []), dtype=float)[mask]
            M = np.asarray(getattr(out, "M", []), dtype=float)[mask]
            T = np.asarray(getattr(out, "T", []), dtype=float)[mask]
            tau_t = np.asarray(getattr(out, "tau_torsion", []), dtype=float)[mask]
            sigma = np.asarray(getattr(out, "sigma", []), dtype=float)[mask]
            margin = np.asarray(getattr(out, "margin", []), dtype=float)[mask]
            dy = np.asarray(getattr(out, "dy_diag", np.zeros_like(xg)), dtype=float)[mask]
            rz = np.asarray(getattr(out, "rz_diag", np.zeros_like(xg)), dtype=float)[mask]
        else:
            x = np.array([], dtype=float)
            dy = np.array([], dtype=float)
            rz = np.array([], dtype=float)
            N = V = M = T = tau_t = sigma = margin = np.array([], dtype=float)

        import csv
        try:
            with open(fn, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)

                # Unified table header (avoid duplicated header blocks)
                w.writerow(["TYPE", "name", "combo", "x_mm", "dy_mm", "rz_rad", "Rxn_FX_N", "Rxn_FY_N", "Rxn_MZ_Nmm", "Rxn_MX_Nmm", "N_N", "V_N", "M_Nmm", "T_Nmm", "tau_torsion_Nmm2", "sigma_Nmm2", "MS"])

                node_rows = []
                for i, p in enumerate(pts_sorted, start=1):
                    r = out.reactions.get(p.name, {}) if getattr(out, "reactions", None) else {}
                    node_rows.append([
                        "NODE",
                        p.name,
                        "",
                        f"{x_nodes[i-1]:.6f}",
                        f"{dy_nodes[i-1]:.9g}",
                        "",
                        f"{float(r.get('FX', 0.0)):.9g}",
                        f"{float(r.get('FY', 0.0)):.9g}",
                        f"{float(r.get('MZ', 0.0)):.9g}",
                        f"{float(r.get('MX', 0.0)):.9g}",
                        "",
                        "",
                        "",
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
                            f"{rz[i]:.9g}" if i < len(rz) else "",
                            "",
                            "",
                            "",
                            "",
                            f"{N[i]:.9g}" if i < len(N) else "",
                            f"{V[i]:.9g}" if i < len(V) else "",
                            f"{M[i]:.9g}" if i < len(M) else "",
                            f"{T[i]:.9g}" if i < len(T) else "",
                            f"{tau_t[i]:.9g}" if i < len(tau_t) else "",
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
                            merged_row[6] = row[6]
                            merged_row[7] = row[7]
                            merged_row[8] = row[8]
                            merged_row[9] = row[9]
                            diag_rows[idx] = merged_row
                        else:
                            diag_rows.append(row)

                    for row in sorted(diag_rows, key=lambda r: (float(r[3]), 0 if r[0] == "NODE" else 1)):
                        w.writerow(row)
        except Exception as e:
            QMessageBox.critical(self, "Export CSV", self._tr("msg.export_failed", error=e))
            return

        QMessageBox.information(self, "Export CSV", self._tr("msg.export_ok", path=fn))

    def export_standard_report(self):
        if self.last_results is None:
            QMessageBox.information(self, "Export Report", self._tr("msg.no_results"))
            return
        fn, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("action.export_report"),
            "report.html",
            "HTML Files (*.html);;PDF Files (*.pdf)",
        )
        if not fn:
            return

        html = build_standard_report_html(self.project, self.last_results)
        try:
            suffix = Path(fn).suffix.lower()
            if suffix == ".pdf":
                doc = QTextDocument(self)
                doc.setHtml(html)
                printer = QPrinter(QPrinter.PrinterMode.HighResolution)
                printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
                printer.setOutputFileName(fn)
                printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
                # Keep QTextDocument layout width aligned with PDF printable width,
                # otherwise Qt may lay out against an unconstrained page and then
                # uniformly shrink on print, causing visual scale mismatch.
                doc.setDocumentMargin(0)
                doc.setPageSize(printer.pageRect(QPrinter.Unit.Point).size())
                doc.print(printer)
            else:
                Path(fn).write_text(html, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Export Report", self._tr("msg.export_failed", error=e))
            return

        QMessageBox.information(self, "Export Report", self._tr("msg.export_ok", path=fn))

    def on_point_added(self, x: float):
        before = self.project.to_dict()
        from ..core.model import Point
        p = Point(x=self._round_model_value(x))
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
        self.project.points[uid].x = self._round_model_value(new_x)
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
            parts.append(self._tr("selection.points") + ", ".join(self.project.points[uid].name for uid in pids))
        if mids:
            parts.append(self._tr("selection.members") + ", ".join(self.project.members[uid].name for uid in mids))
        self.lbl_sel.setText(self._tr("selection.prefix") + ("; ".join(parts) if parts else self._tr("selection.empty")))

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
            self.lbl_len.setText(f"{abs(xj-xi):.1f} mm")

        if self.gb_assign.isVisible():
            self.tbl_assign.blockSignals(True)
            self.tbl_assign.clearSelection()
            if len(mids) == 1:
                for row in range(self.tbl_assign.rowCount()):
                    id_item = self.tbl_assign.item(row, 0)
                    if id_item is not None and id_item.data(Qt.ItemDataRole.UserRole) == mids[0]:
                        self.tbl_assign.selectRow(row)
                        break
            self.tbl_assign.blockSignals(False)

    def _edit_selected_point_x(self, x: float):
        pids = self.canvas.selected_point_uids()
        if len(pids) != 1:
            return
        self.on_point_moved(pids[0], self._round_model_value(x))

    # ---------------- Refresh ----------------
    def refresh_all(self):
        self.project.rebuild_names()
        self.refresh_tree()
        if getattr(self, "gb_assign", None) is not None and self.gb_assign.isVisible():
            self.populate_assign_property_table()
        # Full sync only when topology changes. For point moves/attribute edits
        # we prefer incremental updates to avoid QGraphicsScene.clear during
        # interactive operations (can hard-crash Qt on Windows).
        try:
            self.canvas.sync(self.project, full=False)
            if self.canvas_xz is not None:
                self.canvas_xz.sync(self.project, full=False)
        except Exception:
            # Fallback to a full rebuild if something unexpected happens.
            self.canvas.sync(self.project, full=True)
            if self.canvas_xz is not None:
                self.canvas_xz.sync(self.project, full=True)

    def refresh_tree(self):
        self.tree.clear()

        assigned_mat_uids = {m.material_uid for m in self.project.members.values() if m.material_uid in self.project.materials}
        assigned_sec_uids = {m.section_uid for m in self.project.members.values() if m.section_uid in self.project.sections}

        root_pts = QTreeWidgetItem([f"{self._tr('tree.points')} ({len(self.project.points)})"])
        root_mem = QTreeWidgetItem([f"{self._tr('tree.members')} ({len(self.project.members)})"])
        root_mats = QTreeWidgetItem([f"{self._tr('tree.materials')} ({len(assigned_mat_uids)})"])
        root_secs = QTreeWidgetItem([f"{self._tr('tree.sections')} ({len(assigned_sec_uids)})"])
        root_constraints = QTreeWidgetItem([self._tr("tree.constraints")])
        root_bushes = QTreeWidgetItem([self._tr("tree.bush")])
        root_loads = QTreeWidgetItem([self._tr("tree.loads")])

        for root in (root_pts, root_mem, root_mats, root_secs, root_constraints, root_bushes, root_loads):
            self.tree.addTopLevelItem(root)

        for p in self.project.sorted_points():
            root_pts.addChild(QTreeWidgetItem([f"{p.name}  x={p.x:.3f}"]))

        mems = sorted(self.project.members.values(), key=lambda m: m.name)
        for m in mems:
            xi = self.project.points[m.i_uid].x
            xj = self.project.points[m.j_uid].x
            mat_name = self.project.materials.get(m.material_uid).name if m.material_uid in self.project.materials else "?"
            sec_name = self.project.sections.get(m.section_uid).name if m.section_uid in self.project.sections else "?"
            root_mem.addChild(QTreeWidgetItem([f"{m.name}  L={abs(xj-xi):.3f}  Mat={mat_name}  Sec={sec_name}"]))

        for mat in sorted((self.project.materials[uid] for uid in assigned_mat_uids), key=lambda x: x.name.lower()):
            root_mats.addChild(QTreeWidgetItem([f"{mat.name} (E={mat.E:.0f}, fy={mat.sigma_y:.1f})"]))

        for sec in sorted((self.project.sections[uid] for uid in assigned_sec_uids), key=lambda x: x.name.lower()):
            root_secs.addChild(QTreeWidgetItem([f"{sec.name} ({sec.type}, A={sec.A:.2f}, Iz={sec.Iz:.2f})"]))

        for p in self.project.sorted_points():
            if p.constraints:
                summary = ", ".join(f"{dof}:{c.value:.3g}" for dof, c in sorted(p.constraints.items()) if getattr(c, "enabled", True))
                if summary:
                    root_constraints.addChild(QTreeWidgetItem([f"{p.name}: {summary}"]))

        for p in self.project.sorted_points():
            bushes = getattr(p, "bushes", {}) or {}
            if bushes:
                summary = ", ".join(f"{dof}:k={b.stiffness:.3g}" for dof, b in sorted(bushes.items()) if getattr(b, "enabled", True))
                if summary:
                    root_bushes.addChild(QTreeWidgetItem([f"{p.name}: {summary}"]))

        active_case = self.project.active_load_case
        for p in self.project.sorted_points():
            nodals = [ld for ld in p.nodal_loads if ld.case == active_case]
            if nodals:
                summary = ", ".join(f"{ld.direction}={ld.value:.3g}" for ld in nodals)
                root_loads.addChild(QTreeWidgetItem([f"{p.name}: {summary} ({active_case})"]))

        for m in mems:
            udls = [ld for ld in m.udl_loads if ld.case == active_case]
            if udls:
                summary = ", ".join(f"{ld.direction}:w1={ld.w1:.3g},w2={ld.w2:.3g}" for ld in udls)
                root_loads.addChild(QTreeWidgetItem([f"{m.name}: {summary} ({active_case})"]))

        self.tree.expandAll()


    # ---------------- File: New/Open/Save (JSON model) ----------------
    def new_project(self):
        # Keep libraries (materials/sections) already loaded; clear model topology.
        chosen_mode = self._choose_spatial_mode(default_mode=self.project.spatial_mode)
        self.project.spatial_mode = chosen_mode
        self._ensure_spatial_views()
        self.project.points.clear()
        self.project.members.clear()
        self.project.rebuild_names()
        self.last_results = None
        self.canvas.clear_support_reactions()
        if self.canvas_xz is not None:
            self.canvas_xz.clear_support_reactions()
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
            project = Project.from_dict(d)
            self.project = project
            self._ensure_spatial_views()
            self._merge_libraries_into_project()
            # Re-bind project to canvas
            self.canvas.project = self.project
            if self.canvas_xz is not None:
                self.canvas_xz.project = self.project
            self.last_results = None
            self.canvas.clear_support_reactions()
            if self.canvas_xz is not None:
                self.canvas_xz.clear_support_reactions()
            self.undo_stack.clear()
            self._schedule_refresh(full=True)
        except Exception as e:
            QMessageBox.critical(self, "Open Failed", str(e))

    # ---------------- Quick add constraints / loads ----------------

    # ---------------- Constraint / Load editors ----------------
    def edit_constraints_selected(self):
        """Open a single dialog to edit nodal constraints for selected point(s)."""
        pids = self.canvas.selected_point_uids()
        if not pids:
            return

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QCheckBox, QWidget, QHBoxLayout, QDoubleSpinBox

        is_3d = getattr(self.project, "spatial_mode", "2D") == "3D"
        dof_order = ["DX", "DY", "DZ", "RX", "RY", "RZ"] if is_3d else ["DX", "DY", "RZ", "RX"]
        dof_labels = {
            "DX": "UX / DX",
            "DY": "UY / DY",
            "DZ": "UZ / DZ",
            "RX": "RX (torsion)",
            "RY": "RY",
            "RZ": "RZ",
        }

        dlg = QDialog(self)
        dlg.setWindowTitle("Constraint")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        lay.addLayout(form)

        def mk_row(label, enabled_default=False, value_default=0.0):
            cb = QCheckBox("Enable")
            cb.setChecked(bool(enabled_default))
            sb = QDoubleSpinBox()
            sb.setRange(-1e12, 1e12)
            sb.setDecimals(3)
            sb.setValue(float(value_default))
            roww = QWidget()
            hl = QHBoxLayout(roww)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(cb)
            hl.addWidget(sb, 1)
            form.addRow(label, roww)
            return cb, sb

        p0 = self.project.points.get(pids[0])
        widgets = {}
        for dof in dof_order:
            enabled, value = (False, 0.0)
            if p0 and dof in p0.constraints:
                enabled, value = (p0.constraints[dof].enabled, p0.constraints[dof].value)
            widgets[dof] = mk_row(dof_labels[dof], enabled, value)

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
            for dof in dof_order:
                cb, sb = widgets[dof]
                if cb.isChecked():
                    p.constraints[dof] = Constraint(dof=dof, value=float(sb.value()), enabled=True)
                else:
                    p.constraints.pop(dof, None)

        after = self.project.to_dict()
        self._push_snapshot("Edit constraints", before, after)
        self._schedule_refresh()

    def edit_bushes_selected(self):
        """Open a dialog to edit nodal spring stiffness for selected point(s)."""
        pids = self.canvas.selected_point_uids()
        if not pids:
            return

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QCheckBox, QWidget, QHBoxLayout, QDoubleSpinBox

        is_3d = getattr(self.project, "spatial_mode", "2D") == "3D"
        dof_order = ["DX", "DY", "DZ", "RX", "RY", "RZ"] if is_3d else ["DX", "DY", "RZ", "RX"]
        dof_labels = {
            "DX": "KX / DX",
            "DY": "KY / DY",
            "DZ": "KZ / DZ",
            "RX": "KRX",
            "RY": "KRY",
            "RZ": "KRZ",
        }

        dlg = QDialog(self)
        dlg.setWindowTitle("Bush")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        lay.addLayout(form)

        def mk_row(label, enabled_default=False, value_default=0.0):
            cb = QCheckBox("Enable")
            cb.setChecked(bool(enabled_default))
            sb = QDoubleSpinBox()
            sb.setRange(0.0, 1e15)
            sb.setDecimals(3)
            sb.setValue(max(0.0, float(value_default)))
            roww = QWidget()
            hl = QHBoxLayout(roww)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(cb)
            hl.addWidget(sb, 1)
            form.addRow(label, roww)
            return cb, sb

        p0 = self.project.points.get(pids[0])
        widgets = {}
        bushes0 = getattr(p0, "bushes", {}) if p0 else {}
        for dof in dof_order:
            enabled, value = (False, 0.0)
            if dof in bushes0:
                enabled, value = (bushes0[dof].enabled, bushes0[dof].stiffness)
            widgets[dof] = mk_row(dof_labels[dof], enabled, value)

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
            bushes = getattr(p, "bushes", None)
            if bushes is None:
                p.bushes = {}
                bushes = p.bushes
            for dof in dof_order:
                cb, sb = widgets[dof]
                if cb.isChecked() and sb.value() > 0:
                    bushes[dof] = Bush(dof=dof, stiffness=float(sb.value()), enabled=True)
                else:
                    bushes.pop(dof, None)

        after = self.project.to_dict()
        self._push_snapshot("Edit bushes", before, after)
        self._schedule_refresh()

    def edit_nodal_loads_selected(self):
        """Open a dialog to edit nodal loads for selected point(s) in active load case."""
        pids = self.canvas.selected_point_uids()
        if not pids:
            return

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QCheckBox, QWidget, QHBoxLayout, QDoubleSpinBox

        is_3d = getattr(self.project, "spatial_mode", "2D") == "3D"
        directions = ["FX", "FY", "FZ", "MX", "MY", "MZ"] if is_3d else ["FX", "FY", "MZ", "MX"]
        direction_labels = {
            "FX": "FX (N)",
            "FY": "FY (N)",
            "FZ": "FZ (N)",
            "MX": "MX (N·mm)",
            "MY": "MY (N·mm)",
            "MZ": "MZ (N·mm)",
        }

        dlg = QDialog(self)
        dlg.setWindowTitle("Load")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        lay.addLayout(form)

        def mk_row(label, enabled_default=False, value_default=0.0):
            cb = QCheckBox("Enable")
            cb.setChecked(bool(enabled_default))
            sb = QDoubleSpinBox()
            sb.setRange(-1e12, 1e12)
            sb.setDecimals(0)
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
        widgets = {}
        for direction in directions:
            enabled, value = (False, 0.0)
            if p0:
                for ld in p0.nodal_loads:
                    if ld.case == lc and ld.direction == direction:
                        enabled, value = (True, ld.value)
                        break
            widgets[direction] = mk_row(direction_labels[direction], enabled, value)

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
            p.nodal_loads = [ld for ld in p.nodal_loads if not (ld.case == lc and ld.direction in directions)]
            for direction in directions:
                cb, sb = widgets[direction]
                if cb.isChecked():
                    p.nodal_loads.append(NodalLoad(direction=direction, value=float(sb.value()), case=lc))

        after = self.project.to_dict()
        self._push_snapshot("Edit loads", before, after)
        self._schedule_refresh()

    def add_udl_to_selected_members(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QDoubleSpinBox
        mids = self.canvas.selected_member_uids()
        if not mids:
            return

        lc = self.project.active_load_case
        w1_default = 0.0
        w2_default = 0.0
        m0 = self.project.members.get(mids[0])
        if m0 is not None:
            for ld in m0.udl_loads:
                if ld.case == lc:
                    w1_default = float(ld.w1)
                    w2_default = float(ld.w2)

        dlg = QDialog(self)
        dlg.setWindowTitle("Distributed Load")
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        lay.addLayout(form)

        sb_w1 = QDoubleSpinBox()
        sb_w1.setRange(-1e12, 1e12)
        sb_w1.setDecimals(0)
        sb_w1.setValue(w1_default)
        form.addRow("w1 @ i-end (N/mm, +Y upward)", sb_w1)

        sb_w2 = QDoubleSpinBox()
        sb_w2.setRange(-1e12, 1e12)
        sb_w2.setDecimals(0)
        sb_w2.setValue(w2_default)
        form.addRow("w2 @ j-end (N/mm, +Y upward)", sb_w2)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        from ..core.model import MemberLoadUDL
        before = self.project.to_dict()
        for mid in mids:
            m = self.project.members.get(mid)
            if m is None:
                continue
            m.udl_loads = [ld for ld in m.udl_loads if ld.case != lc]
            m.udl_loads.append(MemberLoadUDL(direction="Fy", w1=float(sb_w1.value()), w2=float(sb_w2.value()), case=lc))

        after = self.project.to_dict()
        self._push_snapshot("Edit distributed loads", before, after)
        self._schedule_refresh()

    # ---------------- Solve / results wrappers ----------------
    def solve_active(self):
        self.solve_model()



    def _sync_background_visibility_action(self):
        has_bg = getattr(self.canvas, "_bg_item", None) is not None
        visible = bool(has_bg and self.canvas._bg_item.isVisible())
        self.act_bg_visible.blockSignals(True)
        self.act_bg_visible.setChecked(visible if has_bg else True)
        self.act_bg_visible.blockSignals(False)
        self.act_bg_visible.setEnabled(has_bg)
        self.act_bg_visible.setText(self._tr("action.hide_bg") if visible else self._tr("action.show_bg"))

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
        self._sync_background_visibility_action()
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

    def toggle_background_visible(self, on: bool):
        try:
            self.canvas.set_background_visible(bool(on))
            self.act_bg_visible.setText(self._tr("action.hide_bg") if on else self._tr("action.show_bg"))
        except Exception as e:
            QMessageBox.critical(self, "Background", f"Failed to toggle visibility: {e}")

    def clear_background(self):
        self.canvas.clear_background()
        self._sync_background_visibility_action()
        self._schedule_refresh()

    # ---------------- Constraints dialog ----------------
