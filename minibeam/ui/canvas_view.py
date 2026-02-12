from __future__ import annotations
from typing import Optional, Dict, List
import math
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem,
    QGraphicsSimpleTextItem, QMenu, QGraphicsPathItem, QGraphicsPixmapItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QTimer
from PyQt6.QtGui import QPen, QAction, QPainterPath, QPolygonF, QBrush, QPixmap, QImage, QTransform, QColor
# sip is used only to detect deleted Qt objects (prevents RuntimeError on Windows).
try:
    from PyQt6 import sip  # type: ignore
except Exception:  # pragma: no cover
    try:
        import sip  # type: ignore
    except Exception:  # pragma: no cover
        sip = None  # type: ignore

def _isdeleted(obj) -> bool:
    if sip is None:
        return False
    fn = getattr(sip, 'isdeleted', None)
    if fn is None:
        return False
    try:
        return bool(fn(obj))
    except Exception:
        return False

from ..core.model import Project


class PointItem(QGraphicsEllipseItem):
    def __init__(self, uid: str, x: float):
        super().__init__(-6, -6, 12, 12)
        self.uid = uid
        self.setBrush(Qt.GlobalColor.white)
        self.setPen(QPen(Qt.GlobalColor.black, 1))
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(10)
        self.setPos(x, 0)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange:
            # lock y to 0
            p: QPointF = value
            return QPointF(p.x(), 0.0)
        return super().itemChange(change, value)


class MemberItem(QGraphicsLineItem):
    def __init__(self, uid: str, x1: float, x2: float):
        super().__init__(x1, 0, x2, 0)
        self.uid = uid
        # Make members visually prominent
        self.setPen(QPen(Qt.GlobalColor.black, 3))
        self.setFlag(QGraphicsLineItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(1)

    def set_color(self, color: str):
        q = QColor(color)
        if not q.isValid():
            q = QColor(Qt.GlobalColor.black)
        self.setPen(QPen(q, 3))


class BeamCanvas(QGraphicsView):
    selection_changed = pyqtSignal()
    point_moved = pyqtSignal(str, float)   # uid, new_x
    point_added = pyqtSignal(float)        # x
    request_delete_selected_points = pyqtSignal()
    request_edit_constraints = pyqtSignal()  # open DX/DY/RZ dialog
    request_edit_nodal_loads = pyqtSignal()  # open FY/MZ dialog
    background_calibration_ready = pyqtSignal(object, object)  # QPointF, QPointF

    def __init__(self):
        super().__init__()
        from PyQt6.QtGui import QPainter
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setSceneRect(-200, -200, 2000, 200)

        self.project: Optional[Project] = None
        self.mode: str = "select"
        self._point_items: Dict[str, PointItem] = {}
        self._member_items: Dict[str, MemberItem] = {}
        self._labels: List[QGraphicsSimpleTextItem] = []

        # Track point positions at mouse-press so we only emit point_moved when
        # a point actually changed X. This prevents a double-click from causing
        # an unnecessary model refresh between the two clicks (which can crash
        # Qt by invalidating scene items mid-event).
        self._press_point_x: Dict[str, float] = {}

        # Interaction guard.
        #
        # IMPORTANT (Windows/Qt): clearing/rebuilding a QGraphicsScene while a
        # nested event loop is running (e.g., during QMenu.exec) or while a
        # mouse interaction is still being processed can cause a hard crash
        # (0xC0000409). We expose a lightweight "interaction depth" so the
        # MainWindow can defer redraws/rebuilds until the interaction ends.
        self._interaction_depth: int = 0

        # Panning (right-button drag)
        self._panning = False
        self._pan_start = None  # type: ignore
        self._pan_moved = False
        self._suppress_context_menu = False

        # Background image
        self._bg_item = None  # QGraphicsPixmapItem
        self._bg_pixmap_src = None  # original QPixmap
        self._bg_gray = False
        self._bg_opacity = 0.35
        self._bg_calib_points = []  # type: List[QPointF]
        self._bg_calib_mode = False

        # Background calibration markers in scene coordinates
        self._bg_calib_markers: List[QGraphicsEllipseItem] = []
        self._bg_calib_world: List[QPointF] = []

        # Background calibration markers (in scene coords)
        self._bg_calib_markers: List[QGraphicsEllipseItem] = []

        # zoom
        self._zoom = 1.0
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

        # Keep focus behavior predictable for interactive tools.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Connect once. Re-connecting on every redraw can lead to duplicated
        # signals (and in rare cases crashes when combined with scene clears).
        self.scene.selectionChanged.connect(self._on_selection_changed)

    def begin_interaction(self):
        self._interaction_depth += 1

    def end_interaction(self):
        self._interaction_depth = max(0, self._interaction_depth - 1)

    def is_interacting(self) -> bool:
        return self._interaction_depth > 0

    def set_mode(self, mode: str):
        self.mode = mode

    def set_project(self, prj: Project):
        self.project = prj

    def sync(self, prj: Project, *, full: bool = False):
        """Synchronize the scene with the given project.

        Why: On Windows, clearing/rebuilding a QGraphicsScene too frequently
        (especially near interactive drags) can still trigger hard crashes.
        For point moves we therefore prefer an incremental update.
        """
        self.project = prj

        if full or (set(self._point_items.keys()) != set(prj.points.keys())) or (set(self._member_items.keys()) != set(prj.members.keys())):
            self._full_rebuild()
        else:
            self._incremental_update()

    def _full_rebuild(self):
        # Caller must ensure this is not executed during an active interaction.
        self.scene.clear()
        self._point_items.clear()
        self._member_items.clear()
        self._labels.clear()

        # baseline reference axis (light dashed)
        pen_axis = QPen(Qt.GlobalColor.lightGray, 1)
        pen_axis.setStyle(Qt.PenStyle.DashLine)
        axis = self.scene.addLine(-1e6, 0, 1e6, 0, pen_axis)
        axis.setZValue(0)

        if not self.project:
            return

        # members first
        for m in self.project.members.values():
            xi = self.project.points[m.i_uid].x
            xj = self.project.points[m.j_uid].x
            it = MemberItem(m.uid, xi, xj)
            it.set_color(getattr(m, "color", "#000000"))
            self.scene.addItem(it)
            self._member_items[m.uid] = it

        # points
        for p in self.project.points.values():
            it = PointItem(p.uid, p.x)
            self.scene.addItem(it)
            self._point_items[p.uid] = it

        # labels/markers
        self._rebuild_labels_and_markers()

    def _incremental_update(self):
        if not self.project:
            return

        # Update point positions
        for uid, p in self.project.points.items():
            it = self._point_items.get(uid)
            if it is not None:
                it.setPos(p.x, 0)

        # Update member geometry
        for mid, m in self.project.members.items():
            it = self._member_items.get(mid)
            if it is None:
                continue
            xi = self.project.points[m.i_uid].x
            xj = self.project.points[m.j_uid].x
            it.setLine(xi, 0, xj, 0)
            it.set_color(getattr(m, "color", "#000000"))

        # Rebuild labels/markers (cheap enough, and avoids stale pointers)
        self._rebuild_labels_and_markers()

    def _rebuild_labels_and_markers(self):
        # Remove existing labels/markers only (keep point/member items).
        for t in self._labels:
            try:
                self.scene.removeItem(t)
            except Exception:
                pass
        self._labels.clear()

        if not self.project:
            return

        # Member length labels
        for m in self.project.members.values():
            xi = self.project.points[m.i_uid].x
            xj = self.project.points[m.j_uid].x
            mid = (xi + xj) / 2.0
            t = QGraphicsSimpleTextItem(f"{abs(xj-xi):.2f}")
            t.setPos(mid - 10, -28)
            t.setBrush(Qt.GlobalColor.darkBlue)
            t.setZValue(5)
            t.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            t.setAcceptHoverEvents(False)
            self.scene.addItem(t)
            self._labels.append(t)

        # Point labels + markers
        for p in self.project.points.values():
            t = QGraphicsSimpleTextItem(p.name or "")
            t.setPos(p.x - 8, 10)
            t.setBrush(Qt.GlobalColor.black)
            t.setZValue(12)
            t.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            t.setAcceptHoverEvents(False)
            self.scene.addItem(t)
            self._labels.append(t)

            if "DY" in p.constraints and p.constraints["DY"].enabled:
                c = QGraphicsSimpleTextItem("UY")
                c.setPos(p.x - 10, -55)
                c.setBrush(Qt.GlobalColor.darkRed)
                c.setZValue(12)
                c.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                c.setAcceptHoverEvents(False)
                self.scene.addItem(c)
                self._labels.append(c)
            if "RZ" in p.constraints and p.constraints["RZ"].enabled:
                c = QGraphicsSimpleTextItem("RZ")
                c.setPos(p.x - 10, -70)
                c.setBrush(Qt.GlobalColor.darkRed)
                c.setZValue(12)
                c.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                c.setAcceptHoverEvents(False)
                self.scene.addItem(c)
                self._labels.append(c)

        # Loads (red arrows / red moments)
        # Scale by max magnitude in active load case, keep a nice pixel size.
        active_case = self.project.active_load_case
        max_fy = 0.0
        max_mz = 0.0
        for p in self.project.points.values():
            for ld in p.nodal_loads:
                if ld.case != active_case:
                    continue
                if ld.direction == "FY":
                    max_fy = max(max_fy, abs(ld.value))
                elif ld.direction == "MZ":
                    max_mz = max(max_mz, abs(ld.value))

        def fy_len(val: float) -> float:
            if max_fy <= 1e-12:
                return 0.0
            L = 55.0 * abs(val) / max_fy
            return max(12.0, min(70.0, L))  # pixels

        def mz_rad(val: float) -> float:
            if max_mz <= 1e-12:
                return 0.0
            R = 22.0 + 28.0 * abs(val) / max_mz
            return max(18.0, min(60.0, R))  # pixels

        red_pen = QPen(Qt.GlobalColor.red, 2)
        # Important: do NOT fill load symbols (especially moments). When a
        # QGraphicsPathItem has a brush, Qt will fill any enclosed area and a
        # moment arc can easily look like a solid red disk.
        no_brush = QBrush(Qt.BrushStyle.NoBrush)

        for p in self.project.points.values():
            # draw each load for active case
            for ld in p.nodal_loads:
                if ld.case != active_case:
                    continue
                if ld.direction == "FY":
                    L = fy_len(ld.value)
                    if L <= 0.5:
                        continue
                    # Arrow in screen pixels (ignore transforms)
                    path = QPainterPath()
                    sign = -1.0 if ld.value > 0 else 1.0  # +FY upward means arrow upward
                    y2 = sign * L
                    path.moveTo(0, 0)
                    path.lineTo(0, y2)
                    head = 6.0
                    # Arrow head should point back toward the shaft.
                    # For upward arrow (sign=-1, y2<0), base is at y2+head.
                    # For downward arrow (sign=+1, y2>0), base is at y2-head.
                    path.moveTo(-head, y2 - sign * head)
                    path.lineTo(0, y2)
                    path.lineTo(head, y2 - sign * head)
                    it = QGraphicsPathItem(path)
                    it.setPen(red_pen)
                    it.setBrush(no_brush)
                    it.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                    # Loads should never block selecting/right-clicking the point underneath.
                    it.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    it.setAcceptHoverEvents(False)
                    it.setZValue(20)
                    it.setPos(p.x, 0)
                    self.scene.addItem(it)
                    self._labels.append(it)

                    txt = QGraphicsSimpleTextItem(f"{ld.value:.1f}")
                    txt.setBrush(Qt.GlobalColor.red)
                    txt.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                    txt.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    txt.setPos(p.x + 8, 0 + y2)
                    txt.setZValue(21)
                    self.scene.addItem(txt)
                    self._labels.append(txt)

                elif ld.direction == "MZ":
                    R = mz_rad(ld.value)
                    if R <= 1.0:
                        continue
                    cw = ld.value < 0
                    # arc from 30deg to 330deg
                    start = 30
                    span = 300 if cw else -300
                    rect = (-R, -R, 2 * R, 2 * R)
                    path = QPainterPath()
                    path.arcMoveTo(*rect, start)
                    path.arcTo(*rect, start, span)
                    # open arrow head at the end of the arc (no fill)
                    end_angle = start + span
                    import math
                    ang = math.radians(end_angle)
                    ex, ey = R * math.cos(ang), -R * math.sin(ang)
                    # Tangent direction at the end point.
                    # For user-friendly visualization we want the arrow head to
                    # point "outward". Depending on Qt's arc angle conventions,
                    # the raw tangent can be flipped, so we reverse it by pi.
                    tang = ang + (-math.pi / 2 if cw else math.pi / 2) + math.pi
                    ah = 7.0
                    left = tang + math.radians(25)
                    right = tang - math.radians(25)
                    path.moveTo(ex, ey)
                    # Arrow head should point outward (along the rotation direction), not inward.
                    path.lineTo(ex + ah * math.cos(left), ey - ah * math.sin(left))
                    path.moveTo(ex, ey)
                    path.lineTo(ex + ah * math.cos(right), ey - ah * math.sin(right))
                    it = QGraphicsPathItem(path)
                    it.setPen(red_pen)
                    it.setBrush(no_brush)
                    it.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                    it.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    it.setAcceptHoverEvents(False)
                    it.setZValue(20)
                    it.setPos(p.x, -30)  # lift a bit
                    self.scene.addItem(it)
                    self._labels.append(it)

                    txt = QGraphicsSimpleTextItem(f"{ld.value:.1f}")
                    txt.setBrush(Qt.GlobalColor.red)
                    txt.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                    txt.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    txt.setPos(p.x + 8, -30 - R)
                    txt.setZValue(21)
                    self.scene.addItem(txt)
                    self._labels.append(txt)


    # ---------------- Background image ----------------
    def set_background(self, pix: QPixmap):
        # Place at z=-10 behind everything
        self._bg_pixmap_src = QPixmap(pix)
        if self._bg_item is None:
            self._bg_item = QGraphicsPixmapItem()
            self._bg_item.setZValue(-10)
            self.scene.addItem(self._bg_item)
        self._bg_item.setVisible(True)
        self._apply_background_pixmap()
        self._bg_item.setOpacity(self._bg_opacity)

    def clear_background(self):
        if self._bg_item is not None:
            try:
                self.scene.removeItem(self._bg_item)
            except Exception:
                pass
        self._bg_item = None
        self._bg_pixmap_src = None
        self._bg_calib_points.clear()
        self._bg_calib_mode = False
        self._clear_bg_calib_markers()

    def _clear_bg_calib_markers(self):
        for it in self._bg_calib_markers:
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        self._bg_calib_markers.clear()

    def set_background_opacity(self, alpha: float):
        self._bg_opacity = max(0.0, min(1.0, float(alpha)))
        if self._bg_item is not None:
            self._bg_item.setOpacity(self._bg_opacity)

    def set_background_grayscale(self, on: bool):
        self._bg_gray = bool(on)
        self._apply_background_pixmap()

    def set_background_visible(self, on: bool):
        if self._bg_item is None:
            return
        self._bg_item.setVisible(bool(on))


    def _apply_background_pixmap(self):
        if self._bg_item is None or self._bg_pixmap_src is None:
            return
        pix = self._bg_pixmap_src
        if self._bg_gray:
            img = pix.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
            pix = QPixmap.fromImage(img)
        self._bg_item.setPixmap(pix)
        self._bg_item.setOffset(0, 0)

    def start_background_calibration(self):
        """Enter calibration mode: user clicks two points on background."""
        self._bg_calib_points = []
        self._bg_calib_mode = True
        # ensure selection isn't confusing
        for it in self._point_items.values():
            try:
                it.setSelected(False)
            except Exception:
                pass

    def _emit_bg_calib_ready(self):
        pts = getattr(self, '_pending_bg_calib', None)
        if pts is None:
            return
        p1, p2 = pts
        self.background_calibration_ready.emit(p1, p2)
        self._pending_bg_calib = None

    def apply_background_calibration(self, p1: QPointF, p2: QPointF, real_dist_mm: float):
        self._apply_background_calibration(p1, p2, real_dist_mm)

    def _apply_background_calibration(self, p1: QPointF, p2: QPointF, real_dist_mm: float):
        if self._bg_item is None:
            return
        # Work in background item's local coordinates to be robust to any
        # previous transforms.
        p1l = self._bg_item.mapFromScene(p1)
        p2l = self._bg_item.mapFromScene(p2)
        dx = p2l.x() - p1l.x()
        dy = p2l.y() - p1l.y()
        pix_dist = (dx * dx + dy * dy) ** 0.5
        if pix_dist <= 1e-9 or real_dist_mm <= 0:
            return
        angle = math.degrees(math.atan2(dy, dx))
        # Rotate by -angle to make the clicked segment horizontal, scale to
        # match real distance, and translate so that the first clicked point
        # lands exactly at the scene origin (0,0). This makes it easy to drag
        # beam points to overlap the calibrated markers.
        s = real_dist_mm / pix_dist
        t = QTransform()
        t.rotate(-angle)
        t.scale(s, s)
        t.translate(-p1l.x(), -p1l.y())
        self._bg_item.setTransform(t, combine=False)

        # Draw calibration markers at (0,0) and (real_dist, 0) in scene.
        self._set_bg_calib_markers(float(real_dist_mm))

    def _set_bg_calib_markers(self, real_dist_mm: float):
        self._clear_bg_calib_markers()
        pen = QPen(Qt.GlobalColor.darkBlue, 2)
        brush = QBrush(Qt.GlobalColor.transparent)
        for x in (0.0, real_dist_mm):
            it = QGraphicsEllipseItem(-6, -6, 12, 12)
            it.setPen(pen)
            it.setBrush(brush)
            it.setZValue(50)
            it.setPos(x, 0.0)
            it.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            self.scene.addItem(it)
            self._bg_calib_markers.append(it)

        # Backwards compatibility for older main_window.py
    def redraw(self):
        self.sync(self.project, full=True)  # type: ignore[arg-type]

    def _on_selection_changed(self):
        self.selection_changed.emit()

    # selection helpers
    def selected_point_uids(self) -> List[str]:
        dead = []
        out: List[str] = []
        for uid, it in list(self._point_items.items()):
            if _isdeleted(it):
                dead.append(uid)
                continue
            try:
                if it.isSelected():
                    out.append(uid)
            except RuntimeError:
                dead.append(uid)
        for uid in dead:
            self._point_items.pop(uid, None)
        return out

    def selected_member_uids(self) -> List[str]:
        dead = []
        out: List[str] = []
        for uid, it in list(self._member_items.items()):
            if _isdeleted(it):
                dead.append(uid)
                continue
            try:
                if it.isSelected():
                    out.append(uid)
            except RuntimeError:
                dead.append(uid)
        for uid in dead:
            self._member_items.pop(uid, None)
        return out

    def select_point(self, uid: str):
        """Select a point item by uid (after redraw)."""
        it = self._point_items.get(uid)
        if it is None:
            return
        for pit in self._point_items.values():
            pit.setSelected(False)
        it.setSelected(True)
        QTimer.singleShot(0, self.selection_changed.emit)

    def select_member(self, uid: str):
        """Select a member item by uid (after redraw)."""
        it = self._member_items.get(uid)
        if it is None:
            return
        for pit in self._point_items.values():
            pit.setSelected(False)
        for mit in self._member_items.values():
            mit.setSelected(False)
        it.setSelected(True)
        QTimer.singleShot(0, self.selection_changed.emit)

    # events
    def mousePressEvent(self, event):
        self.begin_interaction()
        # Keep keyboard focus on canvas after clicking.
        self.setFocus(Qt.FocusReason.MouseFocusReason)

        # Right-button pan
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._pan_start = event.pos()
            self._pan_moved = False
            event.accept()
            return

        # Background calibration click
        if event.button() == Qt.MouseButton.LeftButton and self._bg_calib_mode and self._bg_item is not None:
            pos = self.mapToScene(event.position().toPoint())
            self._bg_calib_points.append(QPointF(pos.x(), pos.y()))
            # tiny marker
            mk = self.scene.addEllipse(pos.x()-3, pos.y()-3, 6, 6, QPen(Qt.GlobalColor.red, 1), QBrush(Qt.GlobalColor.red))
            mk.setZValue(50)
            self._labels.append(mk)
            if len(self._bg_calib_points) >= 2:
                self._bg_calib_mode = False
                # Ask MainWindow to request distance via signal
                self.end_interaction()
                self.selection_changed.emit()  # no-op but keeps UI in sync
                self._pending_bg_calib = (self._bg_calib_points[0], self._bg_calib_points[1])  # type: ignore
                # Emit a custom Qt event via QTimer for main window to query
                QTimer.singleShot(0, lambda: self._emit_bg_calib_ready())
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.mode == "add_point":
            pos = self.mapToScene(event.position().toPoint())
            self.point_added.emit(float(pos.x()))
            event.accept()
            self.end_interaction()
            return

        # Let the base class update selection first, then snapshot point x positions for move detection.
        super().mousePressEvent(event)

        self._press_point_x = {}
        try:
            for uid, it in list(self._point_items.items()):
                if _isdeleted(it):
                    continue
                if it.isSelected():
                    self._press_point_x[uid] = float(it.pos().x())
            if not self._press_point_x and event.button() == Qt.MouseButton.LeftButton:
                sp = self.mapToScene(event.position().toPoint())
                for it in self.scene.items(QPointF(sp.x(), sp.y())):
                    if isinstance(it, PointItem):
                        self._press_point_x[it.uid] = float(it.pos().x())
                        break
        except Exception:
            pass

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            self._pan_moved = True
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # End panning
        if event.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            # If mouse moved, suppress context menu
            if self._pan_moved:
                self._suppress_context_menu = True
            self._pan_start = None
            self._pan_moved = False
            self.end_interaction()
            event.accept()
            return

        # Emit point_moved only if X truly changed
        super().mouseReleaseEvent(event)
        try:
            moved = False
            for uid, oldx in self._press_point_x.items():
                it = self._point_items.get(uid)
                if it is None:
                    continue
                newx = float(it.pos().x())
                if abs(newx - oldx) > 1e-6:
                    self.point_moved.emit(uid, newx)
                    moved = True
            self._press_point_x = {}
        finally:
            self.end_interaction()

    def wheelEvent(self, event):
        # Wheel zoom (CAD style). Zoom around mouse.
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom *= factor
        self.scale(factor, factor)
        event.accept()

    def drawForeground(self, painter, rect):
        # Simple axes + rulers in screen space
        super().drawForeground(painter, rect)
        painter.save()
        painter.resetTransform()
        w = self.viewport().width()
        h = self.viewport().height()
        pen = QPen(Qt.GlobalColor.gray, 1)
        painter.setPen(pen)

        # Axes indicator bottom-left
        ox, oy = 40, h - 40
        painter.drawLine(ox, oy, ox + 60, oy)      # +X
        painter.drawLine(ox, oy, ox, oy - 60)      # +Y
        painter.drawText(ox + 64, oy + 4, "X")
        painter.drawText(ox - 10, oy - 64, "Y")

        # Ruler along bottom (x)
        # Determine tick spacing from current view scale
        # map 100 scene units to screen pixels
        p0 = self.mapFromScene(0, 0)
        p100 = self.mapFromScene(100, 0)
        px_per_100 = max(1, abs(p100.x() - p0.x()))
        # choose nice spacing
        spacing_scene = 100
        if px_per_100 < 40:
            spacing_scene = 500
        elif px_per_100 < 80:
            spacing_scene = 200
        elif px_per_100 > 200:
            spacing_scene = 50

        # visible scene rect
        vis = self.mapToScene(self.viewport().rect()).boundingRect()
        x_start = int(vis.left() // spacing_scene * spacing_scene)
        x_end = int(vis.right() // spacing_scene * spacing_scene + spacing_scene)
        y_r = h - 18
        for x in range(x_start, x_end + 1, spacing_scene):
            pt = self.mapFromScene(x, 0)
            painter.drawLine(pt.x(), y_r, pt.x(), y_r - 6)
            painter.drawText(pt.x() + 2, y_r - 8, f"{x:g}")

        painter.restore()

    # NOTE: mouseMoveEvent/mouseReleaseEvent are defined once above.
    # A previous refactor accidentally duplicated these handlers which caused
    # right-button panning to get "stuck" (the second mouseReleaseEvent
    # implementation didn't clear the _panning flag). Keep a single canonical
    # implementation to avoid Qt state corruption.

    def contextMenuEvent(self, event):
        if getattr(self, '_suppress_context_menu', False):
            self._suppress_context_menu = False
            event.accept()
            return

        # QMenu.exec runs a nested event loop. We must block scene rebuilds
        # until the menu closes to avoid hard crashes on Windows.
        self.begin_interaction()
        menu = QMenu(self)
        menu.aboutToHide.connect(self.end_interaction)

        scene_pos = self.mapToScene(event.pos())
        # itemAt() returns the top-most graphics item. Load symbols (moments)
        # can sit above points visually; we still want the point to be the
        # target for right-click actions. Therefore search for a PointItem
        # under the cursor explicitly.
        item = None
        for it in self.items(event.pos()):
            if isinstance(it, PointItem):
                item = it
                break

        # If right-clicked on a point, select it (single selection) so that
        # subsequent actions apply to the intended point.
        if isinstance(item, PointItem):
            for it in self._point_items.values():
                it.setSelected(False)
            item.setSelected(True)
            # Emit selection change on the next tick (Qt doesn't always emit
            # synchronously for programmatic selection changes).
            QTimer.singleShot(0, self.selection_changed.emit)            # Point context menu (edit, do not create duplicates)
            a_c = QAction("Constraint...", self)
            a_c.triggered.connect(lambda: self.request_edit_constraints.emit())
            menu.addAction(a_c)

            a_l = QAction("Load...", self)
            a_l.triggered.connect(lambda: self.request_edit_nodal_loads.emit())
            menu.addAction(a_l)

            menu.addSeparator()
            a_del = QAction("Delete Point", self)
            a_del.triggered.connect(lambda: self.request_delete_selected_points.emit())
            menu.addAction(a_del)

        else:
            # Blank area context menu
            a_add = QAction("Add Point Here", self)
            a_add.triggered.connect(lambda: self.point_added.emit(float(scene_pos.x())))
            menu.addAction(a_add)

            if self.selected_point_uids():
                a_del = QAction("Delete Selected Point(s)", self)
                a_del.triggered.connect(lambda: self.request_delete_selected_points.emit())
                menu.addAction(a_del)

            # Intentionally no "Rebuild" item here in Phase-1. Rebuild is
            # available on the ribbon, while right-click is reserved for quick
            # delete actions per the user's workflow.

        menu.exec(event.globalPos())
