import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _stub_pyqt_modules(monkeypatch):
    qtgui = types.SimpleNamespace(QAction=type("QAction", (), {}))
    qtwidgets = types.SimpleNamespace(QMainWindow=type("QMainWindow", (), {}), QWidget=type("QWidget", (), {}))
    pyqt6 = types.SimpleNamespace(QtGui=qtgui, QtWidgets=qtwidgets)

    monkeypatch.setitem(sys.modules, "PyQt6", pyqt6)
    monkeypatch.setitem(sys.modules, "PyQt6.QtGui", qtgui)
    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", qtwidgets)


class _FakeRibbonBar:
    def __init__(self, parent):
        self.parent = parent
        self.stylesheet = None
        self.min_height = None
        self.max_height = None

    def addCategory(self, _title):
        raise AssertionError("No categories should be added in this test")

    def setCornerWidget(self, _widget):
        raise AssertionError("setCornerWidget must not be called")

    def setStyleSheet(self, stylesheet):
        self.stylesheet = stylesheet

    def setMinimumHeight(self, height):
        self.min_height = height

    def setMaximumHeight(self, height):
        self.max_height = height


class _FakeMainWindow:
    def __init__(self):
        self.menubar = None
        self.right_area_calls = 0

    def setMenuBar(self, menubar):
        self.menubar = menubar

    def place_ribbon_right_area(self):
        self.right_area_calls += 1


def test_build_skips_corner_widget_and_calls_optional_right_area_hook(monkeypatch):
    from minibeam.common_ui.ribbon.factory import PyQtRibbonFactory
    from minibeam.common_ui.ribbon.registry import ActionRegistry
    from minibeam.common_ui.ribbon.spec import RibbonSpec

    monkeypatch.setitem(
        sys.modules,
        "pyqtribbon",
        types.SimpleNamespace(RibbonBar=_FakeRibbonBar),
    )

    mainwindow = _FakeMainWindow()
    ribbon = PyQtRibbonFactory().build(mainwindow, RibbonSpec(tabs=[]), ActionRegistry())

    assert isinstance(ribbon, _FakeRibbonBar)
    assert mainwindow.menubar is ribbon
    assert mainwindow.right_area_calls == 1
    assert "font-size: 11px" in ribbon.stylesheet
    assert ribbon.min_height == 78
    assert ribbon.max_height == 92


def test_build_without_optional_right_area_hook(monkeypatch):
    from minibeam.common_ui.ribbon.factory import PyQtRibbonFactory
    from minibeam.common_ui.ribbon.registry import ActionRegistry
    from minibeam.common_ui.ribbon.spec import RibbonSpec

    monkeypatch.setitem(
        sys.modules,
        "pyqtribbon",
        types.SimpleNamespace(RibbonBar=_FakeRibbonBar),
    )

    class _NoHookMainWindow:
        def __init__(self):
            self.menubar = None

        def setMenuBar(self, menubar):
            self.menubar = menubar

    mainwindow = _NoHookMainWindow()
    ribbon = PyQtRibbonFactory().build(mainwindow, RibbonSpec(tabs=[]), ActionRegistry())

    assert isinstance(ribbon, _FakeRibbonBar)
    assert mainwindow.menubar is ribbon


class _FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def disconnect(self, callback):
        self._callbacks.remove(callback)

    def emit(self):
        for callback in list(self._callbacks):
            callback()


class _FakeActionForChanged:
    def __init__(self):
        self.changed = _FakeSignal()

    def isEnabled(self):
        return True

    def icon(self):
        return None

    def toolTip(self):
        return ""

    def isCheckable(self):
        return False

    def text(self):
        return "Action"


class _FakeButtonForChanged:
    def __init__(self):
        self.deleted = False
        self.sync_calls = 0

    def setEnabled(self, _enabled):
        if self.deleted:
            raise RuntimeError("wrapped C/C++ object has been deleted")
        self.sync_calls += 1

    def setIcon(self, _icon):
        pass

    def setToolTip(self, _tip):
        pass

    def setText(self, _text):
        pass


def test_action_changed_auto_disconnects_after_button_deleted():
    from minibeam.common_ui.ribbon.factory import PyQtRibbonFactory

    factory = PyQtRibbonFactory()
    action = _FakeActionForChanged()
    button = _FakeButtonForChanged()

    factory._bind_action_changed(button, action, "Action")
    assert button.sync_calls == 1

    button.deleted = True
    action.changed.emit()
    action.changed.emit()

    assert len(action.changed._callbacks) == 0


class _FakeButtonForWidth:
    def __init__(self):
        self.min_width = None
        self.max_width = None

    def setMinimumWidth(self, width):
        self.min_width = width

    def setMaximumWidth(self, width):
        self.max_width = width


class _FakePanelForWidth:
    def addLargeButton(self, _text, _icon):
        return _FakeButtonForWidth()

    def addMediumButton(self, _text, _icon):
        return _FakeButtonForWidth()

    def addSmallButton(self, _text, _icon):
        return _FakeButtonForWidth()


def test_build_button_applies_uniform_width_for_all_sizes():
    from minibeam.common_ui.ribbon.factory import PyQtRibbonFactory

    panel = _FakePanelForWidth()
    for size in ("L", "M", "S"):
        button = PyQtRibbonFactory._build_button(panel, "T", None, size)
        assert button.min_width == PyQtRibbonFactory.UNIFORM_BUTTON_WIDTH
        assert button.max_width == PyQtRibbonFactory.UNIFORM_BUTTON_WIDTH


def test_main_ribbon_spec_uses_large_icons_for_interactive_items():
    from minibeam.ui.ribbon_setup import _build_spec

    class _MW:
        _tr = staticmethod(lambda key, **kwargs: key)

    spec = _build_spec(_MW())
    sizes = [
        item.size
        for tab in spec.tabs
        for group in tab.groups
        for item in group.items
        if item.kind in {"action", "toggle"}
    ]

    assert sizes
    assert set(sizes) == {"L"}
