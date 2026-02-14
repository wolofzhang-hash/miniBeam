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

    def addCategory(self, _title):
        raise AssertionError("No categories should be added in this test")

    def setCornerWidget(self, _widget):
        raise AssertionError("setCornerWidget must not be called")


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
