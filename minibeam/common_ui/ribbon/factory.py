from __future__ import annotations

from abc import ABC, abstractmethod

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMainWindow

from .registry import ActionRegistry
from .spec import RibbonItem, RibbonSpec


class RibbonFactoryBase(ABC):
    @abstractmethod
    def build(self, mainwindow: QMainWindow, spec: RibbonSpec, registry: ActionRegistry):
        raise NotImplementedError


class PyQtRibbonFactory(RibbonFactoryBase):
    def build(self, mainwindow: QMainWindow, spec: RibbonSpec, registry: ActionRegistry):
        from pyqtribbon import RibbonBar

        ribbonbar = RibbonBar(mainwindow)
        for tab in spec.tabs:
            category = ribbonbar.addCategory(tab.title)
            for group in tab.groups:
                panel = category.addPanel(group.title)
                for item in group.items:
                    self._render_item(panel, item, registry)

        mainwindow.setMenuBar(ribbonbar)

        place_right_area = getattr(mainwindow, "place_ribbon_right_area", None)
        if callable(place_right_area):
            place_right_area()

        return ribbonbar

    def _render_item(self, panel, item: RibbonItem, registry: ActionRegistry):
        if item.kind == "widget":
            panel.addWidget(registry.get_widget(item.key))
            return

        action = registry.get_action(item.key)
        text = item.text_override or action.text()

        if item.kind == "toggle":
            button = self._build_button(panel, text, action.icon(), item.size)
            button.setCheckable(True)
            button.setChecked(action.isChecked())
            button.clicked.connect(action.trigger)
            action.toggled.connect(button.setChecked)
            action.changed.connect(lambda b=button, a=action, t=text: self._sync_button_state(b, a, t))
            self._sync_button_state(button, action, text)
            return

        button = self._build_button(panel, text, action.icon(), item.size)
        button.clicked.connect(action.trigger)
        action.changed.connect(lambda b=button, a=action, t=text: self._sync_button_state(b, a, t))
        self._sync_button_state(button, action, text)

    @staticmethod
    def _build_button(panel, text, icon, size: str):
        if size == "L":
            return panel.addLargeButton(text, icon)
        if size == "M":
            return panel.addMediumButton(text, icon)
        return panel.addSmallButton(text, icon)

    @staticmethod
    def _sync_button_state(button, action: QAction, default_text: str):
        button.setEnabled(action.isEnabled())
        button.setIcon(action.icon())
        button.setToolTip(action.toolTip())
        if action.isCheckable():
            button.setCheckable(True)
            button.setChecked(action.isChecked())
        button.setText(default_text or action.text())
