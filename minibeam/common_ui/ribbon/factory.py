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
            button = self._build_button(panel, text, action.icon())
            button.setCheckable(True)
            button.setChecked(action.isChecked())
            button.clicked.connect(action.trigger)
            action.toggled.connect(button.setChecked)
            self._bind_action_changed(button, action, text)
            return

        button = self._build_button(panel, text, action.icon())
        button.clicked.connect(action.trigger)
        self._bind_action_changed(button, action, text)

    def _bind_action_changed(self, button, action: QAction, default_text: str):
        def on_action_changed():
            try:
                self._sync_button_state(button, action, default_text)
            except RuntimeError:
                try:
                    action.changed.disconnect(on_action_changed)
                except (RuntimeError, TypeError):
                    pass

        action.changed.connect(on_action_changed)
        on_action_changed()

    @staticmethod
    def _build_button(panel, text, icon):
        return panel.addLargeButton(text, icon)

    @staticmethod
    def _sync_button_state(button, action: QAction, default_text: str):
        button.setEnabled(action.isEnabled())
        button.setIcon(action.icon())
        button.setToolTip(action.toolTip())
        if action.isCheckable():
            button.setCheckable(True)
            button.setChecked(action.isChecked())
        button.setText(default_text or action.text())
