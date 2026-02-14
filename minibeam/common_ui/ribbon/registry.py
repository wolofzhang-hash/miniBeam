from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QWidget


class ActionRegistry:
    def __init__(self):
        self._actions: dict[str, QAction] = {}
        self._widget_factories: dict[str, Callable[[], QWidget]] = {}

    def register(self, key: str, action: QAction):
        self._actions[key] = action

    def register_widget(self, key: str, factory: Callable[[], QWidget]):
        self._widget_factories[key] = factory

    def get_action(self, key: str) -> QAction:
        if key not in self._actions:
            raise KeyError(f"Unknown action key: {key}")
        return self._actions[key]

    def get_widget(self, key: str) -> QWidget:
        if key not in self._widget_factories:
            raise KeyError(f"Unknown widget key: {key}")
        return self._widget_factories[key]()
