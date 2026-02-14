from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RibbonItemKind = Literal["action", "toggle", "widget"]


@dataclass(slots=True)
class RibbonItem:
    key: str
    kind: RibbonItemKind
    text_override: str | None = None


@dataclass(slots=True)
class RibbonGroup:
    title: str
    items: list[RibbonItem] = field(default_factory=list)


@dataclass(slots=True)
class RibbonTab:
    title: str
    groups: list[RibbonGroup] = field(default_factory=list)


@dataclass(slots=True)
class RibbonSpec:
    tabs: list[RibbonTab] = field(default_factory=list)
