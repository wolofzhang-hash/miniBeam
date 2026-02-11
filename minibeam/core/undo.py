from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class Snapshot:
    label: str
    before: Dict[str, Any]
    after: Dict[str, Any]


class UndoStack:
    """A tiny snapshot-based undo stack.

    For Phase-1 MiniBeam, the model is small, so snapshotting the whole
    Project dict is the most robust and fastest way to deliver Ctrl+Z/Ctrl+Y
    without edge cases.
    """

    def __init__(self):
        self._stack: List[Snapshot] = []
        self._index: int = -1  # points to current snapshot (after applied)

    def clear(self) -> None:
        self._stack.clear()
        self._index = -1

    def can_undo(self) -> bool:
        return self._index >= 0

    def can_redo(self) -> bool:
        return self._index < (len(self._stack) - 1)

    def push(self, label: str, before: Dict[str, Any], after: Dict[str, Any]) -> None:
        # Drop any redo history
        if self._index < (len(self._stack) - 1):
            self._stack = self._stack[: self._index + 1]
        self._stack.append(Snapshot(label=label, before=before, after=after))
        self._index = len(self._stack) - 1

    def undo(self) -> Optional[Dict[str, Any]]:
        if not self.can_undo():
            return None
        snap = self._stack[self._index]
        self._index -= 1
        return snap.before

    def redo(self) -> Optional[Dict[str, Any]]:
        if not self.can_redo():
            return None
        self._index += 1
        snap = self._stack[self._index]
        return snap.after

    def peek_undo_label(self) -> str:
        if not self.can_undo():
            return ""
        return self._stack[self._index].label

    def peek_redo_label(self) -> str:
        if not self.can_redo():
            return ""
        return self._stack[self._index + 1].label
