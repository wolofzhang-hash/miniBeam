from __future__ import annotations

from typing import Final, Literal, Tuple

DOF = Literal["DX", "DY", "DZ", "RX", "RY", "RZ"]
FORCE_DIRECTION = Literal["FX", "FY", "FZ", "MX", "MY", "MZ"]
MEMBER_UDL_DIRECTION = Literal["Fy"]

ALL_DOFS: Final[Tuple[DOF, ...]] = ("DX", "DY", "DZ", "RX", "RY", "RZ")
ALL_FORCE_DIRECTIONS: Final[Tuple[FORCE_DIRECTION, ...]] = ("FX", "FY", "FZ", "MX", "MY", "MZ")

PLANAR_2D_NODE_SUPPORT_DOFS: Final[Tuple[DOF, ...]] = ("DZ", "RY")

