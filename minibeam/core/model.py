from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Any
import uuid

Units = Literal["mm-N-Nmm"]

def _uid() -> str:
    return uuid.uuid4().hex[:10]

@dataclass
class Constraint:
    # DOF in PyNite naming (Phase-1+): DX, DY, RZ, RX (torsion about beam axis)
    dof: Literal["DX", "DY", "RZ", "RX"]
    value: float = 0.0
    enabled: bool = True

@dataclass
class NodalLoad:
    # direction in PyNite naming (Phase-1+): FY, MZ, MX (torsion moment about beam axis)
    direction: Literal["FY", "MZ", "MX"]
    value: float
    case: str = "LC1"

@dataclass
class MemberLoadUDL:
    # direction in PyNite naming: Fy
    direction: Literal["Fy"] = "Fy"
    w: float = 0.0  # N/mm (positive +Y)
    case: str = "LC1"

@dataclass
class Point:
    uid: str = field(default_factory=_uid)
    name: str = ""
    x: float = 0.0  # mm
    constraints: Dict[str, Constraint] = field(default_factory=dict)  # key dof
    nodal_loads: List[NodalLoad] = field(default_factory=list)

@dataclass
class Material:
    uid: str = field(default_factory=_uid)
    name: str = "Steel"
    E: float = 210000.0  # N/mm^2
    G: float = 81000.0   # N/mm^2
    nu: float = 0.3
    rho: float = 7.85e-6 # tonne/mm^3 equiv (not used)
    sigma_y: float = 355.0  # N/mm^2

@dataclass
class Section:
    uid: str = field(default_factory=_uid)
    name: str = "Rect100x10"
    type: str = "RectSolid"  # RectSolid, CircleSolid, ISection
    # properties (mm units)
    A: float = 1.0
    Iy: float = 1.0
    Iz: float = 1.0
    J: float = 1.0
    c_z: float = 1.0  # max distance for bending about z (Mz -> stress uses c_z with Iz)

    # Wizard parameters (for sketch/preview). Units: mm.
    p1: float = 0.0
    p2: float = 0.0
    p3: float = 0.0
    p4: float = 0.0

@dataclass
class Member:
    uid: str = field(default_factory=_uid)
    name: str = ""
    i_uid: str = ""
    j_uid: str = ""
    material_uid: str = ""
    section_uid: str = ""
    color: str = "#000000"
    udl_loads: List[MemberLoadUDL] = field(default_factory=list)

@dataclass
class LoadCombo:
    name: str = "COMB1"
    factors: Dict[str, float] = field(default_factory=lambda: {"LC1": 1.0})

@dataclass
class Project:
    units: Units = "mm-N-Nmm"
    points: Dict[str, Point] = field(default_factory=dict)  # uid->Point
    members: Dict[str, Member] = field(default_factory=dict)  # uid->Member
    materials: Dict[str, Material] = field(default_factory=dict)
    sections: Dict[str, Section] = field(default_factory=dict)
    auto_members: bool = True
    load_cases: List[str] = field(default_factory=lambda: ["LC1"])
    active_load_case: str = "LC1"
    combos: Dict[str, LoadCombo] = field(default_factory=lambda: {"COMB1": LoadCombo()})
    active_combo: str = "COMB1"
    safety_factor: float = 1.5

    def sorted_points(self) -> List[Point]:
        return sorted(self.points.values(), key=lambda p: (p.x, p.name))

    def rebuild_names(self):
        pts = self.sorted_points()
        for i, p in enumerate(pts, start=1):
            p.name = f"P{i}"
        # members renamed by order
        mems = list(self.members.values())
        mems_sorted = sorted(mems, key=lambda m: (self.points[m.i_uid].x, self.points[m.j_uid].x))
        for i, m in enumerate(mems_sorted, start=1):
            m.name = f"M{i}"


    # ---------------- Convenience helpers ----------------
    def set_constraint(self, point_uid: str, dof: str, enabled: bool, value: float = 0.0):
        """Set or clear a constraint on a point. Unique per DOF."""
        p = self.points.get(point_uid)
        if p is None:
            return
        if not enabled:
            if dof in p.constraints:
                del p.constraints[dof]
            return
        
        p.constraints[dof] = Constraint(dof=dof, value=float(value), enabled=True)

    def get_constraint(self, point_uid: str, dof: str):
        p = self.points.get(point_uid)
        if p is None:
            return (False, 0.0)
        c = p.constraints.get(dof)
        if c is None or not c.enabled:
            return (False, 0.0)
        return (True, float(c.value))

    def set_nodal_load(self, point_uid: str, direction: str, value: float, case: str = "LC1", enabled: bool = True):
        """Set (replace) a nodal load for a given direction+case. If disabled, remove it."""
        p = self.points.get(point_uid)
        if p is None:
            return
        # remove any existing matching loads
        p.nodal_loads = [ld for ld in p.nodal_loads if not (ld.direction == direction and ld.case == case)]
        if not enabled:
            return
        
        p.nodal_loads.append(NodalLoad(direction=direction, value=float(value), case=case))

    def get_nodal_load(self, point_uid: str, direction: str, case: str = "LC1"):
        p = self.points.get(point_uid)
        if p is None:
            return (False, 0.0)
        for ld in p.nodal_loads:
            if ld.direction == direction and ld.case == case:
                return (True, float(ld.value))
        return (False, 0.0)
    # ---------------- Serialization ----------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire project to a JSON-friendly dict.

        We use this for simple, robust Undo/Redo (snapshot-based) in Phase-1.
        The project sizes are small, so full snapshots are acceptable.
        """
        return {
            "units": self.units,
            "auto_members": self.auto_members,
            "load_cases": list(self.load_cases),
            "active_load_case": self.active_load_case,
            "active_combo": self.active_combo,
            "safety_factor": self.safety_factor,
            "points": {k: asdict(v) for k, v in self.points.items()},
            "members": {k: asdict(v) for k, v in self.members.items()},
            "materials": {k: asdict(v) for k, v in self.materials.items()},
            "sections": {k: asdict(v) for k, v in self.sections.items()},
            "combos": {k: asdict(v) for k, v in self.combos.items()},
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Project":
        """Restore a project from :meth:`to_dict`."""
        prj = Project()
        prj.units = d.get("units", prj.units)
        prj.auto_members = bool(d.get("auto_members", prj.auto_members))
        prj.load_cases = list(d.get("load_cases", prj.load_cases))
        prj.active_load_case = d.get("active_load_case", prj.active_load_case)
        prj.active_combo = d.get("active_combo", prj.active_combo)
        prj.safety_factor = float(d.get("safety_factor", prj.safety_factor))

        prj.points = {}
        for uid, pd in d.get("points", {}).items():
            p = Point(**pd)
            prj.points[uid] = p

        prj.members = {}
        for uid, md in d.get("members", {}).items():
            m = Member(**md)
            prj.members[uid] = m

        prj.materials = {}
        for uid, md in d.get("materials", {}).items():
            mat = Material(**md)
            prj.materials[uid] = mat

        prj.sections = {}
        for uid, sd in d.get("sections", {}).items():
            sec = Section(**sd)
            prj.sections[uid] = sec

        prj.combos = {}
        for name, cd in d.get("combos", {}).items():
            prj.combos[name] = LoadCombo(**cd)

        prj.rebuild_names()
        return prj

    # ---------------- Serialization (for undo/redo & save) ----------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire project to a JSON-safe dict.

        Phase-1 uses this for undo/redo snapshots (fast enough for small models).
        """
        return {
            "units": self.units,
            "auto_members": self.auto_members,
            "load_cases": list(self.load_cases),
            "active_load_case": self.active_load_case,
            "active_combo": self.active_combo,
            "safety_factor": self.safety_factor,
            "points": {uid: asdict(p) for uid, p in self.points.items()},
            "members": {uid: asdict(m) for uid, m in self.members.items()},
            "materials": {uid: asdict(m) for uid, m in self.materials.items()},
            "sections": {uid: asdict(s) for uid, s in self.sections.items()},
            "combos": {name: asdict(c) for name, c in self.combos.items()},
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Project":
        """Restore a project from :meth:`to_dict` output."""
        prj = Project()
        prj.units = d.get("units", prj.units)
        prj.auto_members = bool(d.get("auto_members", prj.auto_members))
        prj.load_cases = list(d.get("load_cases", prj.load_cases))
        prj.active_load_case = d.get("active_load_case", prj.active_load_case)
        prj.active_combo = d.get("active_combo", prj.active_combo)
        prj.safety_factor = float(d.get("safety_factor", prj.safety_factor))

        # Restore materials/sections first
        prj.materials = {}
        for uid, md in d.get("materials", {}).items():
            prj.materials[uid] = Material(**md)
        prj.sections = {}
        for uid, sd in d.get("sections", {}).items():
            prj.sections[uid] = Section(**sd)

        prj.points = {}
        for uid, pd in d.get("points", {}).items():
            # Convert nested dataclasses
            pd = dict(pd)
            pd["constraints"] = {k: Constraint(**cd) for k, cd in pd.get("constraints", {}).items()}
            pd["nodal_loads"] = [NodalLoad(**ld) for ld in pd.get("nodal_loads", [])]
            prj.points[uid] = Point(**pd)

        prj.members = {}
        for uid, md in d.get("members", {}).items():
            md = dict(md)
            md["udl_loads"] = [MemberLoadUDL(**ld) for ld in md.get("udl_loads", [])]
            prj.members[uid] = Member(**md)

        prj.combos = {}
        for name, cd in d.get("combos", {}).items():
            prj.combos[name] = LoadCombo(**cd)

        prj.rebuild_names()
        return prj
