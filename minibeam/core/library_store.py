from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .model import Material, Section


def _lib_dir() -> Path:
    # A simple per-user library folder.
    # Windows: C:\Users\<user>\.minibeam
    # macOS/Linux: ~/.minibeam
    d = Path.home() / ".minibeam"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_material_library() -> List[Material]:
    path = _lib_dir() / "materials.json"
    data = _read_json(path)
    out: List[Material] = []
    for md in data.get("materials", []):
        try:
            # Create with a new uid to avoid collisions.
            md = dict(md)
            md.pop("uid", None)
            out.append(Material(**md))
        except Exception:
            continue
    return out


def save_material_library(materials: Dict[str, Material]) -> None:
    path = _lib_dir() / "materials.json"
    payload = {"materials": [asdict(m) for m in materials.values()]}
    _write_json(path, payload)


def load_section_library() -> List[Section]:
    path = _lib_dir() / "sections.json"
    data = _read_json(path)
    out: List[Section] = []
    for sd in data.get("sections", []):
        try:
            sd = dict(sd)
            sd.pop("uid", None)
            out.append(Section(**sd))
        except Exception:
            continue
    return out


def save_section_library(sections: Dict[str, Section]) -> None:
    path = _lib_dir() / "sections.json"
    payload = {"sections": [asdict(s) for s in sections.values()]}
    _write_json(path, payload)


def merge_by_name(existing: Dict[str, object], incoming: List[object], name_attr: str = "name") -> None:
    """Merge items into the dict by name, skipping duplicates."""
    existing_names = {getattr(v, name_attr, "") for v in existing.values()}
    for obj in incoming:
        n = getattr(obj, name_attr, "")
        if not n or n in existing_names:
            continue
        existing[getattr(obj, "uid")] = obj
        existing_names.add(n)
