"""Single source of truth for the DsPCBSD+ class list.

YOLO class id = COCO category_id - 1 (verified 2026-10-15 against the annotation files).
"""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = ("SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO")

CLASS_FULL_NAMES: dict[str, str] = {
    "SH": "Short",
    "SP": "Spur",
    "SC": "Spurious copper",
    "OP": "Open",
    "MB": "Mouse bite",
    "HB": "Hole breakout",
    "CS": "Conductor scratch",
    "CFO": "Conductor foreign object",
    "BMFO": "Base material foreign object",
}

NUM_CLASSES = len(CLASS_NAMES)


def yolo_id_to_name(yolo_id: int) -> str:
    return CLASS_NAMES[yolo_id]


def coco_id_to_name(coco_id: int) -> str:
    return CLASS_NAMES[coco_id - 1]


def name_to_yolo_id(name: str) -> int:
    return CLASS_NAMES.index(name)
