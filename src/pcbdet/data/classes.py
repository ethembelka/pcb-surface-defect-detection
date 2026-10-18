"""Single source of truth for the DsPCBSD+ class list.

Why a separate module: the class order appears in several places (Ultralytics yaml, COCO json,
confusion-matrix axes, report tables). If each place kept its own copy, one of them would
eventually drift and the per-class numbers would be silently wrong. Everything imports from here.

Id conventions (verified 2026-10-15 against the annotation files):
- YOLO label files use ids 0..8 in the order below.
- COCO json uses category_id 1..9 in the same order, i.e. YOLO id = COCO id - 1.
"""

from __future__ import annotations

# Order matters: index = YOLO class id.
CLASS_NAMES: tuple[str, ...] = ("SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO")

# Long names as used in the dataset paper (Lv et al., 2024), for tables and figure legends.
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
