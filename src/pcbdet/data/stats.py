"""Dataset statistics: how many boxes of each class fall into each COCO size bucket.

Why
---
Research question 2 asks whether accuracy depends on defect *type* or defect *size*. To answer
it we first need to know, per class, how many boxes are small / medium / large — otherwise a
weak class cannot be separated from a class that is simply tiny. The resulting table is also
Table 1 of the report (dataset description).

Where the numbers come from
---------------------------
The COCO annotation files give boxes in pixels. Images are matched to our internal split
(train_fit / val_internal) by file stem using the lists in data/splits; the official val file is
the test part. Size buckets use the COCO thresholds (32^2, 96^2 px) so they line up with the
APs / APm / APl metrics computed later by pycocotools.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pcbdet.config import load_yaml, resolve_path
from pcbdet.data.classes import CLASS_NAMES, coco_id_to_name
from pcbdet.utils.io import read_json, read_lines

SIZE_BUCKETS = ("small", "medium", "large")


@dataclass(frozen=True)
class StatsConfig:
    coco_annotations_dir: Path
    splits_dir: Path
    small_max: int  # area < small_max  -> small
    large_min: int  # area > large_min  -> large
    out_csv: Path

    @staticmethod
    def from_yaml(path: str | Path) -> "StatsConfig":
        d = load_yaml(path)
        th = d.get("size_thresholds", {})
        return StatsConfig(
            coco_annotations_dir=resolve_path(d["coco_annotations_dir"]),
            splits_dir=resolve_path(d["splits_dir"]),
            small_max=int(th.get("small", 32 * 32)),
            large_min=int(th.get("large", 96 * 96)),
            out_csv=resolve_path(d["out_csv"]),
        )


def size_bucket(area: float, small_max: int = 32 * 32, large_min: int = 96 * 96) -> str:
    """COCO size bucket of a box area in px^2 (boundaries inclusive for medium, as in pycocotools)."""
    if area < small_max:
        return "small"
    if area > large_min:
        return "large"
    return "medium"


def stems_from_list(list_path: Path) -> set[str]:
    """Image stems named in a split list file (``./images/train/0003730.jpg`` -> ``0003730``)."""
    return {Path(ln).stem for ln in read_lines(list_path)}


def count_boxes(
    coco_json: Path, stems: set[str] | None, small_max: int, large_min: int
) -> dict[str, Counter]:
    """``{class_name: Counter(small/medium/large)}`` for the images in ``stems`` (None = all)."""
    d = read_json(coco_json)
    keep_ids = {
        im["id"] for im in d["images"] if stems is None or Path(im["file_name"]).stem in stems
    }
    counts: dict[str, Counter] = {name: Counter() for name in CLASS_NAMES}
    for a in d["annotations"]:
        if a["image_id"] in keep_ids:
            counts[coco_id_to_name(a["category_id"])][size_bucket(a["area"], small_max, large_min)] += 1
    return counts


def dataset_stats(cfg: StatsConfig) -> list[dict]:
    """Rows: part, class, small, medium, large, total, small_pct — plus an ALL row per part."""
    train_json = cfg.coco_annotations_dir / "instances_train2017.json"
    val_json = cfg.coco_annotations_dir / "instances_val2017.json"
    parts = [
        ("train_fit", train_json, stems_from_list(cfg.splits_dir / "train_fit.txt")),
        ("val_internal", train_json, stems_from_list(cfg.splits_dir / "val_internal.txt")),
        ("test", val_json, None),
    ]
    rows: list[dict] = []
    for part, js, stems in parts:
        counts = count_boxes(js, stems, cfg.small_max, cfg.large_min)
        total_all = Counter()
        for name in CLASS_NAMES:
            c = counts[name]
            total_all.update(c)
            rows.append(_row(part, name, c))
        rows.append(_row(part, "ALL", total_all))
    return rows


def _row(part: str, cls: str, c: Counter) -> dict:
    total = sum(c[b] for b in SIZE_BUCKETS)
    return {
        "part": part,
        "class": cls,
        **{b: c[b] for b in SIZE_BUCKETS},
        "total": total,
        "small_pct": round(100 * c["small"] / total, 1) if total else 0.0,
    }
