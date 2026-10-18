"""Class x size analysis (research question 2): AP and recall per class and COCO size bucket.

Two complementary views:
- ``per_class_size_ap``: pycocotools AP50 / AP50:95 for each class restricted to small / medium /
  large GT boxes (pycocotools' own area ranges, so numbers line up with APs/APm/APl).
- ``per_class_size_recall``: at the decision threshold, the share of GT boxes of each size that
  were found (from the GT-level matching). AP needs enough boxes to be meaningful, so cells with
  fewer than ``low_support`` GT boxes are flagged.
"""

from __future__ import annotations

import contextlib
import io
from collections import Counter
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pcbdet.analysis.gt_matching import GTRecord
from pcbdet.data.classes import CLASS_NAMES
from pcbdet.data.stats import SIZE_BUCKETS

_AREA_IDX = {"all": 0, "small": 1, "medium": 2, "large": 3}


def per_class_size_ap(gt: dict[str, Any], predictions: list[dict], low_support: int) -> list[dict]:
    """Rows: class, size, n_gt, ap50, ap50_95, low_support (bool)."""
    coco_gt = COCO()
    coco_gt.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions) if predictions else None
    if coco_dt is None:
        return []
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    counts = Counter()
    for a in gt["annotations"]:
        counts[(a["category_id"], _bucket_of(a, ev.params.areaRng))] += 1
    rows = []
    for k, cat_id in enumerate(ev.params.catIds):
        for size in ("all",) + SIZE_BUCKETS:
            a = _AREA_IDX[size]
            p_all = prec[:, :, k, a, -1]
            p50 = prec[0, :, k, a, -1]
            n = sum(v for (c, s), v in counts.items() if c == cat_id and (size == "all" or s == size))
            rows.append({
                "class": CLASS_NAMES[cat_id - 1], "size": size, "n_gt": n,
                "ap50": round(float(np.mean(p50[p50 > -1])), 4) if (p50 > -1).any() else None,
                "ap50_95": round(float(np.mean(p_all[p_all > -1])), 4) if (p_all > -1).any() else None,
                "low_support": n < low_support,
            })
    return rows


def _bucket_of(ann: dict, area_rng: list[list[float]]) -> str:
    area = ann.get("area", ann["bbox"][2] * ann["bbox"][3])
    for name, (lo, hi) in zip(("all", "small", "medium", "large"), area_rng):
        if name != "all" and lo <= area < hi:
            return name
    return "large"


def per_class_size_recall(gt_records: list[GTRecord], low_support: int) -> list[dict]:
    """Rows: class, size, n_gt, n_found, recall, median_iou (of found boxes), low_support."""
    rows = []
    for cat in range(1, len(CLASS_NAMES) + 1):
        for size in ("all",) + SIZE_BUCKETS:
            recs = [r for r in gt_records if r.category_id == cat and (size == "all" or r.size == size)]
            found = [r for r in recs if r.matched]
            rows.append({
                "class": CLASS_NAMES[cat - 1], "size": size, "n_gt": len(recs), "n_found": len(found),
                "recall": round(len(found) / len(recs), 4) if recs else None,
                "median_iou": round(float(np.median([r.iou for r in found])), 4) if found else None,
                "low_support": len(recs) < low_support,
            })
    return rows
