"""COCO metrics with pycocotools: AP, AP50, AP75, APs/APm/APl, AR@100 and per-class AP.

These are the numbers the reference papers report (Lv et al. 2024 Table), so we compute them
with the same library rather than trusting Ultralytics' own mAP implementation.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pcbdet.data.classes import CLASS_NAMES

# index into COCOeval.stats
_STATS = {
    "map50_95": 0, "map50": 1, "ap75": 2, "ap_small": 3, "ap_medium": 4, "ap_large": 5,
    "ar1": 6, "ar10": 7, "ar100": 8, "ar_small": 9, "ar_medium": 10, "ar_large": 11,
}


def _coco_from_dict(d: dict[str, Any]) -> COCO:
    coco = COCO()
    coco.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def coco_metrics(gt: dict[str, Any], predictions: list[dict]) -> dict[str, Any]:
    """Evaluate COCO-format predictions against a COCO GT dict.

    Returns the 12 standard stats plus ``per_class`` = ``{name: {ap50, ap50_95}}``.
    An empty prediction list yields zeros instead of crashing.
    """
    coco_gt = _coco_from_dict(gt)
    if not predictions:
        out = {k: 0.0 for k in _STATS}
        out["per_class"] = {n: {"ap50": 0.0, "ap50_95": 0.0} for n in CLASS_NAMES}
        return out
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    out: dict[str, Any] = {k: round(float(ev.stats[i]), 5) for k, i in _STATS.items()}

    # precision array: [T iou thresholds, R recall steps, K classes, A areas, M max dets]
    prec = ev.eval["precision"]
    per_class = {}
    for k, cat_id in enumerate(ev.params.catIds):
        name = CLASS_NAMES[cat_id - 1]
        p_all = prec[:, :, k, 0, -1]  # area=all, maxDet=100
        p50 = prec[0, :, k, 0, -1]
        per_class[name] = {
            "ap50_95": round(float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else 0.0, 5),
            "ap50": round(float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else 0.0, 5),
        }
    out["per_class"] = per_class
    return out
