"""Ground-truth-level matching: for every GT box, was it found, by which prediction, with what IoU?

The evaluation code (evaluation/matching.py) is prediction-centric (TP/FP per detection). The
error analysis needs the *other* view: per GT box -> matched or missed (FN), and per prediction
above the decision threshold -> matched or false alarm (FP), together with the IoU of each match
and the box sizes. Same greedy rule as the metrics: predictions in descending score, class-aware,
IoU >= iou_thr, each GT used once; only predictions with score >= conf_thr take part.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pcbdet.data.stats import size_bucket
from pcbdet.evaluation.matching import iou_matrix


@dataclass(frozen=True)
class GTRecord:
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    area: float
    size: str  # small / medium / large
    matched: bool
    iou: float  # IoU with the matching prediction (0 if missed)
    score: float  # score of the matching prediction (0 if missed)


@dataclass(frozen=True)
class FPRecord:
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float
    best_iou_any_class: float  # overlap with the closest GT box of *any* class (label-error hint)
    best_gt_category: int | None


def match_gt_level(
    gt: list[dict], preds: list[dict], iou_thr: float, conf_thr: float
) -> tuple[list[GTRecord], list[FPRecord]]:
    gt_by = defaultdict(list)
    for g in gt:
        gt_by[(g["image_id"], g["category_id"])].append(g)
    pr_by = defaultdict(list)
    for p in preds:
        if p["score"] >= conf_thr:
            pr_by[(p["image_id"], p["category_id"])].append(p)
    all_gt_by_img = defaultdict(list)
    for g in gt:
        all_gt_by_img[g["image_id"]].append(g)

    gt_records: list[GTRecord] = []
    fp_records: list[FPRecord] = []
    for key in set(gt_by) | set(pr_by):
        gts = gt_by.get(key, [])
        prs = sorted(pr_by.get(key, []), key=lambda p: -p["score"])
        gboxes = np.array([g["bbox"] for g in gts], dtype=float).reshape(-1, 4)
        pboxes = np.array([p["bbox"] for p in prs], dtype=float).reshape(-1, 4)
        ious = iou_matrix(pboxes, gboxes)
        g_match = [None] * len(gts)  # (iou, score)
        for i, p in enumerate(prs):
            matched = False
            if len(gts):
                cand = np.array([ious[i, j] if g_match[j] is None else -1.0 for j in range(len(gts))])
                j = int(cand.argmax())
                if cand[j] >= iou_thr:
                    g_match[j] = (float(cand[j]), float(p["score"]))
                    matched = True
            if not matched:
                others = all_gt_by_img.get(key[0], [])
                oboxes = np.array([g["bbox"] for g in others], dtype=float).reshape(-1, 4)
                oi = iou_matrix(pboxes[i : i + 1], oboxes)[0] if len(others) else np.zeros(0)
                best = int(oi.argmax()) if len(oi) else None
                fp_records.append(FPRecord(key[0], key[1], tuple(p["bbox"]), float(p["score"]),
                                           float(oi[best]) if best is not None else 0.0,
                                           others[best]["category_id"] if best is not None else None))
        for j, g in enumerate(gts):
            area = float(g.get("area", g["bbox"][2] * g["bbox"][3]))
            m = g_match[j]
            gt_records.append(GTRecord(key[0], key[1], tuple(g["bbox"]), area, size_bucket(area),
                                       m is not None, m[0] if m else 0.0, m[1] if m else 0.0))
    return gt_records, fp_records
