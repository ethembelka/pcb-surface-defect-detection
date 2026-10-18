"""Class confusions in detail (research question 3): which pairs, how often, in both directions,
at what box sizes — plus the concrete (image, GT box, predicted box) instances for galleries.

Uses the same class-agnostic greedy IoU matching as the confusion matrix (evaluation/confusion.py)
and keeps every matched pair whose classes differ.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from pcbdet.data.classes import CLASS_NAMES
from pcbdet.data.stats import size_bucket
from pcbdet.evaluation.matching import iou_matrix


@dataclass(frozen=True)
class ConfusionInstance:
    image_id: int
    true_category: int
    pred_category: int
    gt_bbox: tuple[float, float, float, float]
    pred_bbox: tuple[float, float, float, float]
    score: float
    iou: float
    gt_size: str


def confusion_instances(gt: list[dict], preds: list[dict], iou_thr: float, conf_thr: float) -> list[ConfusionInstance]:
    gt_by = defaultdict(list)
    for g in gt:
        gt_by[g["image_id"]].append(g)
    pr_by = defaultdict(list)
    for p in preds:
        if p["score"] >= conf_thr:
            pr_by[p["image_id"]].append(p)
    out = []
    for image_id in set(gt_by) & set(pr_by):
        gts, prs = gt_by[image_id], pr_by[image_id]
        gboxes = np.array([g["bbox"] for g in gts], dtype=float).reshape(-1, 4)
        pboxes = np.array([p["bbox"] for p in prs], dtype=float).reshape(-1, 4)
        ious = iou_matrix(gboxes, pboxes)
        while ious.size and (ious >= iou_thr).any():
            gi, pi = np.unravel_index(int(ious.argmax()), ious.shape)
            g, p = gts[gi], prs[pi]
            if g["category_id"] != p["category_id"]:
                area = float(g.get("area", g["bbox"][2] * g["bbox"][3]))
                out.append(ConfusionInstance(image_id, g["category_id"], p["category_id"], tuple(g["bbox"]),
                                             tuple(p["bbox"]), float(p["score"]), float(ious[gi, pi]), size_bucket(area)))
            ious[gi, :] = -1
            ious[:, pi] = -1
    return out


def pair_summary(instances: list[ConfusionInstance], gt_counts: dict[int, int], top: int) -> list[dict]:
    """Top directed confusions with the reverse direction, size mix and median score."""
    cnt = Counter((i.true_category, i.pred_category) for i in instances)
    rows = []
    for (t, p), n in cnt.most_common(top):
        mine = [i for i in instances if i.true_category == t and i.pred_category == p]
        sizes = Counter(i.gt_size for i in mine)
        rows.append({
            "true": CLASS_NAMES[t - 1], "pred": CLASS_NAMES[p - 1], "count": n,
            "share_of_true_class": round(n / gt_counts.get(t, 1), 4),
            "reverse_count": cnt.get((p, t), 0),
            "small": sizes.get("small", 0), "medium": sizes.get("medium", 0), "large": sizes.get("large", 0),
            "median_score": round(float(np.median([i.score for i in mine])), 3),
            "median_iou": round(float(np.median([i.iou for i in mine])), 3),
        })
    return rows
