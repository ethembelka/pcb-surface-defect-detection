"""Box matching shared by the per-class and confusion-matrix metrics.

Everything here works on plain COCO-style records (``bbox = [x, y, w, h]`` in pixels) and NumPy,
so it is independent of the detector and easy to test with hand-made examples.

Greedy matching (the usual detection convention): predictions are taken in descending score
order; each is matched to the still-unmatched ground-truth box with the highest IoU if that
IoU >= ``iou_thr``. A matched prediction is a true positive (TP), an unmatched one a false
positive (FP); ground-truth boxes left unmatched are false negatives (FN).
Because the greedy pass goes by score, matching once at the lowest confidence and then
*filtering by score* gives exactly the result of re-matching at a higher threshold — which is
what lets us sweep thresholds cheaply when choosing the F1-maximising confidence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of ``a`` (N,4) and ``b`` (M,4) boxes given as [x, y, w, h]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


@dataclass(frozen=True)
class Detection:
    """One prediction after matching: its score, class and whether it hit a GT box of that class."""

    image_id: int
    category_id: int
    score: float
    is_tp: bool


def match_per_class(
    gt: list[dict], preds: list[dict], iou_thr: float
) -> tuple[list[Detection], dict[int, int]]:
    """Class-aware greedy matching over all images.

    Returns the list of scored detections (TP/FP flag each) and ``{category_id: n_gt}``.
    Matching is done separately per (image, class): a prediction can only match a GT box of the
    same class, as in standard per-class precision/recall.
    """
    gt_by = defaultdict(list)
    for g in gt:
        gt_by[(g["image_id"], g["category_id"])].append(g["bbox"])
    pr_by = defaultdict(list)
    for p in preds:
        pr_by[(p["image_id"], p["category_id"])].append((p["score"], p["bbox"]))

    n_gt = defaultdict(int)
    for (_, c), boxes in gt_by.items():
        n_gt[c] += len(boxes)

    dets: list[Detection] = []
    for key, plist in pr_by.items():
        plist.sort(key=lambda t: -t[0])
        gboxes = np.array(gt_by.get(key, []), dtype=float).reshape(-1, 4)
        ious = iou_matrix(np.array([b for _, b in plist], dtype=float).reshape(-1, 4), gboxes)
        used = np.zeros(len(gboxes), dtype=bool)
        for i, (score, _) in enumerate(plist):
            is_tp = False
            if len(gboxes):
                cand = np.where(~used, ious[i], -1.0)
                j = int(cand.argmax())
                if cand[j] >= iou_thr:
                    used[j] = True
                    is_tp = True
            dets.append(Detection(key[0], key[1], float(score), is_tp))
    return dets, dict(n_gt)


def prf_at_threshold(
    dets: list[Detection], n_gt: dict[int, int], thr: float, category_id: int | None = None
) -> tuple[float, float, float, int, int, int]:
    """Precision, recall, F1 (and TP, FP, FN) at confidence >= ``thr``; one class or micro over all."""
    sel = [d for d in dets if d.score >= thr and (category_id is None or d.category_id == category_id)]
    tp = sum(d.is_tp for d in sel)
    fp = len(sel) - tp
    total_gt = n_gt.get(category_id, 0) if category_id is not None else sum(n_gt.values())
    fn = total_gt - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / total_gt if total_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, tp, fp, fn


def best_f1_threshold(dets: list[Detection], n_gt: dict[int, int], step: float = 0.01) -> tuple[float, float]:
    """Confidence in [step, 1) that maximises micro-F1 -> (threshold, f1). Ties -> higher threshold."""
    best_t, best_f1 = step, -1.0
    for t in np.arange(step, 1.0, step):
        f1 = prf_at_threshold(dets, n_gt, float(t))[2]
        if f1 >= best_f1:
            best_t, best_f1 = float(t), f1
    return round(best_t, 4), best_f1
