"""Confusion matrix including a background row/column (research question 3).

Rows = true class, columns = predicted class. For every image, predictions above the decision
threshold are matched to ground-truth boxes *regardless of class* (greedy on IoU >= iou_thr,
each box used once). A matched pair increments ``M[true][pred]`` — the off-diagonal entries are
the class confusions we are after (SP/MB/SC, CS/CFO). An unmatched GT box is a miss:
``M[true][background]``; an unmatched prediction is a false alarm: ``M[background][pred]``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pcbdet.data.classes import CLASS_NAMES, NUM_CLASSES
from pcbdet.evaluation.matching import iou_matrix

BG = NUM_CLASSES  # index of the background row/column
LABELS = list(CLASS_NAMES) + ["background"]


def confusion_matrix(gt: list[dict], preds: list[dict], iou_thr: float, conf_thr: float) -> np.ndarray:
    """(C+1)x(C+1) integer matrix, rows = true, cols = predicted; last index = background."""
    gt_by = defaultdict(list)
    for g in gt:
        gt_by[g["image_id"]].append(g)
    pr_by = defaultdict(list)
    for p in preds:
        if p["score"] >= conf_thr:
            pr_by[p["image_id"]].append(p)

    m = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)
    for image_id in set(gt_by) | set(pr_by):
        gts, prs = gt_by.get(image_id, []), pr_by.get(image_id, [])
        gboxes = np.array([g["bbox"] for g in gts], dtype=float).reshape(-1, 4)
        pboxes = np.array([p["bbox"] for p in prs], dtype=float).reshape(-1, 4)
        ious = iou_matrix(gboxes, pboxes)
        g_used = np.zeros(len(gts), dtype=bool)
        p_used = np.zeros(len(prs), dtype=bool)
        # greedy by IoU: take the best remaining pair while it clears the threshold
        while ious.size and (ious >= iou_thr).any():
            gi, pi = np.unravel_index(int(ious.argmax()), ious.shape)
            m[gts[gi]["category_id"] - 1, prs[pi]["category_id"] - 1] += 1
            g_used[gi] = p_used[pi] = True
            ious[gi, :] = -1
            ious[:, pi] = -1
        for gi, g in enumerate(gts):
            if not g_used[gi]:
                m[g["category_id"] - 1, BG] += 1
        for pi, p in enumerate(prs):
            if not p_used[pi]:
                m[BG, p["category_id"] - 1] += 1
    return m


def normalize_rows(m: np.ndarray) -> np.ndarray:
    """Row-normalised matrix (each true class sums to 1); empty rows stay 0."""
    s = m.sum(axis=1, keepdims=True)
    return np.where(s > 0, m / np.maximum(s, 1), 0.0)


def top_confusions(m: np.ndarray, k: int = 5) -> list[dict]:
    """The k largest off-diagonal class->class confusions (background excluded), as row shares."""
    n = normalize_rows(m)
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and m[i, j] > 0:
                pairs.append({"true": CLASS_NAMES[i], "pred": CLASS_NAMES[j], "count": int(m[i, j]),
                              "share_of_true": round(float(n[i, j]), 4)})
    return sorted(pairs, key=lambda d: -d["count"])[:k]
