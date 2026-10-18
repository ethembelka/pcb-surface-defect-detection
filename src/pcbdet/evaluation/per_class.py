"""Per-class precision / recall / F1 at a decision threshold, and the threshold choice.

AP summarises the whole precision-recall curve; an inspection system, however, runs at one
confidence threshold. We report P / R / F1 at the confidence that maximises micro-F1, chosen on
val_internal (never on the test split) and then applied unchanged to the test split.
"""

from __future__ import annotations

from pcbdet.data.classes import CLASS_NAMES, name_to_yolo_id
from pcbdet.evaluation.matching import Detection, best_f1_threshold, match_per_class, prf_at_threshold


def select_threshold(gt: list[dict], preds: list[dict], iou_thr: float, step: float) -> tuple[float, float]:
    """F1-maximising confidence on a (selection) split -> (threshold, micro_f1)."""
    dets, n_gt = match_per_class(gt, preds, iou_thr)
    return best_f1_threshold(dets, n_gt, step)


def per_class_prf(gt: list[dict], preds: list[dict], iou_thr: float, thr: float) -> dict:
    """``{"micro": {...}, "per_class": {name: {p, r, f1, tp, fp, fn, n_gt}}}`` at confidence ``thr``."""
    dets, n_gt = match_per_class(gt, preds, iou_thr)
    return {"micro": _pack(dets, n_gt, thr, None), "per_class": {
        name: _pack(dets, n_gt, thr, name_to_yolo_id(name) + 1) for name in CLASS_NAMES
    }}


def _pack(dets: list[Detection], n_gt: dict[int, int], thr: float, cat: int | None) -> dict:
    p, r, f1, tp, fp, fn = prf_at_threshold(dets, n_gt, thr, cat)
    return {"p": round(p, 5), "r": round(r, 5), "f1": round(f1, 5), "tp": tp, "fp": fp, "fn": fn,
            "n_gt": n_gt.get(cat, 0) if cat is not None else sum(n_gt.values())}
