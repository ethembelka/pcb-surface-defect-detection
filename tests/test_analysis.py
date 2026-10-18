"""Hand-checked examples for the error-analysis helpers (GT-level matching, size recall, crops, pairs)."""

from __future__ import annotations

from pcbdet.analysis.confusion_pairs import confusion_instances, pair_summary
from pcbdet.analysis.error_gallery import crop_window
from pcbdet.analysis.gt_matching import match_gt_level
from pcbdet.analysis.size_analysis import per_class_size_recall


def box(x, y, w, h):
    return [float(x), float(y), float(w), float(h)]


GT = [
    {"image_id": 1, "category_id": 1, "bbox": box(0, 0, 10, 10), "area": 100},       # small, found
    {"image_id": 1, "category_id": 1, "bbox": box(50, 50, 40, 40), "area": 1600},    # medium, missed
    {"image_id": 2, "category_id": 2, "bbox": box(0, 0, 20, 20), "area": 400},       # small, predicted as class 1
]
PREDS = [
    {"image_id": 1, "category_id": 1, "bbox": box(1, 1, 10, 10), "score": 0.9},     # matches GT 1 (IoU ~0.68)
    {"image_id": 1, "category_id": 1, "bbox": box(100, 100, 10, 10), "score": 0.6}, # FP, nothing nearby
    {"image_id": 2, "category_id": 1, "bbox": box(0, 0, 20, 20), "score": 0.8},     # FP for class 1, overlaps class-2 GT
    {"image_id": 2, "category_id": 2, "bbox": box(0, 0, 20, 20), "score": 0.1},     # below threshold
]


def test_gt_level_matching_and_fp_hints():
    gts, fps = match_gt_level(GT, PREDS, iou_thr=0.5, conf_thr=0.33)
    by = {(r.image_id, r.bbox): r for r in gts}
    assert by[(1, (0.0, 0.0, 10.0, 10.0))].matched and abs(by[(1, (0.0, 0.0, 10.0, 10.0))].iou - 81 / 119) < 1e-6
    assert not by[(1, (50.0, 50.0, 40.0, 40.0))].matched and by[(1, (50.0, 50.0, 40.0, 40.0))].size == "medium"
    assert not by[(2, (0.0, 0.0, 20.0, 20.0))].matched  # its only same-class prediction is below threshold
    assert len(fps) == 2
    hints = {(f.image_id, f.score): f for f in fps}
    assert hints[(1, 0.6)].best_iou_any_class == 0.0
    assert hints[(2, 0.8)].best_iou_any_class == 1.0 and hints[(2, 0.8)].best_gt_category == 2


def test_size_recall_rows():
    gts, _ = match_gt_level(GT, PREDS, 0.5, 0.33)
    rows = {(r["class"], r["size"]): r for r in per_class_size_recall(gts, low_support=30)}
    assert rows[("SH", "all")]["n_gt"] == 2 and rows[("SH", "all")]["recall"] == 0.5
    assert rows[("SH", "small")]["recall"] == 1.0 and rows[("SH", "medium")]["recall"] == 0.0
    assert rows[("SP", "all")]["recall"] == 0.0 and rows[("SH", "all")]["low_support"]


def test_confusion_instances_and_pairs():
    inst = confusion_instances(GT, PREDS, iou_thr=0.5, conf_thr=0.33)
    assert len(inst) == 1 and (inst[0].true_category, inst[0].pred_category) == (2, 1)
    rows = pair_summary(inst, {1: 2, 2: 1}, top=5)
    assert rows[0]["true"] == "SP" and rows[0]["pred"] == "SH" and rows[0]["count"] == 1
    assert rows[0]["share_of_true_class"] == 1.0 and rows[0]["reverse_count"] == 0 and rows[0]["small"] == 1


def test_crop_window_centres_and_clamps():
    assert crop_window(box(100, 100, 10, 10), 226, 226, margin=1.0, min_px=0) == (90, 90, 120, 120)
    x0, y0, x1, y1 = crop_window(box(2, 2, 10, 10), 226, 226, margin=1.0, min_px=48)
    assert (x0, y0) == (0, 0) and (x1 - x0, y1 - y0) == (48, 48)  # shifted, not shrunk
    x0, y0, x1, y1 = crop_window(box(220, 220, 5, 5), 226, 226, margin=1.0, min_px=48)
    assert (x1, y1) == (226, 226) and (x1 - x0) == 48
