"""Hand-checked examples for the evaluation code (no model, no real data).

Covers: IoU, greedy per-class matching, F1-max threshold selection, the confusion matrix with
background, the file-stem -> image_id mapping (open issue E3), and pycocotools on a tiny GT.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcbdet.data.coco import filter_gt, stem_to_image_id
from pcbdet.evaluation.coco_eval import coco_metrics
from pcbdet.evaluation.confusion import BG, confusion_matrix, normalize_rows, top_confusions
from pcbdet.evaluation.matching import best_f1_threshold, iou_matrix, match_per_class, prf_at_threshold
from pcbdet.evaluation.per_class import per_class_prf, select_threshold


def box(x, y, w, h):
    return [float(x), float(y), float(w), float(h)]


def test_iou_matrix_hand_values():
    a = np.array([box(0, 0, 10, 10)])
    b = np.array([box(0, 0, 10, 10), box(5, 0, 10, 10), box(20, 20, 5, 5)])
    iou = iou_matrix(a, b)[0]
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(50 / 150)  # overlap 5x10=50, union 100+100-50
    assert iou[2] == 0.0


# Two images, two classes. Class 1 has 2 GT boxes, class 2 has 1.
GT = [
    {"image_id": 1, "category_id": 1, "bbox": box(0, 0, 10, 10)},
    {"image_id": 1, "category_id": 1, "bbox": box(50, 50, 10, 10)},
    {"image_id": 2, "category_id": 2, "bbox": box(0, 0, 20, 20)},
]
PREDS = [
    {"image_id": 1, "category_id": 1, "bbox": box(0, 0, 10, 10), "score": 0.9},     # TP
    {"image_id": 1, "category_id": 1, "bbox": box(1, 1, 10, 10), "score": 0.8},     # duplicate -> FP
    {"image_id": 1, "category_id": 1, "bbox": box(80, 80, 10, 10), "score": 0.3},   # FP
    {"image_id": 2, "category_id": 1, "bbox": box(0, 0, 20, 20), "score": 0.7},     # wrong class -> FP (cls 1)
    {"image_id": 2, "category_id": 2, "bbox": box(0, 0, 20, 20), "score": 0.2},     # TP but low score
]


def test_match_per_class_flags():
    dets, n_gt = match_per_class(GT, PREDS, 0.5)
    assert n_gt == {1: 2, 2: 1}
    flags = {(d.image_id, d.category_id, d.score): d.is_tp for d in dets}
    assert flags[(1, 1, 0.9)] is True
    assert flags[(1, 1, 0.8)] is False  # GT already taken by the higher-scoring box
    assert flags[(1, 1, 0.3)] is False
    assert flags[(2, 1, 0.7)] is False  # class mismatch
    assert flags[(2, 2, 0.2)] is True


def test_prf_at_thresholds_hand_computed():
    dets, n_gt = match_per_class(GT, PREDS, 0.5)
    p, r, f1, tp, fp, fn = prf_at_threshold(dets, n_gt, 0.5)  # keeps 0.9, 0.8, 0.7
    assert (tp, fp, fn) == (1, 2, 2)
    assert p == pytest.approx(1 / 3) and r == pytest.approx(1 / 3)
    p, r, f1, tp, fp, fn = prf_at_threshold(dets, n_gt, 0.1)  # keeps all five
    assert (tp, fp, fn) == (2, 3, 1)
    assert prf_at_threshold(dets, n_gt, 0.5, category_id=2)[:3] == (0.0, 0.0, 0.0)


def test_best_f1_threshold_picks_highest_f1():
    dets, n_gt = match_per_class(GT, PREDS, 0.5)
    thr, f1 = best_f1_threshold(dets, n_gt, step=0.1)
    # F1 at 0.9 (1 TP, 0 FP, 2 FN) = 0.5 is the maximum; ties resolve to the higher threshold
    assert thr == pytest.approx(0.9) and f1 == pytest.approx(0.5)
    assert select_threshold(GT, PREDS, 0.5, 0.1) == (pytest.approx(0.9), pytest.approx(0.5))


def test_per_class_prf_structure():
    out = per_class_prf(GT, PREDS, 0.5, thr=0.5)
    assert out["micro"]["tp"] == 1 and out["per_class"]["SH"]["n_gt"] == 2  # SH = category 1
    assert out["per_class"]["SP"]["fn"] == 1  # SP = category 2, its only pred is below 0.5


def test_confusion_matrix_with_background():
    m = confusion_matrix(GT, PREDS, iou_thr=0.5, conf_thr=0.5)
    # image 1: GT(1)@0,0 <-> pred 0.9 (class 1) ; GT(1)@50,50 missed ; pred 0.8 & dup unmatched -> bg row
    # image 2: GT(2) matched to pred 0.7 of class 1 -> confusion SP(true)->SH(pred)
    assert m[0, 0] == 1            # SH correctly SH
    assert m[0, BG] == 1           # one SH missed
    assert m[1, 0] == 1            # SP predicted as SH
    assert m[BG, 0] == 1           # one false alarm predicted as SH (the 0.8 duplicate)
    assert m.sum() == 4
    n = normalize_rows(m)
    assert n[0].sum() == pytest.approx(1.0)
    assert top_confusions(m)[0] == {"true": "SP", "pred": "SH", "count": 1, "share_of_true": 1.0}


def test_stem_to_image_id_and_filter():
    coco = {"images": [{"id": 0, "file_name": "Y_011791.jpg"}, {"id": 7, "file_name": "0003730.jpg"}],
            "annotations": [{"id": 1, "image_id": 7, "category_id": 1, "bbox": box(0, 0, 5, 5), "area": 25}],
            "categories": [{"id": 1, "name": "SH"}]}
    assert stem_to_image_id(coco) == {"Y_011791": 0, "0003730": 7}
    sub = filter_gt(coco, {"0003730"})
    assert [im["id"] for im in sub["images"]] == [7] and len(sub["annotations"]) == 1
    with pytest.raises(ValueError):
        stem_to_image_id({"images": [{"id": 0, "file_name": "a.jpg"}, {"id": 1, "file_name": "a.png"}]})


def test_coco_metrics_perfect_and_empty():
    gt = {"images": [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"}],
          "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": box(10, 10, 20, 20),
                           "area": 400.0, "iscrowd": 0}],
          "categories": [{"id": i + 1, "name": n} for i, n in
                         enumerate(["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"])]}
    perfect = [{"image_id": 1, "category_id": 1, "bbox": box(10, 10, 20, 20), "score": 0.9}]
    m = coco_metrics(gt, perfect)
    assert m["map50"] == pytest.approx(1.0) and m["map50_95"] == pytest.approx(1.0)
    assert m["per_class"]["SH"]["ap50"] == pytest.approx(1.0)
    assert m["ap_small"] == pytest.approx(1.0)  # 400 px^2 < 32^2 -> small
    assert coco_metrics(gt, [])["map50"] == 0.0
