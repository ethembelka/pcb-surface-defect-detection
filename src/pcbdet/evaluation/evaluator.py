"""End-to-end evaluation of one trained run on one split -> metrics.json (+ tables, figure).

Flow (docs/EVALUATION_PROTOCOL.md)
---------------------------------
1. images of the split + their COCO ground truth (data/coco.py)
2. predictions as COCO results json via the Detector interface (cached; ``--overwrite`` redoes)
3. COCO metrics with pycocotools (coco_eval.py)
4. decision threshold: F1-max on the *selection* split (val_internal); the test split reuses the
   threshold stored in val_internal's metrics.json (so test is never used to choose anything)
5. per-class P/R/F1 at that threshold (per_class.py), confusion matrix (confusion.py)
6. optional efficiency numbers (efficiency.py)
7. everything -> outputs/results/<run_id>/<split>/metrics.json, plus config/git snapshot
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pcbdet.config import DatasetConfig, EvalConfig
from pcbdet.data.coco import load_gt_for_split, split_image_paths, stem_to_image_id
from pcbdet.evaluation.coco_eval import coco_metrics
from pcbdet.evaluation.confusion import LABELS, confusion_matrix, normalize_rows, top_confusions
from pcbdet.evaluation.efficiency import efficiency
from pcbdet.evaluation.per_class import per_class_prf, select_threshold
from pcbdet.models.registry import get_detector
from pcbdet.utils.io import read_json, write_json, write_yaml
from pcbdet.utils.run import git_hash


def evaluate_run(
    run_dir: Path,
    cfg: EvalConfig,
    split: str,
    weights: Path | None = None,
    model_name: str | None = None,
    with_efficiency: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_id = run_dir.name
    weights = weights or run_dir / "weights" / "best.pt"
    model_name = model_name or _model_name_from_run(run_dir)
    imgsz = run_imgsz(run_dir, cfg.imgsz)  # a 320 px model must be evaluated at 320 px
    out_dir = cfg.results_dir / run_id / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. images + ground truth
    dataset = DatasetConfig.from_yaml(cfg.datasets[split])
    images = split_image_paths(dataset, "val")
    gt = load_gt_for_split(cfg.coco_annotations_dir / cfg.coco_json[split], images)
    stem_to_id = stem_to_image_id(gt)

    # 2. predictions (COCO results json)
    pred_json = out_dir / "predictions.json"
    if overwrite or not pred_json.exists():
        get_detector(model_name).export_predictions(
            weights, images, stem_to_id, pred_json, imgsz=imgsz, conf=cfg.conf,
            iou_nms=cfg.iou_nms, max_det=cfg.max_det, half=cfg.half, batch=cfg.batch,
        )
    preds = read_json(pred_json)

    # 3. COCO metrics
    metrics: dict[str, Any] = {"run_id": run_id, "split": split, "imgsz": imgsz, "n_images": len(images),
                               "n_gt_boxes": len(gt["annotations"]), "n_predictions": len(preds)}
    metrics.update(coco_metrics(gt, preds))

    # 4. decision threshold
    if split == cfg.decision_select_on:
        thr, f1 = select_threshold(gt["annotations"], preds, cfg.iou_match, cfg.decision_grid_step)
        metrics["decision_threshold"] = {"conf": thr, "selected_on": split, "micro_f1": round(f1, 5)}
    else:
        sel = cfg.results_dir / run_id / cfg.decision_select_on / "metrics.json"
        if not sel.exists():
            raise FileNotFoundError(
                f"evaluate '{cfg.decision_select_on}' first: the decision threshold is selected there ({sel})"
            )
        thr = read_json(sel)["decision_threshold"]["conf"]
        metrics["decision_threshold"] = {"conf": thr, "selected_on": cfg.decision_select_on}

    # 5. P/R/F1 per class + confusion matrix at that threshold
    prf = per_class_prf(gt["annotations"], preds, cfg.iou_match, thr)
    metrics["precision"], metrics["recall"], metrics["f1"] = prf["micro"]["p"], prf["micro"]["r"], prf["micro"]["f1"]
    for name, d in prf["per_class"].items():
        metrics["per_class"][name].update(d)
    cm = confusion_matrix(gt["annotations"], preds, cfg.iou_match, thr)
    metrics["confusion"] = {"labels": LABELS, "counts": cm.tolist(),
                            "row_normalized": normalize_rows(cm).round(4).tolist(),
                            "top_confusions": top_confusions(cm)}

    # 6. efficiency (optional: slow, hardware dependent)
    if with_efficiency:
        metrics["efficiency"] = efficiency(weights, images[:64], imgsz, cfg.eff_half, cfg.eff_warmup, cfg.eff_iters)

    # 7. snapshot
    write_json(metrics, out_dir / "metrics.json")
    write_yaml({"eval": cfg.raw, "weights": str(weights), "model": model_name}, out_dir / "config.yaml")
    (out_dir / "git_hash.txt").write_text(git_hash() + "\n", encoding="utf-8")
    return metrics


def _run_train_cfg(run_dir: Path) -> dict[str, Any]:
    """The ``train`` block of the run's config.yaml (written by the trainer); {} if missing."""
    cfg = run_dir / "config.yaml"
    if cfg.exists():
        from pcbdet.config import load_yaml

        return dict(load_yaml(cfg).get("train", {}))
    return {}


def _model_name_from_run(run_dir: Path) -> str:
    return str(_run_train_cfg(run_dir).get("model", "yolo"))


def run_imgsz(run_dir: Path, default: int) -> int:
    """Image size the run was trained with (evaluation must use the same), else ``default``."""
    return int(_run_train_cfg(run_dir).get("imgsz", default))
