"""Error analysis of one evaluated run/split -> analysis.json, tables and figures (phase A4).

Inputs are the artefacts of ``scripts/evaluate.py``: ``predictions.json`` and ``metrics.json`` in
``outputs/results/<run_id>/<split>/``. Nothing is re-predicted; the model and the decision
threshold stay exactly as evaluated.

Produces
--------
tables/   size_ap_<tag>.csv, size_recall_<tag>.csv, iou_<tag>.csv, confusion_pairs_<tag>.csv
figures/  iou_boxplot_<tag>.png, gallery_fn_<cls>_<tag>.png, gallery_fp_<cls>_<tag>.png,
          gallery_pair_<true>_as_<pred>_<tag>.png
results/  analysis.json (the numbers behind the findings written to docs/EXPERIMENTS.md)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pcbdet.analysis.confusion_pairs import confusion_instances, pair_summary
from pcbdet.analysis.error_gallery import grid, label, render_crop
from pcbdet.analysis.gt_matching import match_gt_level
from pcbdet.analysis.iou_analysis import iou_figure, iou_summary
from pcbdet.analysis.size_analysis import per_class_size_ap, per_class_size_recall
from pcbdet.config import DatasetConfig, EvalConfig
from pcbdet.data.classes import CLASS_NAMES
from pcbdet.data.coco import load_gt_for_split, split_image_paths, stem_to_image_id
from pcbdet.reporting.tables import write_csv
from pcbdet.utils.io import read_json, write_json


def analyze_run(run_id: str, cfg: EvalConfig, split: str) -> dict[str, Any]:
    a = cfg.raw.get("analysis", {})
    top_pairs, per_cls, pair_ex = int(a.get("top_pairs", 5)), int(a.get("gallery_per_class", 12)), int(a.get("pair_examples", 8))
    margin, min_px, size, low = float(a.get("crop_margin", 1.0)), int(a.get("crop_min_px", 48)), int(a.get("crop_size", 160)), int(a.get("low_support", 30))

    res_dir = cfg.results_dir / run_id / split
    metrics = read_json(res_dir / "metrics.json")
    preds = read_json(res_dir / "predictions.json")
    thr = float(metrics["decision_threshold"]["conf"])
    tag = f"{run_id}_{split}"

    dataset = DatasetConfig.from_yaml(cfg.datasets[split])
    images = split_image_paths(dataset, "val")
    gt = load_gt_for_split(cfg.coco_annotations_dir / cfg.coco_json[split], images)
    id_to_path = {stem_to_image_id(gt)[p.stem]: p for p in images}
    gt_counts = Counter(g["category_id"] for g in gt["annotations"])

    # 1. class x size
    size_ap = per_class_size_ap(gt, preds, low)
    gt_recs, fp_recs = match_gt_level(gt["annotations"], preds, cfg.iou_match, thr)
    size_rec = per_class_size_recall(gt_recs, low)
    # 2. IoU
    iou_rows = iou_summary(gt_recs)
    # 3. confusions
    inst = confusion_instances(gt["annotations"], preds, cfg.iou_match, thr)
    pairs = pair_summary(inst, gt_counts, top_pairs)

    tables = [
        write_csv(size_ap, cfg.tables_dir / f"size_ap_{tag}.csv"),
        write_csv(size_rec, cfg.tables_dir / f"size_recall_{tag}.csv"),
        write_csv(iou_rows, cfg.tables_dir / f"iou_{tag}.csv"),
        write_csv(pairs, cfg.tables_dir / f"confusion_pairs_{tag}.csv") if pairs else None,
    ]
    figures = [iou_figure(gt_recs, cfg.figures_dir / f"iou_boxplot_{tag}.png", f"{run_id} / {split}: IoU of found boxes")]

    # 4. galleries
    gt_by_img = defaultdict(list)
    for g in gt["annotations"]:
        gt_by_img[g["image_id"]].append(g)
    pr_by_img = defaultdict(list)
    for p in preds:
        if p["score"] >= thr:
            pr_by_img[p["image_id"]].append(p)

    def context(image_id):
        return ([tuple(g["bbox"]) for g in gt_by_img[image_id]],
                [(tuple(p["bbox"]), label(p["category_id"], p["score"])) for p in pr_by_img[image_id]])

    fp_hint = {}  # share of FPs that overlap a GT box of another class (possible label issues)
    for cat in range(1, len(CLASS_NAMES) + 1):
        name = CLASS_NAMES[cat - 1]
        fns = sorted([r for r in gt_recs if r.category_id == cat and not r.matched], key=lambda r: -r.area)[:per_cls]
        tiles = []
        for r in fns:
            gts, prs = context(r.image_id)
            tiles.append(render_crop(id_to_path[r.image_id], gts, prs, r.bbox, margin, min_px, size,
                                     f"{id_to_path[r.image_id].stem} FN {name} {r.size}"))
        figures.append(grid(tiles, 4, cfg.figures_dir / f"gallery_fn_{name}_{tag}.png", f"Missed {name} boxes (green = GT, red = predictions ≥ {thr})"))
        fps = sorted([f for f in fp_recs if f.category_id == cat], key=lambda f: -f.score)[:per_cls]
        tiles = []
        for f in fps:
            gts, prs = context(f.image_id)
            tiles.append(render_crop(id_to_path[f.image_id], gts, prs, f.bbox, margin, min_px, size,
                                     f"{id_to_path[f.image_id].stem} FP {name} {f.score:.2f}"))
        figures.append(grid(tiles, 4, cfg.figures_dir / f"gallery_fp_{name}_{tag}.png", f"False alarms predicted as {name} (green = GT, red = predictions ≥ {thr})"))
        all_fp = [f for f in fp_recs if f.category_id == cat]
        fp_hint[name] = {"n_fp": len(all_fp),
                         "overlap_other_class_iou>=0.3": sum(f.best_iou_any_class >= 0.3 for f in all_fp),
                         "no_gt_nearby_iou<0.1": sum(f.best_iou_any_class < 0.1 for f in all_fp)}
    for row in pairs:
        t, p = CLASS_NAMES.index(row["true"]) + 1, CLASS_NAMES.index(row["pred"]) + 1
        ex = sorted([i for i in inst if i.true_category == t and i.pred_category == p], key=lambda i: -i.score)[:pair_ex]
        tiles = []
        for i in ex:
            gts, prs = context(i.image_id)
            tiles.append(render_crop(id_to_path[i.image_id], gts, prs, i.gt_bbox, margin, min_px, size,
                                     f"{id_to_path[i.image_id].stem} {row['true']}→{row['pred']} {i.score:.2f}"))
        figures.append(grid(tiles, 4, cfg.figures_dir / f"gallery_pair_{row['true']}_as_{row['pred']}_{tag}.png",
                            f"True {row['true']} predicted as {row['pred']} (green = GT, red = predictions ≥ {thr})"))

    summary = {
        "run_id": run_id, "split": split, "decision_conf": thr,
        "n_gt": len(gt_recs), "n_found": sum(r.matched for r in gt_recs), "n_fn": sum(not r.matched for r in gt_recs),
        "n_fp": len(fp_recs),
        "fn_by_size": dict(Counter(r.size for r in gt_recs if not r.matched)),
        "gt_by_size": dict(Counter(r.size for r in gt_recs)),
        "size_ap": size_ap, "size_recall": size_rec, "iou": iou_rows, "confusion_pairs": pairs, "fp_hint": fp_hint,
        "tables": [str(t) for t in tables if t], "figures": [str(f) for f in figures if f],
    }
    write_json(summary, res_dir / "analysis.json")
    return summary
