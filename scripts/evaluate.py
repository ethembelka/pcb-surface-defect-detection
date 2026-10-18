"""Evaluate a trained run on a split (thin CLI around pcbdet.evaluation.evaluator).

Usage (from the repository root):
    python scripts/evaluate.py --run outputs/runs/<run_id> --split val_internal      # selects the threshold
    python scripts/evaluate.py --run outputs/runs/<run_id> --split test --efficiency # final numbers, ONCE

Options:
    --weights     weights file (default <run>/weights/best.pt)
    --config      evaluation config (default configs/eval/default.yaml)
    --efficiency  also measure params / GFLOPs / FPS on this machine (slow-ish)
    --overwrite   recompute predictions even if predictions.json exists
    --label       model label for the report tables (default: run_id)

Outputs: outputs/results/<run_id>/<split>/{predictions.json, metrics.json, config.yaml, git_hash.txt},
         reports/tables/{main_results,per_class}_<run_id>_<split>.csv,
         reports/figures/confusion_matrix_<run_id>_<split>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbdet.config import EvalConfig, resolve_path  # noqa: E402
from pcbdet.evaluation.evaluator import evaluate_run  # noqa: E402
from pcbdet.reporting.tables import write_report_tables  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val_internal", choices=["val_internal", "test"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default="configs/eval/default.yaml")
    ap.add_argument("--efficiency", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    cfg = EvalConfig.from_yaml(args.config)
    run_dir = resolve_path(args.run)
    m = evaluate_run(run_dir, cfg, args.split, weights=resolve_path(args.weights) if args.weights else None,
                     with_efficiency=args.efficiency, overwrite=args.overwrite)
    files = write_report_tables(m, cfg.tables_dir, cfg.figures_dir, cfg.references, args.label or run_dir.name)

    print(f"\n[{m['run_id']} / {m['split']}]  imgsz {m['imgsz']}  images {m['n_images']}  gt {m['n_gt_boxes']}  preds {m['n_predictions']}")
    print(f"  AP50 {m['map50']:.4f}  AP75 {m['ap75']:.4f}  AP50:95 {m['map50_95']:.4f}  "
          f"APs {m['ap_small']:.4f}  APm {m['ap_medium']:.4f}  APl {m['ap_large']:.4f}")
    print(f"  decision conf {m['decision_threshold']['conf']} (selected on {m['decision_threshold']['selected_on']}): "
          f"P {m['precision']:.4f}  R {m['recall']:.4f}  F1 {m['f1']:.4f}")
    print(f"  {'class':<6}{'AP50':>8}{'AP50:95':>9}{'P':>8}{'R':>8}{'F1':>8}{'n_gt':>7}")
    for name, d in m["per_class"].items():
        print(f"  {name:<6}{d['ap50']:>8.3f}{d['ap50_95']:>9.3f}{d['p']:>8.3f}{d['r']:>8.3f}{d['f1']:>8.3f}{d['n_gt']:>7}")
    print("  top confusions:", ", ".join(f"{c['true']}->{c['pred']} {c['count']}" for c in m["confusion"]["top_confusions"]))
    if "efficiency" in m:
        e = m["efficiency"]
        print(f"  efficiency: {e['params']/1e6:.2f} M params, {e['gflops']} GFLOPs, {e['ms_per_image']} ms/img "
              f"({e['fps']} FPS) on {e['device']}")
    print("  written:", *[str(f) for f in files], sep="\n    ")


if __name__ == "__main__":
    main()
