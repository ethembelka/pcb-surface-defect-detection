"""Error analysis of an evaluated run (thin CLI around pcbdet.analysis.analyzer).

Usage (from the repository root, after scripts/evaluate.py for the same run/split):
    python scripts/analyze.py --run outputs/runs/<run_id> --split test

Outputs: outputs/results/<run_id>/<split>/analysis.json, reports/tables/{size_ap,size_recall,iou,confusion_pairs}_*.csv,
         reports/figures/{iou_boxplot,gallery_fn_*,gallery_fp_*,gallery_pair_*}_*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbdet.analysis.analyzer import analyze_run  # noqa: E402
from pcbdet.config import EvalConfig, resolve_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["val_internal", "test"])
    ap.add_argument("--config", default="configs/eval/default.yaml")
    args = ap.parse_args()

    cfg = EvalConfig.from_yaml(args.config)
    s = analyze_run(resolve_path(args.run).name, cfg, args.split)

    print(f"\n[{s['run_id']} / {s['split']}] conf ≥ {s['decision_conf']}: GT {s['n_gt']}, found {s['n_found']}, "
          f"missed {s['n_fn']} {s['fn_by_size']}, false alarms {s['n_fp']}")
    print(f"\n  {'class':<6}{'size':<8}{'n_gt':>6}{'AP50':>8}{'AP50:95':>9}{'recall':>8}{'medIoU':>8}")
    rec = {(r['class'], r['size']): r for r in s['size_recall']}
    for r in s["size_ap"]:
        q = rec[(r["class"], r["size"])]
        flag = " *" if r["low_support"] else ""
        f = lambda v: f"{v:8.3f}" if v is not None else f"{'-':>8}"
        print(f"  {r['class']:<6}{r['size']:<8}{r['n_gt']:>6}{f(r['ap50'])}{f(r['ap50_95']):>9}{f(q['recall'])}{f(q['median_iou'])}{flag}")
    print("  (* fewer than low_support GT boxes)")
    print("\n  confusion pairs (true -> pred): count, reverse, small/medium/large, median score")
    for p in s["confusion_pairs"]:
        print(f"  {p['true']:>5} -> {p['pred']:<5} {p['count']:>4}  rev {p['reverse_count']:>3}  "
              f"{p['small']}/{p['medium']}/{p['large']}  score {p['median_score']}")
    print("\n  FP hints (n_fp, overlapping another class' GT, nothing nearby):")
    for k, v in s["fp_hint"].items():
        print(f"  {k:<6}{v['n_fp']:>5}{v['overlap_other_class_iou>=0.3']:>6}{v['no_gt_nearby_iou<0.1']:>6}")
    print(f"\n  written {len(s['tables'])} tables, {len(s['figures'])} figures; analysis.json")


if __name__ == "__main__":
    main()
