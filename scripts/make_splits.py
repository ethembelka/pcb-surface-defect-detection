"""Create the train_fit / val_internal lists from the official DsPCBSD+ train split.

Usage:
    python scripts/make_splits.py --config configs/splits.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbdet.config import SplitConfig  # noqa: E402
from pcbdet.data.splits import make_splits  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/splits.yaml")
    args = ap.parse_args()

    cfg = SplitConfig.from_yaml(args.config)
    stats = make_splits(cfg)

    print(f"source: {cfg.yolo_root} / {cfg.source_split}   seed={cfg.seed}  val_ratio={cfg.val_ratio}")
    print(f"total {stats['n_total']}  ->  train_fit {stats['n_train_fit']}  val_internal {stats['n_val_internal']}")
    print(f"{'class':<6}{'train_fit':>10}{'val_internal':>14}{'val %':>8}")
    tr, va = stats["images_per_class"]["train_fit"], stats["images_per_class"]["val_internal"]
    for name in tr:
        tot = tr[name] + va[name]
        print(f"{name:<6}{tr[name]:>10}{va[name]:>14}{100 * va[name] / tot if tot else 0:>7.1f}%")
    print(f"lists written to {cfg.out_dir}")


if __name__ == "__main__":
    main()
