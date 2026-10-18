"""Print and save the class x object-size table (report Table 1).

Usage (from the repository root):
    python scripts/dataset_stats.py --config configs/stats.yaml

Requires data/splits/*.txt (run scripts/make_splits.py first). Writes reports/tables/dataset_stats.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbdet.data.stats import SIZE_BUCKETS, StatsConfig, dataset_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/stats.yaml")
    args = ap.parse_args()

    cfg = StatsConfig.from_yaml(args.config)
    rows = dataset_stats(cfg)

    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for part in ("train_fit", "val_internal", "test"):
        print(f"\n[{part}]  {'class':<6}" + "".join(f"{b:>8}" for b in SIZE_BUCKETS) + f"{'total':>8}{'small%':>8}")
        for r in rows:
            if r["part"] == part:
                print(f"{'':<16}{r['class']:<6}" + "".join(f"{r[b]:>8}" for b in SIZE_BUCKETS)
                      + f"{r['total']:>8}{r['small_pct']:>8}")
    print(f"\nwritten {cfg.out_csv}")


if __name__ == "__main__":
    main()
