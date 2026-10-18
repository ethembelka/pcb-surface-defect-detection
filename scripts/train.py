"""Train a detector from a config file (thin CLI around pcbdet.engine.trainer).

Usage (from the repository root):
    python scripts/train.py --config configs/train/yolo11s_640.yaml
    python scripts/train.py --config configs/train/smoke_local.yaml          # 2-epoch smoke test
    python scripts/train.py --config ... --run-id yolo11s_640_s0 --resume    # continue a cut-off run

Options:
    --data-root  dataset root (Data_YOLO) when the data lives elsewhere, e.g. on Colab
    --project    where run directories go (default from the config: outputs/runs)
    --run-id     fixed run name instead of the dated default (needed for --resume)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcbdet.config import TrainConfig, resolve_path  # noqa: E402
from pcbdet.engine.trainer import run_training  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.project:
        cfg = replace(cfg, project=resolve_path(args.project))
    run_training(cfg, run_id=args.run_id, data_root=args.data_root, resume=args.resume)


if __name__ == "__main__":
    main()
