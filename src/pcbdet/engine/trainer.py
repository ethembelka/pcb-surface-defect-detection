"""Training flow: config -> run directory -> detector.train() -> summary.

This is the single place that knows how a training run is set up. ``scripts/train.py`` only
parses arguments and calls ``run_training``; the Colab notebook calls the same script, so local
and cloud runs are configured identically.

Steps
-----
1. Fix the seed.
2. Build the run_id (``{date}_{model}_{imgsz}_s{seed}[_variant]``) unless one is given
   (a fixed run_id is what makes ``--resume`` find the previous run).
3. Create ``<project>/<run_id>/`` with ``config.yaml`` + ``git_hash.txt`` (reproducibility).
4. Train through the ``Detector`` interface; return the run directory.
"""

from __future__ import annotations

from pathlib import Path

from pcbdet.config import DatasetConfig, TrainConfig
from pcbdet.models.registry import get_detector
from pcbdet.utils.io import read_json
from pcbdet.utils.run import init_run_dir, make_run_id
from pcbdet.utils.seed import set_seed


def run_training(
    cfg: TrainConfig,
    run_id: str | None = None,
    data_root: str | Path | None = None,
    resume: bool = False,
) -> Path:
    set_seed(cfg.seed)
    dataset = DatasetConfig.from_yaml(cfg.data).with_root(data_root)

    if run_id is None:
        run_id = make_run_id(cfg.model, cfg.imgsz, cfg.seed)
        if cfg.variant:
            run_id = f"{run_id}_{cfg.variant}"
    run_dir = init_run_dir(cfg.project, run_id, {"train": cfg.as_dict(), "dataset": dataset.to_ultralytics_dict()})

    detector = get_detector(cfg.model)
    detector.train(cfg, dataset, run_dir, resume=resume)

    summary = read_json(run_dir / "train_summary.json")
    print(f"\nrun_id: {run_id}\nrun_dir: {run_dir}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return run_dir
