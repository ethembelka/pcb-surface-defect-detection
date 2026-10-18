"""``Detector`` implementation for Ultralytics models (YOLO11 and, later, RT-DETR).

What it does
------------
- Writes the dataset description (with an *absolute* path) to ``<run_dir>/data.yaml`` and
  hands that file to Ultralytics, so the library never sees a relative path it would resolve
  against its own datasets directory.
- Trains with the arguments from ``TrainConfig``; Ultralytics writes weights, ``results.csv``
  and plots directly into ``run_dir`` (``project`` = parent, ``name`` = run_id).
- Resumes from ``<run_dir>/weights/last.pt`` when asked, which is how a Colab session that was
  cut off continues where it stopped.
- After training, records what we need for the time decision in ``train_summary.json``:
  seconds per epoch, total time and peak GPU memory.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from pcbdet.config import DatasetConfig, TrainConfig
from pcbdet.utils.io import write_json, write_yaml


class UltralyticsDetector:
    name = "ultralytics"

    def train(self, cfg: TrainConfig, dataset: DatasetConfig, run_dir: Path, resume: bool = False) -> Path:
        from ultralytics import YOLO  # imported here so data-only scripts don't need torch

        run_dir.mkdir(parents=True, exist_ok=True)
        data_yaml = write_yaml(dataset.to_ultralytics_dict(), run_dir / "data.yaml")
        last = run_dir / "weights" / "last.pt"

        t0 = time.perf_counter()
        if resume and last.exists():
            YOLO(str(last)).train(resume=True)
        else:
            YOLO(cfg.model).train(
                data=str(data_yaml),
                epochs=cfg.epochs,
                imgsz=cfg.imgsz,
                batch=cfg.batch,
                seed=cfg.seed,
                patience=cfg.patience,
                workers=cfg.workers,
                optimizer=cfg.optimizer,
                cache=cfg.cache,
                fraction=cfg.fraction,
                device=cfg.device,
                project=str(run_dir.parent),
                name=run_dir.name,
                exist_ok=True,
                deterministic=True,
                plots=True,
                verbose=True,
            )
        elapsed = time.perf_counter() - t0

        write_json(self._summary(run_dir, elapsed), run_dir / "train_summary.json")
        return run_dir

    @staticmethod
    def _summary(run_dir: Path, elapsed_s: float) -> dict:
        """Epoch count/time from Ultralytics' results.csv plus peak GPU memory."""
        epochs_done, best_map50, best_map = 0, None, None
        results = run_dir / "results.csv"
        if results.exists():
            with open(results, encoding="utf-8") as f:
                rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
            epochs_done = len(rows)
            if rows:
                best_map50 = max(float(r.get("metrics/mAP50(B)", 0)) for r in rows)
                best_map = max(float(r.get("metrics/mAP50-95(B)", 0)) for r in rows)
        peak_vram_gb = None
        try:
            import torch

            if torch.cuda.is_available():
                peak_vram_gb = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        except ImportError:
            pass
        return {
            "epochs_done": epochs_done,
            "total_time_s": round(elapsed_s, 1),
            "time_per_epoch_s": round(elapsed_s / epochs_done, 1) if epochs_done else None,
            "best_val_internal_map50": best_map50,
            "best_val_internal_map50_95": best_map,
            "peak_vram_gb": peak_vram_gb,
        }

    def export_predictions(self, weights: Path, dataset: DatasetConfig, split: str, out_json: Path) -> Path:
        raise NotImplementedError("implemented in phase A3 (evaluation)")
