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
from pcbdet.utils.io import read_json, write_json, write_yaml


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

    def export_predictions(
        self,
        weights: Path,
        image_paths: list[Path],
        stem_to_id: dict[str, int],
        out_json: Path,
        imgsz: int = 640,
        conf: float = 0.001,
        iou_nms: float = 0.7,
        max_det: int = 300,
        half: bool = True,
        batch: int = 32,
    ) -> Path:
        """Run inference with the Ultralytics *validator* pipeline and write a COCO results json.

        Why the validator and not ``model.predict``: the validator is the pipeline the checkpoint
        was selected with (rectangular letterbox with padding, multi-label class-wise NMS, conf
        0.001, NMS IoU 0.7) and it matches how reference detectors are evaluated (class-wise NMS).
        ``predict`` (single label per box, no padding) gave ~2.6 AP50 points less on the same
        weights (measured 2026-10-19), so mixing the two would make our numbers incomparable.

        The validator's ``save_json`` output uses the image *file stem* as ``image_id`` (and drops
        leading zeros of numeric stems); we rewrite it to the COCO json's numeric ids via
        ``stem_to_id`` (see data/coco.py). ``category_id`` is already 1-based in that file.
        """
        import tempfile

        from ultralytics import YOLO

        from pcbdet.data.coco import remap_ultralytics_ids

        work = Path(tempfile.mkdtemp(prefix="pcbdet_val_"))
        list_file = work / "images.txt"
        list_file.write_text("".join(f"{p.resolve()}\n" for p in image_paths), encoding="utf-8")
        data_yaml = write_yaml({"path": str(work), "train": "images.txt", "val": "images.txt",
                                "names": {i: n for i, n in enumerate(_class_names())}}, work / "data.yaml")

        model = YOLO(str(weights))
        model.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=batch, workers=0, conf=conf,
                  iou=iou_nms, max_det=max_det, half=half, save_json=True, plots=False, verbose=False,
                  project=str(work), name="val", exist_ok=True)
        raw = read_json(work / "val" / "predictions.json")
        results = remap_ultralytics_ids(raw, stem_to_id)
        write_json(results, out_json, indent=None)
        return out_json


def _class_names() -> tuple[str, ...]:
    from pcbdet.data.classes import CLASS_NAMES

    return CLASS_NAMES
