"""The ``Detector`` interface every model implementation must follow.

Why an interface
----------------
Training, evaluation and analysis code depend only on this contract, not on Ultralytics
(dependency inversion, docs/ARCHITECTURE.md). Adding another detector family later (e.g.
RT-DETR in phase A5) means writing one new class that implements these methods and registering
it in ``registry.py``; nothing else changes.

The methods
-----------
- ``train``  runs one training job and returns the directory holding ``weights/best.pt``.
- ``export_predictions`` writes detections on a split as a COCO *results* json
  (``[{image_id, category_id, bbox, score}, ...]``), which is the only thing the evaluator reads.
  It is implemented in phase A3 together with the evaluation code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pcbdet.config import DatasetConfig, TrainConfig  # noqa: F401  (DatasetConfig is part of the train signature)


class Detector(Protocol):
    name: str

    def train(self, cfg: TrainConfig, dataset: DatasetConfig, run_dir: Path, resume: bool = False) -> Path:
        """Train and return the directory that contains ``weights/best.pt`` and ``results.csv``."""
        ...

    def export_predictions(
        self, weights: Path, image_paths: list[Path], stem_to_id: dict[str, int], out_json: Path, **infer
    ) -> Path:
        """Run inference on ``image_paths`` and write a COCO results json whose ``image_id`` values
        come from ``stem_to_id``. ``infer`` holds imgsz / conf / iou_nms / max_det / half / batch."""
        ...
