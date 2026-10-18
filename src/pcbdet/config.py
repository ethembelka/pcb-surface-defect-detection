"""YAML config loading into typed dataclasses.

Purpose
-------
Project rule (CLAUDE.md): no path, threshold or hyperparameter is hard-coded. Every setting is
read from a YAML file under ``configs/`` and turned into a small *typed* object here, so the rest
of the code gets attribute access with IDE support instead of raw dict lookups.

Why paths are resolved against the repo root
--------------------------------------------
Configs store paths relative to the repository (``data/DsPCBSD/...``) so they work on any
machine and on Colab. Ultralytics, however, resolves a relative dataset ``path`` against its own
``datasets`` directory (``~/datasets``), not the current folder. ``resolve_path`` therefore makes
every path absolute *before* anything is handed to Ultralytics.

Contents
--------
- ``SplitConfig``   -> configs/splits.yaml   (internal train_fit / val_internal split)
- ``DatasetConfig`` -> configs/dspcbsd*.yaml (Ultralytics dataset description)
Training / evaluation dataclasses will be added in later steps (A1.4, A3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# .../pcb-surface-defect-detection  (this file lives in src/pcbdet/, two levels below the root)
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a plain dict (empty file -> empty dict)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(p: str | Path) -> Path:
    """Make a config path absolute, relative to the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@dataclass(frozen=True)
class SplitConfig:
    """Settings for the internal train_fit / val_internal split (configs/splits.yaml).

    ``frozen=True``: a config is read once and never mutated, which keeps runs reproducible.
    """

    yolo_root: Path  # Data_YOLO directory containing images/ and labels/
    source_split: str  # official split to divide (always "train"; official val is the test set)
    val_ratio: float  # fraction of images held out as val_internal (0.10)
    seed: int  # shuffle seed; same seed + same files -> identical split
    out_dir: Path  # git-tracked list files are written here (data/splits)
    train_list: str = "train_fit.txt"
    val_list: str = "val_internal.txt"
    stats_file: str = "split_stats.json"

    @staticmethod
    def from_yaml(path: str | Path) -> "SplitConfig":
        d = load_yaml(path)
        return SplitConfig(
            yolo_root=resolve_path(d["yolo_root"]),
            source_split=d.get("source_split", "train"),
            val_ratio=float(d["val_ratio"]),
            seed=int(d.get("seed", 0)),
            out_dir=resolve_path(d["out_dir"]),
            train_list=d.get("train_list", "train_fit.txt"),
            val_list=d.get("val_list", "val_internal.txt"),
            stats_file=d.get("stats_file", "split_stats.json"),
        )


@dataclass(frozen=True)
class DatasetConfig:
    """Ultralytics dataset yaml (configs/dspcbsd*.yaml) with ``path`` made absolute.

    Two files exist on purpose:
    - ``dspcbsd.yaml``      train = train_fit, val = val_internal  -> training & checkpoint selection
    - ``dspcbsd_test.yaml`` val = official val                     -> final reporting only
    """

    path: Path  # dataset root (Data_YOLO); ``train``/``val`` are relative to it
    train: str  # directory or .txt list of images
    val: str
    names: dict[int, str] = field(default_factory=dict)  # YOLO class id -> short name

    @staticmethod
    def from_yaml(path: str | Path) -> "DatasetConfig":
        d = load_yaml(path)
        return DatasetConfig(
            path=resolve_path(d["path"]),
            train=str(d["train"]),
            val=str(d["val"]),
            names={int(k): str(v) for k, v in d["names"].items()},
        )

    def to_ultralytics_dict(self) -> dict[str, Any]:
        """Dict ready to be dumped to a temporary yaml for Ultralytics (absolute ``path``)."""
        return {"path": str(self.path), "train": self.train, "val": self.val, "names": self.names}

    def with_root(self, new_root: str | Path | None) -> "DatasetConfig":
        """Same config with another dataset root (Colab unpacks the data elsewhere)."""
        if new_root is None:
            return self
        return DatasetConfig(path=resolve_path(new_root), train=self.train, val=self.val, names=self.names)


@dataclass(frozen=True)
class TrainConfig:
    """One training run (configs/train/*.yaml). Field names follow the Ultralytics arguments.

    ``fraction`` < 1 trains on a subset — used only for the smoke test, never for reported runs.
    """

    model: str  # weights file or model name, e.g. "yolo11s.pt" (COCO-pretrained)
    data: Path  # dataset yaml (configs/dspcbsd.yaml)
    imgsz: int
    epochs: int
    batch: int
    seed: int
    patience: int  # early-stopping patience (epochs without val improvement)
    workers: int  # dataloader processes
    optimizer: str  # "auto" lets Ultralytics pick (SGD/AdamW) from the schedule
    cache: bool  # cache images in RAM
    fraction: float = 1.0
    project: Path = Path("outputs/runs")  # run directories are created under here
    device: str | None = None  # None -> auto (GPU if available)
    variant: str = ""  # free label appended to the run_id, e.g. "smoke"

    @staticmethod
    def from_yaml(path: str | Path) -> "TrainConfig":
        d = load_yaml(path)
        return TrainConfig(
            model=str(d["model"]),
            data=resolve_path(d["data"]),
            imgsz=int(d["imgsz"]),
            epochs=int(d["epochs"]),
            batch=int(d["batch"]),
            seed=int(d.get("seed", 0)),
            patience=int(d.get("patience", 100)),
            workers=int(d.get("workers", 4)),
            optimizer=str(d.get("optimizer", "auto")),
            cache=bool(d.get("cache", False)),
            fraction=float(d.get("fraction", 1.0)),
            project=resolve_path(d.get("project", "outputs/runs")),
            device=None if d.get("device") in (None, "auto") else str(d["device"]),
            variant=str(d.get("variant", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        """Plain dict (paths as strings) for the config snapshot written next to the run."""
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation settings (configs/eval/default.yaml). See the yaml for the meaning of each block."""

    datasets: dict[str, Path]  # split name -> dataset yaml
    coco_annotations_dir: Path
    coco_json: dict[str, str]  # split name -> annotation file name
    imgsz: int
    conf: float
    iou_nms: float
    max_det: int
    half: bool
    batch: int
    iou_match: float
    decision_mode: str
    decision_select_on: str
    decision_grid_step: float
    eff_batch: int
    eff_half: bool
    eff_warmup: int
    eff_iters: int
    results_dir: Path
    tables_dir: Path
    figures_dir: Path
    references: Path
    raw: dict[str, Any] = field(default_factory=dict, compare=False)  # verbatim yaml for the snapshot

    @staticmethod
    def from_yaml(path: str | Path) -> "EvalConfig":
        d = load_yaml(path)
        inf, m, dec, eff = d["inference"], d["matching"], d["decision"], d["efficiency"]
        return EvalConfig(
            datasets={k: resolve_path(v) for k, v in d["datasets"].items()},
            coco_annotations_dir=resolve_path(d["coco_annotations_dir"]),
            coco_json=dict(d["coco_json"]),
            imgsz=int(inf["imgsz"]),
            conf=float(inf["conf"]),
            iou_nms=float(inf["iou_nms"]),
            max_det=int(inf["max_det"]),
            half=bool(inf["half"]),
            batch=int(inf["batch"]),
            iou_match=float(m["iou_match"]),
            decision_mode=str(dec["mode"]),
            decision_select_on=str(dec["select_on"]),
            decision_grid_step=float(dec.get("grid_step", 0.01)),
            eff_batch=int(eff["batch"]),
            eff_half=bool(eff["half"]),
            eff_warmup=int(eff["warmup"]),
            eff_iters=int(eff["iters"]),
            results_dir=resolve_path(d["results_dir"]),
            tables_dir=resolve_path(d["tables_dir"]),
            figures_dir=resolve_path(d["figures_dir"]),
            references=resolve_path(d["references"]),
            raw=d,
        )
