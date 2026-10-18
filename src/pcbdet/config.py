"""YAML config loading into typed dataclasses.

All paths, thresholds and hyperparameters live in ``configs/*.yaml``; nothing is hard-coded.
Relative paths in configs are resolved against the repository root so that commands work
from any working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a plain dict (empty file -> empty dict)."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(p: str | Path) -> Path:
    """Make a config path absolute, relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@dataclass(frozen=True)
class SplitConfig:
    """Settings for the internal train_fit / val_internal split (configs/splits.yaml)."""

    yolo_root: Path  # Data_YOLO directory containing images/ and labels/
    source_split: str  # official split to divide (train)
    val_ratio: float  # fraction of images moved to val_internal
    seed: int
    out_dir: Path  # where the list files are written
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
    """Ultralytics dataset yaml (configs/dspcbsd*.yaml) with paths made absolute."""

    path: Path
    train: str
    val: str
    names: dict[int, str] = field(default_factory=dict)

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
        """Dict with an absolute ``path``; Ultralytics resolves a relative ``path`` against
        its own datasets directory, not the repository, so we never pass a relative one."""
        return {"path": str(self.path), "train": self.train, "val": self.val, "names": self.names}
