"""Tests for run bookkeeping, the model registry and TrainConfig loading."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from pcbdet.config import TrainConfig
from pcbdet.models.registry import get_detector
from pcbdet.models.ultralytics_detector import UltralyticsDetector
from pcbdet.utils.run import init_run_dir, make_run_id


def test_run_id_format():
    assert make_run_id("yolo11s.pt", 640, 0, dt.date(2026, 11, 1)) == "20261101_yolo11s_640_s0"


def test_init_run_dir_writes_config_and_hash(tmp_path: Path):
    run_dir = init_run_dir(tmp_path, "r1", {"a": 1})
    assert (run_dir / "config.yaml").read_text(encoding="utf-8").strip() == "a: 1"
    assert (run_dir / "git_hash.txt").read_text(encoding="utf-8").strip()


def test_registry_resolves_yolo_and_rejects_unknown():
    assert isinstance(get_detector("yolo11s.pt"), UltralyticsDetector)
    assert isinstance(get_detector("rtdetr-l.pt"), UltralyticsDetector)
    with pytest.raises(KeyError):
        get_detector("fasterrcnn.pt")


def test_run_imgsz_read_from_run_config(tmp_path: Path):
    from pcbdet.evaluation.evaluator import run_imgsz

    run_dir = init_run_dir(tmp_path, "r320", {"train": {"model": "yolo11s.pt", "imgsz": 320}})
    assert run_imgsz(run_dir, default=640) == 320
    assert run_imgsz(tmp_path / "missing", default=640) == 640


def test_train_config_defaults_and_paths(tmp_path: Path):
    yml = tmp_path / "t.yaml"
    yml.write_text("model: yolo11s.pt\ndata: configs/dspcbsd.yaml\nimgsz: 640\nepochs: 1\nbatch: 8\n", encoding="utf-8")
    cfg = TrainConfig.from_yaml(yml)
    assert cfg.data.is_absolute() and cfg.project.is_absolute()
    assert cfg.fraction == 1.0 and cfg.device is None and cfg.variant == ""
    assert isinstance(cfg.as_dict()["data"], str)
