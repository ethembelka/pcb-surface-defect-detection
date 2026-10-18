"""Tests for the internal train_fit / val_internal split (no real data needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcbdet.config import SplitConfig
from pcbdet.data.classes import CLASS_NAMES, NUM_CLASSES
from pcbdet.data.splits import (
    ImageRecord,
    make_splits,
    read_label_classes,
    split_records,
    stratify_by_rarest_class,
    to_list_line,
)


def _records(n: int = 200) -> list[ImageRecord]:
    """Synthetic images: class 0 is rare (every 20th image), the rest cycle through 1..8."""
    recs = []
    for i in range(n):
        classes = {1 + (i % (NUM_CLASSES - 1))}
        if i % 20 == 0:
            classes.add(0)
        recs.append(ImageRecord(f"img{i:04d}", Path(f"/data/images/train/img{i:04d}.jpg"), frozenset(classes)))
    return recs


def test_no_overlap_and_nothing_lost():
    recs = _records()
    train, val = split_records(recs, 0.1, seed=0)
    assert set(r.stem for r in train).isdisjoint(r.stem for r in val)
    assert len(train) + len(val) == len(recs)


def test_every_class_present_on_both_sides():
    recs = _records()
    train, val = split_records(recs, 0.1, seed=0)
    for c in range(NUM_CLASSES):
        assert any(c in r.classes for r in train), f"class {CLASS_NAMES[c]} missing from train_fit"
        assert any(c in r.classes for r in val), f"class {CLASS_NAMES[c]} missing from val_internal"


def test_rare_class_gets_its_share():
    recs = _records()
    _, val = split_records(recs, 0.1, seed=0)
    rare_total = sum(0 in r.classes for r in recs)  # 10 images
    rare_val = sum(0 in r.classes for r in val)
    assert rare_val == max(1, round(rare_total * 0.1))


def test_deterministic_for_same_seed_and_different_for_other_seed():
    recs = _records()
    a = split_records(recs, 0.1, seed=0)
    b = split_records(recs, 0.1, seed=0)
    c = split_records(recs, 0.1, seed=1)
    assert [r.stem for r in a[1]] == [r.stem for r in b[1]]
    assert [r.stem for r in a[1]] != [r.stem for r in c[1]]


def test_stratum_is_rarest_class():
    recs = [
        ImageRecord("a", Path("a.jpg"), frozenset({0, 1})),
        ImageRecord("b", Path("b.jpg"), frozenset({1})),
        ImageRecord("c", Path("c.jpg"), frozenset({1})),
        ImageRecord("d", Path("d.jpg"), frozenset()),
    ]
    strata = stratify_by_rarest_class(recs)
    assert [r.stem for r in strata[0]] == ["a"]  # class 0 appears once, class 1 three times
    assert sorted(r.stem for r in strata[1]) == ["b", "c"]
    assert [r.stem for r in strata[-1]] == ["d"]


def test_invalid_ratio_rejected():
    with pytest.raises(ValueError):
        split_records(_records(), 1.0, seed=0)


def test_read_label_classes(tmp_path: Path):
    lbl = tmp_path / "x.txt"
    lbl.write_text("1 0.5 0.5 0.1 0.1\n7 0.2 0.2 0.1 0.1\n1 0.3 0.3 0.1 0.1\n", encoding="utf-8")
    assert read_label_classes(lbl) == frozenset({1, 7})
    assert read_label_classes(tmp_path / "missing.txt") == frozenset()


def test_list_line_is_relative_to_yolo_root_without_dotdot(tmp_path: Path):
    root = tmp_path / "ds"
    line = to_list_line(root / "images" / "train" / "0001.jpg", root)
    assert line == "./images/train/0001.jpg"
    with pytest.raises(ValueError):  # Ultralytics breaks on "../" in list lines
        to_list_line(tmp_path / "elsewhere" / "0001.jpg", root)


def test_make_splits_end_to_end(tmp_path: Path):
    """Tiny fake dataset on disk -> list files resolve back to existing images."""
    yolo = tmp_path / "Data_YOLO"
    for i in range(40):
        (yolo / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (yolo / "images" / "train" / f"{i:04d}.jpg").write_bytes(b"")
        cls = i % NUM_CLASSES
        (yolo / "labels" / "train" / f"{i:04d}.txt").write_text(f"{cls} 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    out = tmp_path / "splits"
    cfg = SplitConfig(yolo_root=yolo, source_split="train", val_ratio=0.25, seed=0, out_dir=out)
    stats = make_splits(cfg)

    assert stats["n_total"] == 40
    assert stats["n_train_fit"] + stats["n_val_internal"] == 40
    for name in ("train_fit.txt", "val_internal.txt"):
        lines = (out / name).read_text(encoding="utf-8").splitlines()
        assert lines and all(ln.startswith("./images/") and "../" not in ln for ln in lines)
        assert all((yolo / ln[2:]).exists() for ln in lines)
        # copy inside the dataset tree is byte-identical (this is what Ultralytics reads)
        assert (yolo / name).read_bytes() == (out / name).read_bytes()
    assert (out / "split_stats.json").exists()
