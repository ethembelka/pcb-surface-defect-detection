"""Internal train_fit / val_internal split of the official DsPCBSD+ training set.

Why: the dataset has no test set, so the official *val* split is reserved for final reporting
(docs/EVALUATION_PROTOCOL.md). Checkpoint selection therefore needs its own held-out part,
carved out of the official *train* split.

How: every image is assigned to a stratum = its rarest class (by image-level frequency), so
that each class ends up with roughly ``val_ratio`` of its images in val_internal. Within each
stratum the images are shuffled with a fixed seed and the first ``val_ratio`` share goes to
val_internal. The result is deterministic for a given (seed, image list).

The list files are written in the Ultralytics txt format: one image path per line, prefixed
with ``./`` so that Ultralytics resolves it relative to the list file's directory. Ultralytics
replaces *every* ``./`` in a line (so ``../`` breaks); therefore the lists are copied to the
dataset root itself (``<yolo_root>/train_fit.txt``) and contain only ``./images/<split>/<file>``
lines. The git-tracked copy in ``out_dir`` has identical content and is the canonical one.
"""

from __future__ import annotations

import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from pcbdet.config import SplitConfig
from pcbdet.data.classes import CLASS_NAMES, NUM_CLASSES
from pcbdet.utils.io import write_json, write_lines

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class ImageRecord:
    stem: str
    image_path: Path
    classes: frozenset[int]  # YOLO class ids present in the image


def read_label_classes(label_path: Path) -> frozenset[int]:
    """Class ids in a YOLO label file (missing/empty file -> empty set)."""
    if not label_path.exists():
        return frozenset()
    ids: set[int] = set()
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts:
                ids.add(int(parts[0]))
    return frozenset(ids)


def scan_split(yolo_root: Path, split: str) -> list[ImageRecord]:
    """All images of ``<yolo_root>/images/<split>`` with their label classes, sorted by stem."""
    img_dir = yolo_root / "images" / split
    lbl_dir = yolo_root / "labels" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {img_dir}")
    records = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        records.append(ImageRecord(p.stem, p, read_label_classes(lbl_dir / f"{p.stem}.txt")))
    return records


def stratify_by_rarest_class(records: list[ImageRecord]) -> dict[int, list[ImageRecord]]:
    """Group images by their rarest class (image-level frequency); -1 = no labels."""
    freq = Counter(c for r in records for c in r.classes)
    strata: dict[int, list[ImageRecord]] = defaultdict(list)
    for r in records:
        key = min(r.classes, key=lambda c: (freq[c], c)) if r.classes else -1
        strata[key].append(r)
    return strata


def split_records(
    records: list[ImageRecord], val_ratio: float, seed: int
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Deterministic stratified split -> (train_fit, val_internal), each sorted by stem."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    rng = random.Random(seed)
    train, val = [], []
    strata = stratify_by_rarest_class(records)
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: r.stem)
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_ratio)) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    key = lambda r: r.stem  # noqa: E731
    return sorted(train, key=key), sorted(val, key=key)


def class_image_counts(records: list[ImageRecord]) -> dict[str, int]:
    """Number of images containing each class."""
    cnt = Counter(c for r in records for c in r.classes)
    return {CLASS_NAMES[c]: cnt.get(c, 0) for c in range(NUM_CLASSES)}


def to_list_line(image_path: Path, yolo_root: Path) -> str:
    """``./images/<split>/<file>`` relative to ``yolo_root`` (never contains ``..``)."""
    rel = os.path.relpath(image_path, yolo_root).replace(os.sep, "/")
    if rel.startswith(".."):
        raise ValueError(f"{image_path} is not inside {yolo_root}")
    return f"./{rel}"


def make_splits(cfg: SplitConfig) -> dict:
    """Run the split, write list files (tracked copy + copy inside the dataset tree) and stats."""
    records = scan_split(cfg.yolo_root, cfg.source_split)
    train, val = split_records(records, cfg.val_ratio, cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_lines((to_list_line(r.image_path, cfg.yolo_root) for r in train), cfg.out_dir / cfg.train_list)
    write_lines((to_list_line(r.image_path, cfg.yolo_root) for r in val), cfg.out_dir / cfg.val_list)
    for name in (cfg.train_list, cfg.val_list):  # copies Ultralytics reads: "./" = list's own dir
        shutil.copyfile(cfg.out_dir / name, cfg.yolo_root / name)
    stats = {
        "source_split": cfg.source_split,
        "val_ratio": cfg.val_ratio,
        "seed": cfg.seed,
        "n_total": len(records),
        "n_train_fit": len(train),
        "n_val_internal": len(val),
        "images_per_class": {
            "train_fit": class_image_counts(train),
            "val_internal": class_image_counts(val),
        },
    }
    write_json(stats, cfg.out_dir / cfg.stats_file)
    return stats
