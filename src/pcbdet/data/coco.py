"""COCO ground-truth helpers: image lists per split, file-name -> image_id map, filtered GT.

Why this module exists (open issue E3)
--------------------------------------
Image files are named two ways in DsPCBSD+ (``0003730.jpg`` and ``Y_011791.jpg``) while the
COCO json gives every image a *numeric* ``id``. pycocotools matches predictions to ground truth
by that numeric id, so a prediction file must carry the json's ids, not file names. If the ids do
not line up, pycocotools silently reports ~0 AP. Everything that touches ids goes through here
and is covered by tests.

Splits and ground truth
-----------------------
- ``test``         -> all images of ``instances_val2017.json`` (official val).
- ``val_internal`` -> the subset of ``instances_train2017.json`` whose stems are in
                      ``val_internal.txt``; ``filter_gt`` builds that subset as a COCO dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pcbdet.config import DatasetConfig
from pcbdet.utils.io import read_json, read_lines

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def split_image_paths(dataset: DatasetConfig, split: str = "val") -> list[Path]:
    """Image files of ``dataset.<split>``: a directory or an Ultralytics ``.txt`` list (``./`` = list dir)."""
    entry = getattr(dataset, split)
    p = dataset.path / entry
    if p.is_dir():
        return sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
    if p.is_file():
        base = p.parent
        return [(base / ln[2:]) if ln.startswith("./") else Path(ln) for ln in read_lines(p)]
    raise FileNotFoundError(f"{split} entry of dataset not found: {p}")


def stem_to_image_id(coco: dict[str, Any]) -> dict[str, int]:
    """``{file stem: image_id}`` from a COCO dict (stems are unique in DsPCBSD+; checked)."""
    m = {Path(im["file_name"]).stem: int(im["id"]) for im in coco["images"]}
    if len(m) != len(coco["images"]):
        raise ValueError("duplicate image stems in COCO file; stem-based id mapping is not safe")
    return m


def filter_gt(coco: dict[str, Any], stems: set[str]) -> dict[str, Any]:
    """COCO dict restricted to the images whose stems are in ``stems`` (keeps original ids)."""
    images = [im for im in coco["images"] if Path(im["file_name"]).stem in stems]
    keep = {im["id"] for im in images}
    anns = [a for a in coco["annotations"] if a["image_id"] in keep]
    return {"images": images, "annotations": anns, "categories": coco["categories"]}


def load_gt_for_split(coco_json: Path, image_paths: list[Path]) -> dict[str, Any]:
    """Ground truth for exactly the given images (asserts every image is in the json)."""
    coco = read_json(coco_json)
    stems = {p.stem for p in image_paths}
    gt = filter_gt(coco, stems)
    missing = stems - {Path(im["file_name"]).stem for im in gt["images"]}
    if missing:
        raise ValueError(f"{len(missing)} images have no entry in {coco_json.name}, e.g. {sorted(missing)[:3]}")
    return gt
