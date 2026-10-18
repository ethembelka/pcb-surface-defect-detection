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


def remap_ultralytics_ids(raw: list[dict], stem_to_id: dict[str, int]) -> list[dict]:
    """Rewrite the ``image_id`` of an Ultralytics ``save_json`` file to COCO ids.

    Ultralytics writes ``image_id = int(stem)`` when the stem is numeric (so ``0003730`` becomes
    ``3730``) and the stem string otherwise (``Y_011791``). We therefore look up both the exact
    stem and, for numeric ones, the zero-stripped form. Ambiguity (two stems differing only in
    leading zeros) raises instead of silently mis-assigning.
    """
    by_stripped: dict[str, list[str]] = {}
    for stem in stem_to_id:
        if stem.isdigit():
            by_stripped.setdefault(stem.lstrip("0") or "0", []).append(stem)
    out = []
    for p in raw:
        key = str(p["image_id"])
        if key.isdigit():  # numeric id: every stem that strips to it is a candidate, must be unique
            cands = by_stripped.get(key.lstrip("0") or "0", [])
            if len(cands) != 1:
                raise KeyError(f"cannot map Ultralytics image_id {key!r} to a unique COCO image (candidates {cands})")
            stem = cands[0]
        elif key in stem_to_id:
            stem = key
        else:
            raise KeyError(f"Ultralytics image_id {key!r} not found among COCO image stems")
        out.append({"image_id": stem_to_id[stem], "category_id": int(p["category_id"]),
                    "bbox": [round(float(v), 2) for v in p["bbox"]], "score": round(float(p["score"]), 5)})
    return out


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
