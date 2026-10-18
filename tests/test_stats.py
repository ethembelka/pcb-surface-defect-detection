"""Tests for the class x size statistics (synthetic COCO json, no real data)."""

from __future__ import annotations

import json
from pathlib import Path

from pcbdet.data.stats import count_boxes, size_bucket, stems_from_list


def test_size_bucket_boundaries_match_coco():
    assert size_bucket(1023) == "small"
    assert size_bucket(1024) == "medium"  # 32*32 is medium in pycocotools ([32^2, 96^2])
    assert size_bucket(9216) == "medium"  # 96*96 still medium
    assert size_bucket(9217) == "large"


def test_count_boxes_filters_by_stem_and_buckets(tmp_path: Path):
    coco = {
        "images": [{"id": 0, "file_name": "a.jpg"}, {"id": 1, "file_name": "Y_b.jpg"}],
        "annotations": [
            {"image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100},  # SH small
            {"image_id": 0, "category_id": 2, "bbox": [0, 0, 50, 50], "area": 2500},  # SP medium
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 100, 100], "area": 10000},  # SP large (excluded)
        ],
        "categories": [{"id": i, "name": n} for i, n in enumerate(["SH", "SP"], start=1)],
    }
    js = tmp_path / "inst.json"
    js.write_text(json.dumps(coco), encoding="utf-8")

    only_a = count_boxes(js, {"a"}, 1024, 9216)
    assert only_a["SH"]["small"] == 1 and only_a["SP"]["medium"] == 1 and only_a["SP"]["large"] == 0
    everything = count_boxes(js, None, 1024, 9216)
    assert everything["SP"]["large"] == 1


def test_stems_from_list(tmp_path: Path):
    lst = tmp_path / "l.txt"
    lst.write_text("./images/train/0003730.jpg\n./images/train/Y_011791.jpg\n", encoding="utf-8")
    assert stems_from_list(lst) == {"0003730", "Y_011791"}
