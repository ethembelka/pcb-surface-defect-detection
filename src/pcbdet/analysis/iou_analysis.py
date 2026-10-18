"""Localisation quality: distribution of the IoU between found GT boxes and their predictions.

Explains the gap between AP50 and AP50:95. A class that is *found* (high AP50) but whose boxes
are drawn loosely (many matches with IoU < 0.75) loses most of its AP at the stricter IoU
thresholds. Small objects are especially sensitive: a 2-3 px offset on a 15 px box already costs
~0.2 IoU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pcbdet.analysis.gt_matching import GTRecord
from pcbdet.data.classes import CLASS_NAMES
from pcbdet.data.stats import SIZE_BUCKETS


def iou_summary(gt_records: list[GTRecord]) -> list[dict]:
    """Per class (and per size inside the class): n_found, median IoU, share of matches with IoU < 0.75 / >= 0.9."""
    rows = []
    for cat in range(1, len(CLASS_NAMES) + 1):
        for size in ("all",) + SIZE_BUCKETS:
            ious = np.array([r.iou for r in gt_records if r.matched and r.category_id == cat
                             and (size == "all" or r.size == size)])
            rows.append({
                "class": CLASS_NAMES[cat - 1], "size": size, "n_found": int(len(ious)),
                "median_iou": round(float(np.median(ious)), 4) if len(ious) else None,
                "mean_iou": round(float(ious.mean()), 4) if len(ious) else None,
                "share_iou_lt_0.75": round(float((ious < 0.75).mean()), 4) if len(ious) else None,
                "share_iou_ge_0.90": round(float((ious >= 0.90).mean()), 4) if len(ious) else None,
            })
    return rows


def iou_figure(gt_records: list[GTRecord], path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [[r.iou for r in gt_records if r.matched and r.category_id == c] for c in range(1, len(CLASS_NAMES) + 1)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(data, tick_labels=list(CLASS_NAMES), showfliers=False)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.axhline(0.75, color="gray", ls=":", lw=0.8)
    ax.set_ylabel("IoU of matched boxes")
    ax.set_ylim(0.45, 1.0)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
