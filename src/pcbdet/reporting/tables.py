"""Report tables and the confusion-matrix figure from a metrics.json.

Numbers are never typed by hand: every table is regenerated from ``metrics.json`` files, and the
reference rows come from configs/references.yaml. Outputs go to reports/tables and
reports/figures with the run_id and split in the file name.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from pcbdet.config import load_yaml
from pcbdet.data.classes import CLASS_NAMES

MAIN_COLS = ["model", "input", "split", "ap50", "ap75", "ap50_95", "ap_small", "ap_medium", "ap_large", "comparable"]


def main_results_rows(metrics: dict[str, Any], references_yaml: Path, label: str) -> list[dict]:
    """Our run first, then same-split references, then flagged different-split literature."""
    ours = {"model": label, "input": f"{metrics.get('imgsz', '')}", "split": metrics["split"],
            "ap50": metrics["map50"], "ap75": metrics["ap75"], "ap50_95": metrics["map50_95"],
            "ap_small": metrics["ap_small"], "ap_medium": metrics["ap_medium"], "ap_large": metrics["ap_large"],
            "comparable": "ours"}
    rows = [ours]
    for r in load_yaml(references_yaml).get("references", []):
        rows.append({"model": r["name"], "input": r.get("input", ""), "split": "official val",
                     "ap50": r.get("ap50", ""), "ap75": r.get("ap75", ""), "ap50_95": r.get("ap50_95", ""),
                     "ap_small": r.get("ap_small", ""), "ap_medium": r.get("ap_medium", ""),
                     "ap_large": r.get("ap_large", ""),
                     "comparable": "yes" if r.get("same_split") else "no (different split/protocol)"})
    return rows


def per_class_rows(metrics: dict[str, Any], references_yaml: Path) -> list[dict]:
    refs = [r for r in load_yaml(references_yaml).get("references", []) if r.get("per_class_ap50")]
    rows = []
    for name in CLASS_NAMES:
        m = metrics["per_class"][name]
        row = {"class": name, "ap50": m["ap50"], "ap50_95": m["ap50_95"], "p": m.get("p"), "r": m.get("r"),
               "f1": m.get("f1"), "n_gt": m.get("n_gt")}
        for r in refs:
            row[f"ap50_{r['name']}"] = r["per_class_ap50"].get(name, "")
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def confusion_figure(metrics: dict[str, Any], path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = metrics["confusion"]["labels"]
    n = np.array(metrics["confusion"]["row_normalized"])
    c = np.array(metrics["confusion"]["counts"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if c[i, j]:
                ax.text(j, i, f"{n[i, j]:.2f}\n({c[i, j]})", ha="center", va="center", fontsize=7,
                        color="white" if n[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report_tables(metrics: dict[str, Any], cfg_tables_dir: Path, cfg_figures_dir: Path,
                        references_yaml: Path, label: str) -> list[Path]:
    tag = f"{metrics['run_id']}_{metrics['split']}"
    out = [
        write_csv(main_results_rows(metrics, references_yaml, label), cfg_tables_dir / f"main_results_{tag}.csv"),
        write_csv(per_class_rows(metrics, references_yaml), cfg_tables_dir / f"per_class_{tag}.csv"),
        confusion_figure(metrics, cfg_figures_dir / f"confusion_matrix_{tag}.png",
                         f"{label} — {metrics['split']} (conf ≥ {metrics['decision_threshold']['conf']})"),
    ]
    return out
