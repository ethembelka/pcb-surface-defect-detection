"""Run bookkeeping: run_id, output directory, config snapshot and git hash."""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path
from typing import Any

from pcbdet.config import REPO_ROOT
from pcbdet.utils.io import write_yaml


def git_hash(short: bool = False) -> str:
    """Current commit hash (``unknown`` outside a git checkout)."""
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_run_id(model: str, imgsz: int, seed: int, date: dt.date | None = None) -> str:
    """``{YYYYMMDD}_{model}_{imgsz}_s{seed}`` as defined in docs/ARCHITECTURE.md."""
    date = date or dt.date.today()
    model = model.replace(".pt", "")
    return f"{date:%Y%m%d}_{model}_{imgsz}_s{seed}"


def init_run_dir(base: str | Path, run_id: str, config: dict[str, Any]) -> Path:
    """Create ``<base>/<run_id>/`` and write ``config.yaml`` and ``git_hash.txt`` into it."""
    run_dir = Path(base) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(config, run_dir / "config.yaml")
    (run_dir / "git_hash.txt").write_text(git_hash() + "\n", encoding="utf-8")
    return run_dir
