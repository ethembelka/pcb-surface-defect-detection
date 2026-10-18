"""File I/O helpers.

Project rule: side effects (writing to disk) live at the edges. All JSON/YAML/text writes go
through these functions so that encoding (UTF-8), directory creation and line endings are
handled in one place and the rest of the code stays pure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Write ``obj`` as pretty JSON (UTF-8, non-ASCII kept readable); creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # LF on every OS: same bytes on Colab and Windows
        json.dump(obj, f, indent=indent, ensure_ascii=False)
        f.write("\n")
    return path


def write_yaml(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as YAML keeping key order (used for run config snapshots)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    return path


def read_lines(path: str | Path) -> list[str]:
    """Non-empty, stripped lines of a text file."""
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_lines(lines: Iterable[str], path: str | Path) -> Path:
    """One item per line with ``\\n`` endings on every OS (keeps git diffs clean on Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for ln in lines:
            f.write(f"{ln}\n")
    return path
