"""Shared directory, logging, and JSON helpers."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid partial writes."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)
