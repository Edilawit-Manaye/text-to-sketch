"""Compatibility wrapper for ``metrics.evaluate_encoder``."""

from __future__ import annotations

import sys
from pathlib import Path


def _add_project_to_path() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pipeline").exists() and (parent / "prep_data").exists():
            sys.path.insert(0, str(parent))
            return
    raise RuntimeError("Could not find project root directory.")


_add_project_to_path()

from metrics.evaluate_encoder import main


if __name__ == "__main__":
    main()
