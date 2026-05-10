"""Interactive CLI for the main text-to-sketch pipeline."""

from __future__ import annotations

import sys

from utils.paths import (
    DEFAULT_FILTERED_SKETCHES_DIR,
    DEFAULT_STROKE5_DIR,
    DEFAULT_SKETCH_TOKEN_DIR,
)
from pipeline.workflow import run_pipeline

_BANNER = """
╔════════════════════════════════════════════════════════════╗
║          Hand Simulation Pipeline — Text-to-Sketch         ║
║ Stages: Vectorize → Order → Kinematics → Stroke5 → Tok-Dict║
╚════════════════════════════════════════════════════════════╝"""

_ORDERING_METHODS: dict[str, str] = {
    "1": "directional",
    "2": "greedy",
    "3": "tsp",
}
_DEFAULT_ORDERING = "1"


def _prompt_int(prompt: str, default: int, lo: int = 1, hi: int = 10_000) -> int:
    """Prompt for an integer, returning *default* on empty/invalid input."""
    while True:
        try:
            raw = input(prompt).strip()
            if not raw:
                return default
            value = int(raw)
            if lo <= value <= hi:
                return value
            print(f"  Please enter a number between {lo} and {hi}.")
        except (ValueError, EOFError):
            return default


def _prompt_ordering() -> str:
    """Prompt the user to select a stroke-ordering method."""
    print("\nStroke-ordering method:")
    print("  1) Directional bias [default]  — top-left → bottom-right")
    print("  2) Greedy nearest-neighbor     — minimise pen travel locally")
    print("  3) TSP approximation           — globally minimise pen travel")
    while True:
        try:
            raw = input("Choose [1/2/3, default: 1]: ").strip() or _DEFAULT_ORDERING
        except EOFError:
            raw = _DEFAULT_ORDERING
        if raw in _ORDERING_METHODS:
            return _ORDERING_METHODS[raw]
        print("  Invalid choice — please enter 1, 2, or 3.")


def main() -> None:
    sketches_dir = DEFAULT_FILTERED_SKETCHES_DIR
    stroke5_dir = DEFAULT_STROKE5_DIR
    sketch_token_dir = DEFAULT_SKETCH_TOKEN_DIR

    available = list(sketches_dir.rglob("*.png"))

    print(_BANNER)
    print(f"\n  Available sketches : {len(available)}")

    if not available:
        print(f"\n[error] No sketches found in {sketches_dir}")
        print("        Run 'python scripts/prepare_data/filter_sketches_by_points.py' first.")
        sys.exit(1)

    default_n = min(50, len(available))
    n = _prompt_int(
        f"\nHow many sketches to process? [default: {default_n}]: ",
        default=default_n,
        lo=1,
        hi=len(available),
    )
    ordering = _prompt_ordering()

    print(f"\n  Will process {n} sketches with '{ordering}' ordering.")

    run_pipeline(
        sketches_dir=sketches_dir,
        stroke5_dir=stroke5_dir,
        sketch_token_dir=sketch_token_dir,
        n_sketches=n,
        ordering=ordering,
    )

    print(f"\n{'═' * 58}")
    print("  Pipeline complete!")
    print(f"{'═' * 58}\n")


if __name__ == "__main__":
    main()
