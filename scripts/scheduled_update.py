"""CLI wrapper for scheduled feedback-based model updates."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.update_scheduler import main


if __name__ == "__main__":
    main()
