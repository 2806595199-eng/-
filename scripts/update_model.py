"""CLI wrapper for the model update loop."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.model_update import main


if __name__ == "__main__":
    main()
