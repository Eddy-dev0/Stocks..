"""Documentation helpers bundled with the stock predictor package."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = PACKAGE_ROOT / "docs"

__all__ = ["DOCS_PATH"]
