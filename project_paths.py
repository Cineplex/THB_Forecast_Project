"""
Utility module that ensures the repository root is available on sys.path.
Import this module before importing other local packages when executing
modules directly (e.g., `python apis\\extract.py`).
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

__all__ = ["ROOT_DIR"]

