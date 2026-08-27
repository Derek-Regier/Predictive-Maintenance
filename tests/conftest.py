"""
tests/conftest.py

Shared pytest configuration for the test suite.

This file runs automatically before any test module. Its main job is
adding every src subdirectory to sys.path so test files can import from
preprocessing.py, geometry.py, predictor.py, etc. without each file
having to repeat the same path setup boilerplate.

pytest discovers this file automatically — you never import it directly.
"""

import sys
from pathlib import Path

# Navigate from tests/ up to the project root
_ROOT = Path(__file__).resolve().parent.parent

# Register all src subdirectories that tests need to import from.
# Order matters: put more specific paths before generic ones.
_SRC_PATHS = [
    "src/data",        # preprocessing.py, feature_engineering.py
    "src/training",    # shared.py, train.py
    "src/models",      # vae.py and backbone files
    "src/health",      # geometry.py, health_monitor.py
    "src/inference",   # predictor.py
]

for subdir in _SRC_PATHS:
    path = str(_ROOT / subdir)
    if path not in sys.path:
        sys.path.insert(0, path)