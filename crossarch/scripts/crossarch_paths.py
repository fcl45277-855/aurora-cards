"""Locations used by the cross-architecture scripts.

CROSSARCH_ROOT  writable work directory; the scripts create sources/, weights/, data/, finetune_era5_20261020/,
                configs/, climax_finetune_run/, results/ and metadata/ below it (default: <repo>/crossarch_work).
AQ_ERA5_DIR     directory with the Aurora ERA5 inputs (surface_<date>.npz, air_<date>.npz; see docs/DATA.md),
                used to build the 2024 test inputs (default: <repo>/data/era5).

Importing this module also puts the repository root on sys.path, so that cards/ (the NVFP4 quantiser) is importable.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('CROSSARCH_ROOT', REPO / 'crossarch_work')).resolve()
AQ_ERA5_DIR = Path(os.environ.get('AQ_ERA5_DIR', REPO / 'data' / 'era5'))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
