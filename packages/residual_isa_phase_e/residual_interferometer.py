#!/usr/bin/env python3
"""Re-export of experiments/isa/residual_interferometer.py for Phase E package."""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "experiments" / "isa"))
from residual_interferometer import *  # noqa: F401,F403
