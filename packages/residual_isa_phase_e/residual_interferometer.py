#!/usr/bin/env python3
"""Re-export of experiments/isa/residual_interferometer.py for the Phase E package.

Loaded by file path under a private module name so this shim never imports itself.
"""
import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "experiments" / "isa" / "residual_interferometer.py"
_spec = importlib.util.spec_from_file_location("_rct_residual_interferometer", _SRC)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod  # needed for @dataclass
_spec.loader.exec_module(_mod)

__all__ = [n for n in dir(_mod) if not n.startswith("_")]
globals().update({n: getattr(_mod, n) for n in __all__})
