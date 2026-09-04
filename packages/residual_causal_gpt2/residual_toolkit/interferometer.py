"""QPU-like unitary 2-D residual mix.

Imports the project-level residual_interferometer module. If that
module is not on PYTHONPATH (running inside residual_causal_gpt2/),
the artifacts directory is added.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ART = Path("/home/workdir/artifacts")
if str(_ART) not in sys.path:
    sys.path.insert(0, str(_ART))

from residual_interferometer import (  # noqa: E402,F401
    extract_2d_core,
    interference_readout,
    orthonormalize_pair,
    plane_power,
    subtractive_mix,
    sweep_phase,
    total_power,
    unitary_interferometer,
)
