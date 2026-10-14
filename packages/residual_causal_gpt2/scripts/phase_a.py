#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from residual_causal_gpt2.phase_a_frozen_ports import main
except ModuleNotFoundError as e:
    if e.name != "residual_causal_gpt2.phase_a_frozen_ports":
        raise
    sys.exit(
        "phase_a.py: residual_causal_gpt2/phase_a_frozen_ports.py is not in this repository.\n"
        "The Phase A implementation was never committed, so this entry point cannot run.\n"
        "See packages/residual_causal_gpt2/MISSING.md."
    )
if __name__ == "__main__":
    main()
