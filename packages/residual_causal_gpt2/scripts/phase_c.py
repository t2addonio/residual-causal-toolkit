#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from residual_causal_gpt2.phase_c_closed_loop import main
except ModuleNotFoundError as e:
    if e.name != "residual_causal_gpt2.phase_c_closed_loop":
        raise
    sys.exit(
        "phase_c.py: residual_causal_gpt2/phase_c_closed_loop.py is not in this repository.\n"
        "The Phase C implementation was never committed, so this entry point cannot run.\n"
        "See packages/residual_causal_gpt2/MISSING.md."
    )
if __name__ == "__main__":
    main()
