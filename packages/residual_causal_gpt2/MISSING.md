# Known gaps in this package

- `scripts/phase_a.py` imports `residual_causal_gpt2.phase_a_frozen_ports` and
  `scripts/phase_c.py` imports `residual_causal_gpt2.phase_c_closed_loop`.
  **Neither module is committed to this repository**, so both entry points exit
  with an explanatory error. The Phase A–D "16/16" claims cannot be reproduced
  from this repo until those modules (and Phase D result files) are added.
- Modules inside `residual_causal_gpt2/` use relative imports. Run them as modules
  from this folder, e.g. `python -m residual_causal_gpt2.residual_extract` or
  `python -m residual_causal_gpt2.train`, not as `python residual_causal_gpt2/train.py`.
- `residual_toolkit/interferometer.py` loads the implementation from
  `experiments/isa/residual_interferometer.py` at the repo root.
