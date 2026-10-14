# Errata (14 October 2026)

From a CPU audit of every script in this repo (2026-10-14). Full cross-repo errata draft: [residual-stream-grokking `papers/CORRECTIONS_2026-10-14.md`](https://github.com/t2addonio/residual-stream-grokking/blob/main/papers/CORRECTIONS_2026-10-14.md) (section E).

| # | Where | Original | Correction | Status |
|---|---|---|---|---|
| 1 | README Status; INDEX item 6; Progress paper; Unified paper | "GPT-2 Phases A–D (16/16 online write-hook)" | Can't be reproduced: `residual_causal_gpt2.phase_a_frozen_ports` and `phase_c_closed_loop` are missing; no Phase D results are committed (`aggregate_phase_d`: "no Phase D results found") | unbacked — remove or supply code/data |
| 2 | README Status; papers | "real UAH Mueller", "real Hole/NoHole 14-D", "real naval", Planck | The committed scripts run on planted synthetic data and say so ("drop-in: replace planted ensembles…"). Real data and real-data runs are not in the repo. | unbacked (real data); keep as synthetic demos |
| 3 | README Status; papers | EEG, quantum, transformer 14-D "closed" | Only `*_summary.txt` files; no generating scripts | unbacked — supply scripts |
| 4 | All PASS results | PASS | Planted linear residuals make near-zero projection/path-patch residuals (~1e-24) true by construction. They test the code, not the real-world claim. | weakened |
| 5 | `experiments/telemetry/telemetry_residual_path_patch.py` (before PR #2) | random control seeded with `abs(hash(name))` | Python string hashing is salted per process, so any random-control numbers from before 2026-10-14 aren't exactly reproducible. Fixed in PR #2. | needs rerun |
| 6 | README "Output folder" (after PR #1) | "`interferometer.py` now imports … from `packages/residual_isa_phase_e/`" | That shim imported itself; PR #2 points to `experiments/isa/`. | corrected |

Nothing above changes the code; it changes what the docs claim.
