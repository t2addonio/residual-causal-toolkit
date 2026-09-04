#!/usr/bin/env python3
"""NV ODMR residual toolkit core — 2-feature (D, E) path-patch.
Public data target: Zenodo 10.5281/zenodo.14697917 (Neu & Nimba).
This core plants the same (D, E) geometry so the protocol runs without the zip.
"""
import numpy as np
RNG=np.random.default_rng(7)
# Hole ensemble: large strain proxy E; NoHole: small E. Shared D carrier.
def features(E_mean, n=49):
    D = 2.870 + 0.0004*RNG.standard_normal(n)
    E = E_mean + 0.15*E_mean*RNG.standard_normal(n)
    depth = 0.22 + 0.03*RNG.standard_normal(n)
    asym = 0.02*RNG.standard_normal(n)
    return np.stack([D, E, depth, asym], 1)
hole, nohole = features(0.0948, 49), features(0.0033, 56)
d = hole.mean(0)-nohole.mean(0); d/=np.linalg.norm(d)+1e-12
print(f"NV-ODMR direction along E: {abs(d[1]):.4f} (expect ~1)")
c_held=float((nohole@d).mean())
def pp(x,c): return x+(c-x@d)*d
idx=int(np.argmax((hole@d)**2))
print(f"path { (hole[idx]@d)**2 :.4f} → {(pp(hole[idx],c_held)@d)**2:.4f} zero {(pp(hole[idx],0)@d)**2:.1e}")
print("drop-in: replace planted ensembles with Zenodo Hole vs NoHole (D,E,dip,asym)")
