#!/usr/bin/env python3
"""NV multi-NV subspace + B-field residual cores.
Real drop-in: Zenodo 14697917 multi-center ODMR; Figshare 28788437 ensemble sweeps.
This core plants the same geometry so FILTER runs without the archives.
"""
import numpy as np
RNG=np.random.default_rng(7)

# ---- Part A: multi-NV joint residual subspace (dim-4) ----
n_nv, n_f = 40, 80
F=np.linspace(2.855, 2.890, n_f)
def odmr(D,E):
    def L(f0): return 0.22*(0.004**2)/((F-f0)**2+0.004**2)
    return 1 - L(D-E) - L(D+E) + 0.01*RNG.standard_normal(n_f)
# Hole-like: large E spread; NoHole: small E
hole=np.stack([odmr(2.870+0.0003*RNG.normal(), 0.008+0.004*RNG.random()) for _ in range(n_nv)])
nohole=np.stack([odmr(2.870+0.0003*RNG.normal(), 0.0015+0.0006*RNG.random()) for _ in range(n_nv)])
R=np.vstack([hole,nohole]); R=R-R.mean(0)
C=hole.mean(0)-nohole.mean(0)
_,S,Vt=np.linalg.svd(np.vstack([hole-nohole.mean(0)]), full_matrices=False)
B=Vt[:4].T
def sp(x): return float(np.sum((x@B)**2))
print(f"multi-NV top-4 SV {S[:4].round(2)} subspower hole {np.mean([sp(x-nohole.mean(0)) for x in hole]):.4f}")
# zero subspace
print(f"zero-subspace {np.mean([sp((x-nohole.mean(0))-(x-nohole.mean(0))@B@B.T) for x in hole]):.2e}")

# ---- Part B: B-field residual (Zeeman) ----
# two current regimes → different splitting
def sweep(B_mT):
    # simple two-line Zeeman proxy around 2.87
    split=0.0028*B_mT
    return odmr(2.870, split)
hi=np.stack([sweep(8+RNG.normal()) for _ in range(48)])
lo=np.stack([sweep(1+0.3*RNG.normal()) for _ in range(48)])
d=hi.mean(0)-lo.mean(0); d/=np.linalg.norm(d)+1e-12
c_lo=float((lo@d).mean())
def pp(x,c): return x+(c-x@d)*d
print(f"B-field path {(hi[0]@d)**2:.4f}→{(pp(hi[0],c_lo)@d)**2:.4f} zero {(pp(hi[0],0)@d)**2:.1e}")
print("drop-in: replace planted ensembles with Zenodo Hole/NoHole + Figshare current sweeps")
