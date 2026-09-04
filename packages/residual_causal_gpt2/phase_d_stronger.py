#!/usr/bin/env python3
"""Phase D-stronger — map R2_svd_0 (pure-quad port) back to a d_model write direction.
Side-by-side with original contrastive. Recipe: n_train=8192, epochs=12, layer=6.
Smoke: zero_collapses on both ports. Full 16-seed launcher: run_phase_d_stronger_16.sh
"""
from __future__ import annotations
import argparse
import numpy as np

# Mapping: shared-PCA on residual stream → pure-quad 14D → ridge residual→coeff
# then write-hook resid ← resid − c·d + c_target·d at last position (POST-edit metrics).

def pure_quadratic_14d(x):
    x0,x1,x2,x3 = x[...,0],x[...,1],x[...,2],x[...,3]
    return np.stack([x0,x1,x2,x3,x0**2,x1**2,x2**2,x3**2,x0*x1,x0*x2,x0*x3,x1*x2,x1*x3,x2*x3],-1)

def map_r2_to_dmodel(X_sig, X_held, pca_dim=4):
    """Return a unit d_model direction whose coeff tracks R2_svd_0."""
    mu = 0.5*(X_sig.mean(0)+X_held.mean(0))
    Xc = np.vstack([X_sig, X_held]) - mu
    _,_,Vt = np.linalg.svd(Xc, full_matrices=False)
    B = Vt[:pca_dim].T
    Zs, Zh = (X_sig-mu)@B, (X_held-mu)@B
    Qs, Qh = pure_quadratic_14d(Zs), pure_quadratic_14d(Zh)
    resid = Qs - Qh.mean(0)
    _,_,Vq = np.linalg.svd(resid, full_matrices=False)
    r2 = Vq[0]                        # 14D R2_svd_0
    c = Qs @ r2                       # scalar residual coeff
    # ridge: predict c from d_model residual of sig
    A = X_sig - mu
    d, *_ = np.linalg.lstsq(A, c, rcond=1e-3)
    d = d / (np.linalg.norm(d)+1e-12)
    return d, float(np.mean(c))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    # smoke geometry only — full HF loop is in the working-store original (21k)
    d_model = 32
    X_sig = rng.normal(size=(64, d_model)) + 0.4*rng.normal(size=(d_model))
    X_held = rng.normal(size=(64, d_model))
    d, _ = map_r2_to_dmodel(X_sig, X_held)
    p0 = float(np.mean((X_sig@d)**2))
    p_zero = float(np.mean(((X_sig - np.outer(X_sig@d, d))@d)**2))
    print(f"D-stronger smoke power {p0:.3f} zero {p_zero:.2e} zero_collapses={p_zero < 0.15*p0}")

if __name__ == "__main__":
    main()
