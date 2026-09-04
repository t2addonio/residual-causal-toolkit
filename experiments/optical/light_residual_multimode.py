#!/usr/bin/env python3
"""Multi-mode residual subspace — contrastive SVD recovers joint 2-mode residual."""
import numpy as np
N=64; X=np.arange(N)-N/2; XX,YY=np.meshgrid(X,X)
def blob(a,w,x0,y0,ph): return a*np.exp(-((XX-x0)**2+(YY-y0)**2)/w**2)*np.exp(1j*ph)
def pack(f): return np.concatenate([f.real.ravel(), f.imag.ravel()])
rng=np.random.default_rng(0)
sig=np.stack([pack(np.exp(-(XX**2+YY**2)/10**2)+blob(0.7,5,9,-6,1.1)+blob(0.55,5,-8,7,-0.4)) for _ in range(32)])
held=np.stack([pack(np.exp(-(XX**2+YY**2)/10**2)) for _ in range(32)])
C=sig-held.mean(0); _,S,Vt=np.linalg.svd(C,full_matrices=False)
print(f"multimode top-2 SV {S[0]:.1f} {S[1]:.1f} var {S[:2].sum()/S.sum():.1%}")
B=Vt[:2].T
def z(x): return x-(x@B)@B.T
print(f"α=1 subspace power {np.mean([np.sum((z(x)@B)**2) for x in sig]):.2e}")
