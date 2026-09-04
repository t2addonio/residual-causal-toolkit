#!/usr/bin/env python3
"""Optical P2 — multi-mode residual subspace path-patch."""
import numpy as np
N=64; X=np.arange(N)-N/2; XX,YY=np.meshgrid(X,X)
def blob(amp,w,x0,y0,ph): return amp*np.exp(-((XX-x0)**2+(YY-y0)**2)/w**2)*np.exp(1j*ph)
def beam(ph=0): return np.exp(-(XX**2+YY**2)/10**2)*np.exp(1j*ph)
def pack(f): return np.concatenate([f.real.ravel(), f.imag.ravel()])
rng=np.random.default_rng(0)
sig=np.stack([pack(beam()+blob(0.8,4,8,-5,0.8)+blob(0.7,4,-9,7,-0.4)) for _ in range(40)])
held=np.stack([pack(beam()) for _ in range(40)])
C=sig-held.mean(0); _,S,Vt=np.linalg.svd(C, full_matrices=False); B=Vt[:2].T
def sp(x): return np.sum((x@B)**2)
def pp(x,c): return x+(c-x@B)@B.T
ch= (held@B).mean(0)
print(f"opt-P2 sig {np.mean([sp(x) for x in sig]):.1f} path {np.mean([sp(pp(x,ch)) for x in sig]):.1f} zero {np.mean([sp(pp(x,np.zeros(2))) for x in sig]):.1e} sv={S[0]:.1f},{S[1]:.1f}")
