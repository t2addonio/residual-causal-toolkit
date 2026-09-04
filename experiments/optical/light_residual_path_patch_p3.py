#!/usr/bin/env python3
"""Optical P3 — Jones / polarization residual coefficient path-patch.
Carrier = horizontal; residual = vertical (Ey). H-pol intensity stays intact.
"""
import numpy as np
N=48; X=np.arange(N)-N/2; XX,YY=np.meshgrid(X,X); R2=XX**2+YY**2
def Ex(ph=0): return np.exp(-R2/10**2)*np.exp(1j*ph)   # H carrier
def Ey(amp=0.45, ph=0.9, x0=8, y0=-5):
    return amp*np.exp(-((XX-x0)**2+(YY-y0)**2)/4.2**2)*np.exp(1j*ph)
def pack(ex,ey): return np.concatenate([ex.real.ravel(),ex.imag.ravel(),ey.real.ravel(),ey.imag.ravel()])
rng=np.random.default_rng(2)
sig=np.stack([pack(Ex(), Ey()) for _ in range(32)])
held=np.stack([pack(Ex(), np.zeros_like(Ex())) for _ in range(32)])
d=sig.mean(0)-held.mean(0); d/=np.linalg.norm(d)+1e-12
c_held=float((held@d).mean())
def pp(x,c): return x+(c-x@d)*d
pow=lambda X: np.mean((X@d)**2)
print(f"opt-P3 power {pow(sig):.2f} path {np.mean([(pp(x,c_held)@d)**2 for x in sig]):.2f} zero {np.mean([(pp(x,0)@d)**2 for x in sig]):.1e}")
print("H-pol carrier protected; residual ~ Ey")
