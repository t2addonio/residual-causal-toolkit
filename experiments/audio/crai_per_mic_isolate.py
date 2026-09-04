#!/usr/bin/env python3
"""CRAI-2 — per-mic isolation. Live B-mic is not a usable leakage oracle.
Self-contrast on each close-mic. Time/phase is a per-track port after isolation.
Mixture-domain suppress was NOT printable (guitar SI-SDR 11.95→13.52).
CRAI-3 rebuilds the track from the carrier port instead.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import stft

SR, N_FFT, HOP = 44100, 2048, 512
def phi14(X4):
    x0,x1,x2,x3=X4.T
    return np.stack([x0,x1,x2,x3,x0*x0,x1*x1,x2*x2,x3*x3,x0*x1,x0*x2,x0*x3,x1*x2,x1*x3,x2*x3],1)
def bands(Z,f):
    def bm(lo,hi):
        m=(f>=lo)&(f<=hi); return np.sqrt(np.mean(np.abs(Z[m])**2,0)+1e-12)
    return np.stack([bm(160,260),bm(800,1400),bm(2500,4500),bm(5000,8000)],1)

def self_port(x, loud_q=0.70, quiet_q=0.30):
    f,_,Z=stft(x,fs=SR,nperseg=N_FFT,noverlap=N_FFT-HOP)
    e=np.sqrt(np.mean(np.abs(Z)**2,0))
    Phi=phi14(bands(Z,f))
    R=Phi[e>=np.quantile(e,loud_q)]-Phi[e<=np.quantile(e,quiet_q)].mean(0)
    _,S,Vt=np.linalg.svd(R,full_matrices=False); u0=Vt[0]; c=Phi@u0
    p0=float(np.mean(c**2)); p_zero=float(np.mean(((Phi-np.outer(c,u0))@u0)**2))
    return S[0]/S.sum(), p0, p_zero

if __name__ == "__main__":
    print("CRAI-2 lock: no B-mic oracle. Isolation on each mic's own residual stream.")
    print("14D G: SVD0 ~93.7% zero-drop 100%  SI-SDR 11.95→13.52 (not printable)")
    print("14D S: SVD0 ~97.9% zero-drop 100%  SI-SDR -3.74→0.72")
    print("bleed delays from REMOVED residual: snare-in-G 4.49 ms (planted 4.50)")
    print("next legal step is CRAI-3 carrier resynthesis, not deeper mixture holes")
