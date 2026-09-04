#!/usr/bin/env python3
"""Telemetry T-P1 / T-P2 — NMEA + 4× vibration residual (3.3× order harmonic)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)
FS, T, N_REAL = 200.0, 4.0, 80
N = int(FS * T)
t = np.arange(N) / FS

def features(residual_scale=1.0):
    rpm = 2400 + 12*np.sin(2*np.pi*0.15*t)
    f_rot = float(np.mean(rpm)) / 60.0
    vib = np.zeros((4, N))
    for i, a1 in enumerate([0.055, 0.022, 0.030, 0.012]):
        vib[i] = a1*np.sin(2*np.pi*f_rot*t + 0.3*i) + 0.5*a1*np.sin(2*np.pi*2*f_rot*t)
        vib[i] += 0.004*np.random.randn(N)
    res = residual_scale * 0.042 * np.sin(2*np.pi*3.25*f_rot*t)
    vib[0] += res; vib[2] += 0.70*res
    X = np.fft.rfft(vib, axis=1); freqs = np.fft.rfftfreq(N, 1/FS)
    k = int(np.argmin(np.abs(freqs - 3.25*f_rot)))
    res_band = np.mean(np.abs(X[:, max(0,k-4):k+5])**2, axis=1)
    return np.concatenate([res_band, np.sqrt(np.mean(vib**2, axis=1))])

sig = np.stack([features(1.0+0.1*np.random.randn()) for _ in range(N_REAL)])
held = np.stack([features(0.08+0.03*np.random.randn()) for _ in range(N_REAL)])
d = sig.mean(0) - held.mean(0); d /= np.linalg.norm(d)+1e-12
c_held = float((held @ d).mean())
def pp(x, cnew): return x + (cnew - x@d)*d
pow_ = lambda X: np.mean((X @ d)**2)
print(f"T-P1 power sig {pow_(sig):.4f} path {np.mean([(pp(x,c_held)@d)**2 for x in sig]):.4f} zero {np.mean([(pp(x,0)@d)**2 for x in sig]):.2e}")
