#!/usr/bin/env python3
"""CRAI-96k — causal single-insert isolation.
96 kHz. No STFT look-ahead. No other mic. Lock-in bank + causal AMDF f0.
Throughput = host block only (32 samples @ 96 kHz = 0.333 ms).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, stft
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/workdir/artifacts")
RNG = np.random.default_rng(11)
SR, DUR, BLOCK, F0_PRIOR, N_HARM, TAU = 96000, 3.0, 32, 196.0, 8, 0.008
AMDF_WIN, AMDF_HOP = int(0.020 * SR), 64

def tone(t, f, amp=1.0, phase=0.0):
    return amp * np.sin(2 * np.pi * f * t + phase)
def plucked_carrier(t, f0=F0_PRIOR):
    env = 0.72 + 0.28 * np.sin(2 * np.pi * 1.7 * t)
    y = sum(tone(t, f0*k*np.sqrt(1+1.6e-4*(k*k-1)), a, 0.15*k)
            for k, a in enumerate([1.00,0.55,0.32,0.18,0.12,0.08], 1))
    return 0.30 * env * y
def delay_filter(x, sr, delay_s, lp_coef=0.28):
    d = int(round(delay_s * sr)); y = np.zeros_like(x)
    if 0 < d < len(x): y[d:] = x[:-d]
    acc, out = 0.0, np.zeros_like(y)
    for i, v in enumerate(y):
        acc = lp_coef * acc + (1 - lp_coef) * v; out[i] = acc
    return out
def si_sdr(est, ref):
    n = min(len(est), len(ref)); est, ref = est[:n], ref[:n]
    s = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    e = est - s
    return 10.0 * np.log10((np.dot(s, s) + 1e-12) / (np.dot(e, e) + 1e-12))
def amdf_f0(buf, sr, lo=170.0, hi=230.0):
    x = buf - np.mean(buf); n = len(x)
    dmin, dmax = max(1, int(sr/hi)), min(n-2, int(sr/lo))
    best_d, best = dmin, 1e99
    for d in range(dmin, dmax+1):
        e = np.mean(np.abs(x[d:] - x[:-d]))
        if e < best: best, best_d = e, d
    return float(sr / best_d)
def causal_f0_track(x, sr, hop=AMDF_HOP, win=AMDF_WIN, prior=F0_PRIOR):
    n = len(x); f0 = np.full(n, prior); cur = prior
    for i in range(win, n, hop):
        cur = amdf_f0(x[i-win:i], sr)
        f0[i:min(n, i+hop)] = 0.85 * f0[i-1] + 0.15 * cur
    f0[:win] = prior; return f0
def lockin_bank(x, fks, sr, tau=TAU):
    x = np.asarray(x, float); n = len(x)
    alpha = 1.0 - np.exp(-1.0 / (tau * sr))
    b, a = [alpha], [1.0, -(1.0 - alpha)]
    y = np.zeros(n); fks = np.asarray(fks, float)
    if fks.ndim == 1: fks = np.broadcast_to(fks, (n, fks.shape[0]))
    for k in range(fks.shape[1]):
        phase = 2.0 * np.pi * np.cumsum(fks[:, k]) / sr
        osc = np.exp(-1j * phase)
        env = lfilter(b, a, x * osc)
        y += 2.0 * np.real(env * np.conj(osc))
    return y

if __name__ == "__main__":
    t = np.arange(int(SR * DUR)) / SR
    guitar = plucked_carrier(t)
    # snare: short noise bursts
    snare = np.zeros_like(t)
    for t0 in np.arange(0.18, t[-1]-0.05, 0.5):
        i0 = int(t0*SR); i1 = min(len(t), i0+int(0.22*SR))
        th = np.arange(i1-i0)/SR
        burst = np.sin(2*np.pi*210*th)*np.exp(-th/0.045)
        crack = RNG.standard_normal(i1-i0)*np.exp(-th/0.022)
        snare[i0:i1] += 0.45*burst + 1.15*crack
    snare /= np.max(np.abs(snare))+1e-12
    mic_G = guitar + 0.80 * delay_filter(snare, SR, 0.0045)
    f0_G = causal_f0_track(mic_G, SR)
    ks = np.arange(1, N_HARM+1)
    fks = f0_G[:, None] * ks[None, :] * np.sqrt(1+1.6e-4*(ks*ks-1))
    g_iso = lockin_bank(mic_G, fks, SR)
    print(f"CRAI-96k throughput {1000*BLOCK/SR:.3f} ms  SI-SDR G {si_sdr(mic_G,guitar):.2f}→{si_sdr(g_iso,guitar):.2f} f0={np.median(f0_G[AMDF_WIN:]):.1f}")
