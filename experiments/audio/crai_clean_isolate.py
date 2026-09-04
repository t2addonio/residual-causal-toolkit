#!/usr/bin/env python3
"""CRAI-3 — clean per-mic isolation by carrier resynthesis.
Guitar = additive partials + median envelopes (reject snare hits).
Snare  = subtract delay-aligned resynthesized guitar, keep impulse.
No neural net. No reference mic. Output rebuilt from the carrier port.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.linalg import norm
from scipy.io import wavfile
from scipy.ndimage import median_filter
from scipy.signal import istft, stft

OUT = Path("/home/workdir/artifacts")
RNG = np.random.default_rng(11)
SR, DUR, N_FFT, HOP, F0_PRIOR = 44100, 3.0, 4096, 256, 196.0

def tone(t, f, amp=1.0, phase=0.0):
    return amp * np.sin(2 * np.pi * f * t + phase)
def plucked_carrier(t, f0=F0_PRIOR):
    env = 0.72 + 0.28 * np.sin(2 * np.pi * 1.7 * t)
    y = sum(tone(t, f0*k*np.sqrt(1+1.6e-4*(k*k-1)), a, 0.15*k)
            for k, a in enumerate([1.00,0.55,0.32,0.18,0.12,0.08], 1))
    return 0.30 * env * y
def delay_filter(x, sr, delay_s, lp=0.28):
    d = int(round(delay_s * sr)); y = np.zeros_like(x)
    if 0 < d < len(x): y[d:] = x[:-d]
    acc, out = 0.0, np.zeros_like(y)
    for i, v in enumerate(y):
        acc = lp * acc + (1 - lp) * v; out[i] = acc
    return out
def spec(x):
    return stft(x, fs=SR, nperseg=N_FFT, noverlap=N_FFT-HOP, window="hann", boundary="zeros", padded=True)
def si_sdr(est, ref):
    n = min(len(est), len(ref)); est, ref = est[:n], ref[:n]
    s = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12) * ref
    e = est - s
    return 10 * np.log10((np.dot(s, s) + 1e-12) / (np.dot(e, e) + 1e-12))
def harmonic_freqs(f0, n_harm=8):
    return np.array([f0 * k * np.sqrt(1 + 1.6e-4 * (k*k - 1)) for k in range(1, n_harm+1)])
def estimate_f0(x, lo=170.0, hi=230.0):
    t = np.arange(len(x)) / float(SR)
    best_f, best = F0_PRIOR, -1.0
    for f in np.linspace(lo, hi, 121):
        tmpl = sum((0.5/k) * np.sin(2*np.pi*f*k*t) for k in range(1, 6))
        sc = abs(float(np.dot(x, tmpl)))
        if sc > best: best, best_f = sc, float(f)
    return best_f
def extract_envs(Z, freqs, fks, hw=18.0):
    env = np.zeros((len(fks), Z.shape[1]))
    for i, fk in enumerate(fks):
        m = np.abs(freqs - fk) <= hw
        env[i] = np.max(np.abs(Z[m]), axis=0) if np.any(m) else np.abs(Z[int(np.argmin(np.abs(freqs-fk)))])
    return env
def resynth(envs, fks, n, phases=None):
    scale = 2.0 / (N_FFT * 0.5)
    t_frames = np.arange(envs.shape[1]) * HOP
    t = np.arange(n) / float(SR)
    y = np.zeros(n)
    for i, (env, fk) in enumerate(zip(envs, fks)):
        amp = scale * np.interp(t * SR, t_frames, env, left=0, right=0)
        if phases is None:
            y += amp * np.sin(2 * np.pi * fk * t)
        else:
            a, b = phases[i]; nrm = (a*a + b*b)**0.5 + 1e-12
            y += amp * ((a/nrm)*np.sin(2*np.pi*fk*t) + (b/nrm)*np.cos(2*np.pi*fk*t))
    return y
def fit_phase(x, fk, mask):
    t = np.arange(len(x)) / float(SR)
    A = np.stack([np.sin(2*np.pi*fk*t)[mask], np.cos(2*np.pi*fk*t)[mask]], 1)
    coef, *_ = np.linalg.lstsq(A, x[mask], rcond=None)
    return float(coef[0]), float(coef[1])
def gcc_phat_delay(x, y, max_lag_s=0.025):
    n = int(2 ** np.ceil(np.log2(len(x)+len(y))))
    R = np.fft.rfft(x, n=n) * np.conj(np.fft.rfft(y, n=n))
    R /= np.abs(R) + 1e-12
    r = np.fft.irfft(R, n=n); r = np.concatenate([r[-n//2:], r[:n//2]])
    lags = np.arange(-n//2, n-n//2) / float(SR)
    w = np.abs(lags) <= max_lag_s
    return float(lags[w][int(np.argmax(r[w]))])

if __name__ == "__main__":
    t = np.arange(int(SR*DUR)) / SR
    guitar = plucked_carrier(t)
    snare = np.zeros_like(t)
    for t0 in np.arange(0.18, t[-1]-0.05, 0.5):
        i0 = int(t0*SR); i1 = min(len(t), i0+int(0.22*SR)); th = np.arange(i1-i0)/SR
        snare[i0:i1] += np.sin(2*np.pi*210*th)*np.exp(-th/0.045) + RNG.standard_normal(i1-i0)*np.exp(-th/0.022)
    snare /= np.max(np.abs(snare))+1e-12
    mic_G = guitar + 0.80 * delay_filter(snare, SR, 0.0045)
    mic_S = snare + 0.55 * delay_filter(guitar, SR, 0.0062)
    f, tt, ZG = spec(mic_G)
    f0 = estimate_f0(mic_G)
    fks = harmonic_freqs(f0)
    env = median_filter(extract_envs(ZG, f, fks), size=(1, 17))
    # non-hit frames for phase fit: low high-band energy
    hb = np.sqrt(np.mean(np.abs(ZG[f>2500])**2, axis=0)+1e-12)
    mask = np.repeat(hb < np.median(hb), HOP)[:len(mic_G)]
    phases = [fit_phase(mic_G, fk, mask) for fk in fks]
    g_iso = resynth(env, fks, len(mic_G), phases=phases)
    delay = gcc_phat_delay(g_iso, mic_S)
    d = int(round(abs(delay)*SR))
    aligned = np.zeros_like(mic_S)
    if 0 < d < len(mic_S): aligned[d:] = g_iso[:-d]
    else: aligned = g_iso
    scale = float(np.dot(mic_S, aligned) / (np.dot(aligned, aligned)+1e-12))
    s_iso = mic_S - scale * aligned
    print(f"CRAI-3 f0={f0:.2f} delay={1000*abs(delay):.2f}ms  SI-SDR G {si_sdr(mic_G,guitar):.2f}→{si_sdr(g_iso,guitar):.2f}  S {si_sdr(mic_S,snare):.2f}→{si_sdr(s_iso,snare):.2f}")
    print("bar: guitar SI-SDR≥20, snare≥12 on the offline synthetic scene (CRAI-3 lock)")
