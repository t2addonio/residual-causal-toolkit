#!/usr/bin/env python3
"""RF-P2 — joint residual subspace path-patch (two residual tones)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)
FS, N, N_REAL = 1.0, 2048, 96
f_c1, f_c2, f_i1, f_i2, f_i3 = 0.08, 0.15, 0.22, 0.31, 0.38
f_res1, f_res2 = 0.45, 0.48

def make_tone(f, amp, phase, n=N):
    t = np.arange(n) / FS
    return amp * np.exp(1j * (2 * np.pi * f * t + phase))

def apply_cubic_pa(x, a1=1.0, a3=0.12):
    return a1 * x + a3 * x * np.abs(x)**2

def generate_mixture(r1=1.0, r2=0.7, noise=0.012):
    linear = (make_tone(f_c1,1.0,np.random.uniform(0,2*np.pi)) +
              make_tone(f_c2,0.85,np.random.uniform(0,2*np.pi)) +
              make_tone(f_i1,0.35,np.random.uniform(0,2*np.pi)) +
              make_tone(f_i2,0.28,np.random.uniform(0,2*np.pi)) +
              make_tone(f_i3,0.22,np.random.uniform(0,2*np.pi)))
    res = r1*make_tone(f_res1,0.38,np.random.uniform(0,2*np.pi)) + r2*make_tone(f_res2,0.30,np.random.uniform(0,2*np.pi))
    return apply_cubic_pa(linear) + res + noise*(np.random.randn(N)+1j*np.random.randn(N))

def features(x):
    X = np.fft.fft(x)
    k1, k2 = int(f_res1*N), int(f_res2*N)
    cb = np.concatenate([X[k1-4:k1+5].real, X[k1-4:k1+5].imag, X[k2-4:k2+5].real, X[k2-4:k2+5].imag])
    return np.concatenate([np.log1p(np.abs(X)[:N//2]), cb])

sig_f = np.stack([features(generate_mixture(1.0+0.1*np.random.randn(), 0.7+0.08*np.random.randn())) for _ in range(N_REAL)])
held_f = np.stack([features(generate_mixture(0.08+0.03*np.random.randn(), 0.06+0.03*np.random.randn())) for _ in range(N_REAL)])
C = sig_f - held_f.mean(0)
_, S, Vt = np.linalg.svd(C, full_matrices=False)
D_sub = Vt[:2].T
C_held_mean = (held_f @ D_sub).mean(0)

def subspace_power(x, B): return np.sum((x @ B)**2)
def path_patch_sub(x, B, c_new): return x + (c_new - x @ B) @ B.T

rng = np.random.RandomState(13)
Rrand, _ = np.linalg.qr(rng.randn(sig_f.shape[1], 2))
orig, path, zero, randp = [], [], [], []
for x in sig_f:
    orig.append(subspace_power(x, D_sub))
    path.append(subspace_power(path_patch_sub(x, D_sub, C_held_mean), D_sub))
    zero.append(subspace_power(path_patch_sub(x, D_sub, np.zeros(2)), D_sub))
    xr = x + (C_held_mean - x @ Rrand) @ Rrand.T
    randp.append(subspace_power(xr, D_sub))
print(f"RF-P2 signal {np.mean(orig):.1f} path {np.mean(path):.1f} zero {np.mean(zero):.1e} rand {np.mean(randp):.1f} sv={S[0]:.1f},{S[1]:.1f}")
fig, ax = plt.subplots(figsize=(7,4))
ax.bar(["signal","path","zero","random"], [np.mean(orig), np.mean(path), max(np.mean(zero),1e-16), np.mean(randp)])
ax.set_yscale("log"); ax.set_title("RF-P2 subspace path-patch")
fig.tight_layout(); fig.savefig(Path("/home/workdir/artifacts/rf_residual_path_patch_p2.png"), dpi=120); plt.close(fig)
