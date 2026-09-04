#!/usr/bin/env python3
"""RF Residual Causal Toolkit — Path-Patch RF-P1.
Narrowband multi-feature IQ + optical-style path-patch protocol.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

np.random.seed(42)
FS, N, N_REAL = 1.0, 2048, 80
f_c1, f_c2 = 0.08, 0.15
f_i1, f_i2, f_i3 = 0.22, 0.31, 0.38
f_res = 0.45

def make_tone(f, amp, phase, n=N):
    t = np.arange(n) / FS
    return amp * np.exp(1j * (2 * np.pi * f * t + phase))

def apply_cubic_pa(x, a1=1.0, a3=0.12):
    return a1 * x + a3 * x * np.abs(x)**2

def generate_mixture(residual_scale=1.0, residual_phase=0.0, noise_std=0.012):
    linear = (make_tone(f_c1, 1.00, np.random.uniform(0, 2*np.pi))
              + make_tone(f_c2, 0.85, np.random.uniform(0, 2*np.pi))
              + make_tone(f_i1, 0.35, np.random.uniform(0, 2*np.pi))
              + make_tone(f_i2, 0.28, np.random.uniform(0, 2*np.pi))
              + make_tone(f_i3, 0.22, np.random.uniform(0, 2*np.pi)))
    residual = residual_scale * make_tone(f_res, 0.40, residual_phase)
    y = apply_cubic_pa(linear) + residual + noise_std * (np.random.randn(N) + 1j * np.random.randn(N))
    return y, residual

def features(x):
    X = np.fft.fft(x)
    mag = np.abs(X)[:N//2]
    k0 = int(f_res * N)
    complex_bins = np.concatenate([X[k0-5:k0+6].real, X[k0-5:k0+6].imag])
    return np.concatenate([np.log1p(mag), complex_bins]), X

def residual_bin_energy(X, bw=0.012):
    k = int(f_res * N)
    dk = max(1, int(bw * N))
    return np.mean(np.abs(X[k-dk:k+dk+1])**2)

sig_feats, held_feats, sig_X, held_X = [], [], [], []
for i in range(N_REAL):
    y, _ = generate_mixture(residual_scale=1.0 + 0.12*np.random.randn(), residual_phase=np.random.uniform(0, 2*np.pi))
    f, X = features(y); sig_feats.append(f); sig_X.append(X)
    y_h, _ = generate_mixture(residual_scale=0.10 + 0.04*np.random.randn(), residual_phase=np.random.uniform(0, 2*np.pi))
    fh, Xh = features(y_h); held_feats.append(fh); held_X.append(Xh)
sig_feats, held_feats = np.stack(sig_feats), np.stack(held_feats)
D = sig_feats.shape[1]
d = sig_feats.mean(0) - held_feats.mean(0)
d /= (np.linalg.norm(d) + 1e-12)
c_sig, c_held = sig_feats @ d, held_feats @ d
c_held_mean = float(c_held.mean())

def path_patch(x, d, c_new):
    return x + (c_new - np.dot(x, d)) * d

rng = np.random.RandomState(11)
d_rand = rng.randn(D); d_rand /= np.linalg.norm(d_rand)
res_power = lambda x: (np.dot(x, d))**2
orig_p, pp_held_p, pp_zero_p, pp_rand_p = [], [], [], []
for i in range(N_REAL):
    x = sig_feats[i]
    orig_p.append(res_power(x))
    pp_held_p.append(res_power(path_patch(x, d, c_held_mean)))
    pp_zero_p.append(res_power(path_patch(x, d, 0.0)))
    c_r = np.dot(x, d_rand)
    pp_rand_p.append(res_power(x + (c_held_mean - c_r) * d_rand))
orig_p, pp_held_p = np.array(orig_p), np.array(pp_held_p)
pp_zero_p, pp_rand_p = np.array(pp_zero_p), np.array(pp_rand_p)
print(f"RF-P1 power signal {orig_p.mean():.2f} path {pp_held_p.mean():.2f} zero {pp_zero_p.mean():.2e} rand {pp_rand_p.mean():.2f}")

out_png = Path("/home/workdir/artifacts/rf_residual_path_patch_p1.png")
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)
ax.bar(["signal", "path-held", "zero", "random"],
       [orig_p.mean(), pp_held_p.mean(), max(pp_zero_p.mean(), 1e-16), pp_rand_p.mean()])
ax.set_yscale("log"); ax.set_title("RF-P1 path-patch residual power")
fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
print(f"Saved {out_png}")
