#!/usr/bin/env python3
"""NV color-center temperature residual stream.
D(T) ≈ D0 + α ΔT, α ≈ −74 kHz/K. Path-patch / zero / random on common-mode D shift.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/workdir/artifacts")
RNG = np.random.default_rng(11)
D0, ALPHA, E_TYP, FWHM = 2.870, -74e-6, 0.003, 0.004
F = np.linspace(2.84, 2.90, 400)

def lorentz(f, f0, amp, w):
    return amp * (w**2) / ((f - f0)**2 + w**2)

def make_spectrum(T, E=None, noise=0.008, amp=0.25):
    if E is None:
        E = E_TYP * (0.6 + 0.8 * RNG.random())
    D = D0 + ALPHA * (T - 300.0)
    s = 1.0 - lorentz(F, D - E, amp, FWHM) - lorentz(F, D + E, amp, FWHM)
    s += noise * RNG.normal(size=len(F))
    edge = np.mean(np.r_[s[:15], s[-15:]])
    return s / edge

n = 80
T_high = 340.0 + 8 * RNG.standard_normal(n)
T_low = 260.0 + 8 * RNG.standard_normal(n)
X_high = np.array([make_spectrum(t) for t in T_high])
X_low = np.array([make_spectrum(t) for t in T_low])
X = np.vstack([X_high, X_low])
is_high = np.array([True]*n + [False]*n)
R = X - X.mean(0)
d = R[is_high].mean(0) - R[~is_high].mean(0)
d /= np.linalg.norm(d) + 1e-15
c = R @ d
idx = np.where(is_high)[0][np.argmax((c**2)[is_high])]
r = R[idx]; c0 = c[idx]; c_held = c[~is_high].mean()
r_path = c_held * d + (r - c0 * d)
r_zero = r - c0 * d
d_r = RNG.normal(size=len(d)); d_r /= np.linalg.norm(d_r)
r_rand = c_held * d_r + (r - (r @ d_r) * d_r)
print(f"NV-T orig {(r@d)**2:.5f} path {(r_path@d)**2:.5f} zero {(r_zero@d)**2:.5f} rand {(r_rand@d)**2:.5f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(F, X_high.mean(0), label="High-T"); ax[0].plot(F, X_low.mean(0), label="Low-T"); ax[0].legend()
ax[1].bar(["orig","path","zero","rand"], [(r@d)**2,(r_path@d)**2,(r_zero@d)**2,(r_rand@d)**2])
fig.suptitle("NV temperature residual path-patch")
fig.tight_layout(); fig.savefig(OUT / "nv_residual_temperature.png", dpi=120); plt.close(fig)
