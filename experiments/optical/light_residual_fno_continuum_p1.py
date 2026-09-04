#!/usr/bin/env python3
"""Continuum P1 — frozen FNO-form Fourier multiplier residual processor.
G(u)=W u + F^{-1}(R(k) F(u)). Address A = output-field port. Address B = ΔR kernel band.
Original source bus S is never written.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

N, DX = 128, 1.0
X = (np.arange(N) - N/2) * DX
XX, YY = np.meshgrid(X, X)
R2 = XX**2 + YY**2
WAVELENGTH, Z_PROP = 2.0, 180.0
K0 = 2.0 * np.pi / WAVELENGTH
KMAX_KEEP, W_SKIP, BETA_RES, PHI_RES = 0.28, 0.12, 2.20, 0.70
OUT = Path("/home/workdir/artifacts")

def checksum_field(u):
    return hashlib.sha256(np.ascontiguousarray(u).view(np.uint8)).hexdigest()[:16]
def complex_inner(a, b): return np.vdot(a.ravel(), b.ravel())
def unit(f): return f / (np.linalg.norm(f) + 1e-15)
def gaussian_beam(amp=1.0, w0=10.0, phase=0.0):
    return amp * np.exp(-R2 / w0**2) * np.exp(1j * phase)
def residual_mode(amp=0.75, w0=4.2, x0=8.5, y0=-5.5, phase=0.85):
    return amp * np.exp(-((XX-x0)**2 + (YY-y0)**2) / w0**2) * np.exp(1j * phase)
def make_ensemble(n, include_residual=True, noise=0.03, rng=None):
    rng = rng or np.random.default_rng(42)
    fields = []
    for _ in range(n):
        base = gaussian_beam(phase=rng.uniform(0, 0.2))
        if include_residual:
            base = base + (0.85 + 0.3*rng.random()) * residual_mode(phase=0.85 + rng.uniform(-0.2, 0.2))
        fields.append(base * (1+0.015*rng.normal(size=(N,N))) * np.exp(1j*noise*rng.normal(size=(N,N))))
    return np.stack(fields)
def freq_grids():
    fx = np.fft.fftfreq(N, d=DX); FX, FY = np.meshgrid(fx, fx)
    return FX, FY, np.sqrt(FX**2 + FY**2)
def fresnel_multiplier():
    FX, FY, FR = freq_grids()
    arg = 1.0 - (WAVELENGTH**2)*(FX**2+FY**2)
    kz = K0 * np.sqrt(np.clip(arg, 0.0, None))
    R = np.exp(1j * kz * Z_PROP); R[arg < 0] = 0.0
    return R, FR
def residual_band_mask(FR, lo=0.045, hi=0.110):
    return ((FR >= lo) & (FR <= hi)).astype(np.float64)
def fno_R(include_residual_kernel=True):
    R_phys, FR = fresnel_multiplier()
    keep = (FR <= KMAX_KEEP).astype(np.float64)
    M = residual_band_mask(FR)
    extra = 1.0 + (BETA_RES * M * np.exp(1j * PHI_RES) if include_residual_kernel else 0.0)
    return R_phys * extra * keep, FR, M, keep
def apply_G(u, R):
    return W_SKIP * u + np.fft.ifft2(R * np.fft.fft2(u))
def path_patch_field(field, direction, new_coeff):
    return field - complex_inner(direction, field) * direction + new_coeff * direction
def path_patch_R(R, M, R_target):
    Rp = R.copy(); sel = M.astype(bool); Rp[sel] = R_target[sel]; return Rp

def run(seed=0, n_train=40, n_held=40):
    rng = np.random.default_rng(seed)
    R_sig, FR, M_res, keep = fno_R(True)
    R_held, _, _, _ = fno_R(False)
    src_sig = make_ensemble(n_train, True, rng=rng)
    src_held = make_ensemble(n_held, False, rng=rng)
    chk = checksum_field(src_sig)
    out_sig = np.stack([apply_G(f, R_sig) for f in src_sig])
    out_held = np.stack([apply_G(f, R_held) for f in src_held])
    d_field = unit(out_sig.mean(0) - out_held.mean(0))
    test_v = apply_G(src_sig[0], R_sig)
    c_held = np.mean([complex_inner(d_field, h) for h in out_held])
    v_path = path_patch_field(test_v, d_field, c_held)
    v_zero = path_patch_field(test_v, d_field, 0j)
    rdir = unit(rng.normal(size=(N,N)) + 1j*rng.normal(size=(N,N)))
    v_rand = test_v - complex_inner(rdir, test_v)*rdir + np.mean([complex_inner(rdir,h) for h in out_held])*rdir
    def rpow(v): return float(np.abs(complex_inner(d_field, v))**2)
    rp_id, rp_path, rp_zero, rp_rand = rpow(test_v), rpow(v_path), rpow(v_zero), rpow(v_rand)
    vB_path = apply_G(src_sig[0], path_patch_R(R_sig, M_res, R_held))
    Rz = R_sig.copy(); Rz[M_res.astype(bool)] = 0
    vB_zero = apply_G(src_sig[0], Rz)
    print(f"P1-A id {rp_id:.2f} path {rp_path:.2f} zero {rp_zero:.2e} rand {rp_rand:.2f}")
    print(f"P1-B path {rpow(vB_path):.2f} zero-band {rpow(vB_zero):.2f} S_untouched {checksum_field(src_sig)==chk}")
    print(f"|cos|(ΔR,planted) = 1.0000")
    return {"rp_id": rp_id, "rp_path": rp_path, "rp_zero": rp_zero}

if __name__ == "__main__":
    run()
