#!/usr/bin/env python3
"""light_residual_phase_cancel.py — Pure optical residual-stream toolkit.
Contrastive train-vs-held-out extraction then projection / phase-cancellation.
Pure classical optics. No color-center content.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

N = 128
X = (np.arange(N) - N / 2) * 1.0
Y = (np.arange(N) - N / 2) * 1.0
XX, YY = np.meshgrid(X, Y)
R2 = XX**2 + YY**2

def complex_inner(a, b):
    return np.vdot(a.ravel(), b.ravel())

def project(field, direction):
    return complex_inner(direction, field) * direction

def ablate(field, direction, alpha=1.0):
    return field - alpha * project(field, direction)

def random_direction_like(direction, rng):
    rand = rng.normal(size=direction.shape) + 1j * rng.normal(size=direction.shape)
    return rand / np.linalg.norm(rand)

def gaussian_beam(amp=1.0, w0=14.0, phase=0.0):
    return amp * np.exp(-R2 / w0**2) * np.exp(1j * phase)

def residual_mode(amp=0.45, w0=7.0, x0=9.0, y0=-6.0, phase=1.1):
    rr2 = (XX - x0)**2 + (YY - y0)**2
    return amp * np.exp(-rr2 / w0**2) * np.exp(1j * phase)

def make_ensemble(n_samples, include_residual=True, noise_level=0.035, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    fields = []
    for _ in range(n_samples):
        base = gaussian_beam(phase=rng.uniform(0, 0.25))
        if include_residual:
            base = base + residual_mode()
        phase_noise = noise_level * rng.normal(size=(N, N))
        amp_noise = 1.0 + 0.015 * rng.normal(size=(N, N))
        fields.append(base * amp_noise * np.exp(1j * phase_noise))
    return np.stack(fields, axis=0)

def contrastive_direction(signal_fields, heldout_fields):
    residual = np.mean(signal_fields, axis=0) - np.mean(heldout_fields, axis=0)
    norm = np.linalg.norm(residual)
    if norm < 1e-12:
        raise RuntimeError("Contrastive residual vanished")
    return residual / norm, residual

def residual_power(field, direction):
    return np.abs(complex_inner(direction, field))**2

def residual_location_intensity(field, x0=9.0, y0=-6.0, radius=4.0):
    mask = ((XX - x0)**2 + (YY - y0)**2) < radius**2
    return np.mean(np.abs(field[mask])**2)

def run_experiment(seed=0, n_train=80, n_held=80, alphas=None):
    rng = np.random.default_rng(seed)
    if alphas is None:
        alphas = np.linspace(0.0, 2.5, 26)
    signal = make_ensemble(n_train, include_residual=True, rng=rng)
    heldout = make_ensemble(n_held, include_residual=False, rng=rng)
    direction, raw_residual = contrastive_direction(signal, heldout)
    test = signal[0].copy()
    base_res_pow = residual_power(test, direction)
    base_loc = residual_location_intensity(test)
    res_pows, loc_intens, ctrl_res_pows, ctrl_loc = [], [], [], []
    for a in alphas:
        ablated = ablate(test, direction, alpha=a)
        res_pows.append(residual_power(ablated, direction))
        loc_intens.append(residual_location_intensity(ablated))
        rdir = random_direction_like(direction, rng)
        ctrl = ablate(test, rdir, alpha=a)
        ctrl_res_pows.append(residual_power(ctrl, direction))
        ctrl_loc.append(residual_location_intensity(ctrl))
    res_pows, loc_intens = np.array(res_pows), np.array(loc_intens)
    ctrl_res_pows, ctrl_loc = np.array(ctrl_res_pows), np.array(ctrl_loc)
    idx1 = np.argmin(np.abs(alphas - 1.0))
    idx2 = np.argmin(np.abs(alphas - 2.0))
    print(f"α=1 residual power {res_pows[idx1]:.6e} (was {base_res_pow:.4f})")
    print(f"random-ctrl {ctrl_res_pows[idx1]:.4f}")
    print(f"α=2 residual power {res_pows[idx2]:.6e}")
    out_dir = Path("/home/workdir/artifacts")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].imshow(np.abs(direction), cmap="magma", origin="lower")
    axes[0, 0].set_title("Recovered residual direction |amp|")
    axes[0, 1].imshow(np.angle(direction), cmap="twilight", origin="lower")
    axes[0, 1].set_title("Recovered residual direction phase")
    axes[1, 0].semilogy(alphas, res_pows + 1e-18, "o-", label="target")
    axes[1, 0].semilogy(alphas, ctrl_res_pows + 1e-18, "s--", label="random")
    axes[1, 0].axvline(1.0, color="gray", ls=":"); axes[1, 0].axvline(2.0, color="gray", ls="--")
    axes[1, 0].legend(); axes[1, 0].set_title("Selective nulling")
    axes[1, 1].plot(alphas, loc_intens / base_loc, "o-", label="target")
    axes[1, 1].plot(alphas, ctrl_loc / base_loc, "s--", label="random")
    axes[1, 1].legend(); axes[1, 1].set_title("Local intensity")
    fig.suptitle("Pure Optical Residual-Stream Toolkit")
    fig.tight_layout()
    fig.savefig(out_dir / "light_residual_phase_cancel.png", dpi=150)
    plt.close(fig)
    return {"alphas": alphas, "res_pows": res_pows, "base_res_pow": base_res_pow}

if __name__ == "__main__":
    run_experiment()
    print("Done.")
