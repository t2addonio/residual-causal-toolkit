#!/usr/bin/env python3
"""light_residual_sae.py — residual optical component as sparse dictionary atom 0."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

N = 64
X = (np.arange(N) - N / 2) * 1.0
Y = (np.arange(N) - N / 2) * 1.0
XX, YY = np.meshgrid(X, Y)
R2 = XX**2 + YY**2

def gaussian_beam(amp=1.0, w0=10.0, phase=0.0):
    return amp * np.exp(-R2 / w0**2) * np.exp(1j * phase)

def residual_mode(amp=0.8, w0=4.2, x0=8.5, y0=-5.5, phase=0.85):
    rr2 = (XX - x0)**2 + (YY - y0)**2
    return amp * np.exp(-rr2 / w0**2) * np.exp(1j * phase)

def make_ensemble(n_samples, include_residual=True, noise_level=0.03, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    fields = []
    for _ in range(n_samples):
        base = gaussian_beam(phase=rng.uniform(0, 0.2))
        if include_residual:
            a = 0.85 + 0.3 * rng.random()
            ph = rng.uniform(-0.2, 0.2)
            base = base + a * residual_mode(phase=0.85 + ph)
        phase_noise = noise_level * rng.normal(size=(N, N))
        amp_noise = 1.0 + 0.015 * rng.normal(size=(N, N))
        fields.append(base * amp_noise * np.exp(1j * phase_noise))
    return np.stack(fields, axis=0)

def complex_inner(a, b):
    return np.vdot(a.ravel(), b.ravel())

def residual_power(field, direction):
    return float(np.abs(complex_inner(direction, field))**2)

def run_experiment(seed=0, n_train=400, n_held=100, n_random_atoms=16):
    rng = np.random.default_rng(seed)
    signal = make_ensemble(n_train, include_residual=True, rng=rng)
    heldout = make_ensemble(n_held, include_residual=False, rng=rng)
    mean_held = np.mean(heldout, axis=0)
    direction = (np.mean(signal, axis=0) - mean_held)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    residuals = signal - mean_held[None, ...]
    dictionary = [direction]
    for _ in range(n_random_atoms):
        r = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
        dictionary.append(r / np.linalg.norm(r))
    dictionary = np.stack(dictionary, axis=0)
    base_powers, residual_coeffs, abl_powers, random_abl_powers = [], [], [], []
    for res in residuals:
        c0 = complex_inner(direction, res)
        residual_coeffs.append(c0)
        base_powers.append(residual_power(res, direction))
        abl_powers.append(residual_power(res - c0 * direction, direction))
        rand_dir = dictionary[1 + rng.integers(0, n_random_atoms)]
        c_rand = complex_inner(rand_dir, res)
        random_abl_powers.append(residual_power(res - c_rand * rand_dir, direction))
    base_powers, abl_powers = np.array(base_powers), np.array(abl_powers)
    random_abl_powers = np.array(random_abl_powers)
    print(f"baseline {base_powers.mean():.4f} residual-atom {abl_powers.mean():.6e} random-atom {random_abl_powers.mean():.4f}")
    out_dir = Path("/home/workdir/artifacts")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(np.abs(direction), cmap="magma", origin="lower")
    axes[0, 0].set_title("Residual dictionary atom")
    axes[1, 1].hist(base_powers, bins=30, alpha=0.7, label="baseline")
    axes[1, 1].hist(abl_powers, bins=30, alpha=0.7, label="residual atom")
    axes[1, 1].hist(random_abl_powers, bins=30, alpha=0.5, label="random atom")
    axes[1, 1].legend()
    axes[1, 2].bar(["baseline", "residual atom", "random atom"],
                   [base_powers.mean(), max(abl_powers.mean(), 1e-16), random_abl_powers.mean()])
    axes[1, 2].set_yscale("log")
    fig.suptitle("Residual Sparse Dictionary")
    fig.tight_layout()
    fig.savefig(out_dir / "light_residual_sae.png", dpi=150)
    plt.close(fig)
    return {"mean_base": float(base_powers.mean()), "mean_abl": float(abl_powers.mean())}

if __name__ == "__main__":
    run_experiment()
