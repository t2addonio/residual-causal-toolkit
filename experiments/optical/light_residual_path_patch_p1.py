#!/usr/bin/env python3
"""
light_residual_path_patch_p1.py

T³ Mode B — Experiment P1: Single-mode residual path patch
Path-patch the residual coefficient of a residual-containing field
with the coefficient from a held-out field (and with zero / random
controls). Tests whether the residual component mediates the
difference between signal and held-out ensembles.
Pure classical optics. Single residual mode.
"""

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

def residual_mode(amp=0.75, w0=4.2, x0=8.5, y0=-5.5, phase=0.85):
    rr2 = (XX - x0)**2 + (YY - y0)**2
    return amp * np.exp(-rr2 / w0**2) * np.exp(1j * phase)

def make_ensemble(n, include_residual=True, noise=0.03, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    fields = []
    for _ in range(n):
        base = gaussian_beam(phase=rng.uniform(0, 0.2))
        if include_residual:
            a = 0.85 + 0.3 * rng.random()
            ph = rng.uniform(-0.2, 0.2)
            base = base + a * residual_mode(phase=0.85 + ph)
        pn = noise * rng.normal(size=(N, N))
        an = 1.0 + 0.015 * rng.normal(size=(N, N))
        fields.append(base * an * np.exp(1j * pn))
    return np.stack(fields)

def complex_inner(a, b):
    return np.vdot(a.ravel(), b.ravel())

def residual_power(field, direction):
    return float(np.abs(complex_inner(direction, field))**2)

def local_intensity(field, x0=8.5, y0=-5.5, radius=5.0):
    mask = ((XX - x0)**2 + (YY - y0)**2) < radius**2
    return float(np.mean(np.abs(field[mask])**2))

def path_patch(field, direction, new_coeff):
    old_coeff = complex_inner(direction, field)
    return field - old_coeff * direction + new_coeff * direction

def run_p1(seed=0, n_train=200, n_held=100):
    rng = np.random.default_rng(seed)
    print("Experiment P1 — Single-mode residual path patch")
    signal = make_ensemble(n_train, include_residual=True, rng=rng)
    heldout = make_ensemble(n_held, include_residual=False, rng=rng)
    residual = np.mean(signal, axis=0) - np.mean(heldout, axis=0)
    direction = residual / (np.linalg.norm(residual) + 1e-12)
    test_sig = signal[0]
    test_held = heldout[0]
    c_sig = complex_inner(direction, test_sig)
    c_held_mean = np.mean([complex_inner(direction, h) for h in heldout])
    field_id = test_sig.copy()
    field_patch_held = path_patch(test_sig, direction, c_held_mean)
    field_patch_zero = path_patch(test_sig, direction, 0.0 + 0.0j)
    rdir = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    rdir = rdir / np.linalg.norm(rdir)
    c_sig_rand = complex_inner(rdir, test_sig)
    c_held_rand = np.mean([complex_inner(rdir, h) for h in heldout])
    field_patch_rand = test_sig - c_sig_rand * rdir + c_held_rand * rdir

    def metrics(field, label):
        rp = residual_power(field, direction)
        li = local_intensity(field)
        print(f"    {label:22s}  residual power = {rp:10.4f}   local intensity = {li:.6f}")
        return rp, li

    rp_id, li_id = metrics(field_id, "signal (no patch)")
    rp_held, li_held = metrics(field_patch_held, "path-patch → held")
    rp_zero, li_zero = metrics(field_patch_zero, "path-patch → zero")
    rp_rand, li_rand = metrics(field_patch_rand, "random-dir patch")
    rp_held_field, li_held_field = metrics(test_held, "held-out field")

    out_dir = Path("/home/workdir/artifacts")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes[0, 0].imshow(np.abs(direction), cmap="magma", origin="lower")
    axes[0, 0].set_title("Residual direction |amp|")
    axes[0, 1].imshow(np.abs(test_sig)**2, cmap="inferno", origin="lower")
    axes[0, 1].set_title("Signal field |E|² (no patch)")
    axes[0, 2].imshow(np.abs(field_patch_held)**2, cmap="inferno", origin="lower")
    axes[0, 2].set_title("Path-patched → held coefficient |E|²")
    for ax in axes[0]:
        ax.set_xticks([]); ax.set_yticks([])
    labels = ["signal", "path-patch\n→ held", "path-patch\n→ zero", "random-dir\npatch", "held-out\nfield"]
    colors = ["C0", "C2", "C3", "C1", "C4"]
    axes[1, 0].bar(labels, [rp_id, rp_held, max(rp_zero, 1e-16), rp_rand, rp_held_field], color=colors, alpha=0.85)
    axes[1, 0].set_yscale("log"); axes[1, 0].set_title("Residual power under path patching")
    axes[1, 1].bar(labels, [li_id, li_held, li_zero, li_rand, li_held_field], color=colors, alpha=0.85)
    axes[1, 1].set_title("Local intensity under path patching")
    axes[1, 2].axis("off")
    fig.suptitle("P1 — Single-mode Residual Path Patch")
    fig.tight_layout()
    fig.savefig(out_dir / "light_residual_path_patch_p1.png", dpi=150)
    plt.close(fig)
    print("Done. P1 path-patch complete.")

if __name__ == "__main__":
    run_p1()
