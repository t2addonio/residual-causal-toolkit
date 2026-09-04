#!/usr/bin/env python3
"""Residual Interferometer — QPU-like mix for residual cores.
PHASE(φ) + BEAM(θ) unitary on a 2-D residual plane. Complement never written.
Does NOT turn a residual stream into a QPU.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def orthonormalize_pair(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    u = unit(np.asarray(u, dtype=float).ravel())
    v = np.asarray(v, dtype=float).ravel()
    v = unit(v - np.dot(u, v) * u)
    return u, v

def plane_coords(R: np.ndarray, u: np.ndarray, v: np.ndarray):
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        return float(np.dot(R, u)), float(np.dot(R, v))
    return R @ u, R @ v

def reconstruct_from_plane(R, u, v, alpha, beta):
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        perp = R - np.dot(R, u) * u - np.dot(R, v) * v
        return perp + alpha * u + beta * v
    perp = R - np.outer(R @ u, u) - np.outer(R @ v, v)
    return perp + np.outer(alpha, u) + np.outer(beta, v)

def su2(theta: float, phi: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -np.exp(1j * phi) * s], [np.exp(-1j * phi) * s, c]], dtype=complex)

def apply_su2_real_plane(alpha, beta, theta, phi):
    U = su2(theta, phi)
    stacked = np.stack([np.asarray(alpha, dtype=complex), np.asarray(beta, dtype=complex)], axis=0)
    mixed = U @ stacked
    return mixed[0].real, mixed[1].real

def phase_rotate_plane(alpha, beta, phi):
    c, s = np.cos(phi), np.sin(phi)
    return c * alpha - s * beta, s * alpha + c * beta

def unitary_interferometer(R, u, v, theta=0.0, phi=0.0, mode="su2"):
    u, v = orthonormalize_pair(u, v)
    alpha, beta = plane_coords(R, u, v)
    if mode == "phase":
        a2, b2 = phase_rotate_plane(alpha, beta, phi)
    else:
        a2, b2 = apply_su2_real_plane(alpha, beta, theta, phi)
    return reconstruct_from_plane(R, u, v, a2, b2)

def subtractive_mix(R, u, v, coeff: float):
    u, v = orthonormalize_pair(u, v)
    alpha, beta = plane_coords(R, u, v)
    return reconstruct_from_plane(R, u, v, (1.0 - coeff) * alpha, (1.0 - coeff) * beta)

def path_patch_plane(R, u, v, alpha_t, beta_t):
    u, v = orthonormalize_pair(u, v)
    if np.asarray(R).ndim == 1:
        return reconstruct_from_plane(R, u, v, float(alpha_t), float(beta_t))
    return reconstruct_from_plane(R, u, v, np.full(len(R), alpha_t), np.full(len(R), beta_t))

def plane_power(R, u, v) -> float:
    alpha, beta = plane_coords(R, u, v)
    return float(np.mean(np.asarray(alpha) ** 2 + np.asarray(beta) ** 2))

def total_power(R) -> float:
    R = np.asarray(R)
    return float(np.dot(R, R)) if R.ndim == 1 else float(np.mean(np.sum(R ** 2, axis=-1)))

def interference_readout(R, u, v) -> float:
    alpha, _ = plane_coords(R, u, v)
    return float(np.mean(np.asarray(alpha) ** 2))

def fringe_contrast(curve) -> float:
    curve = np.asarray(curve, dtype=float)
    return float((curve.max() - curve.min()) / (curve.max() + curve.min() + 1e-12))

def complex_inner(a, b):
    return np.vdot(a.ravel(), b.ravel())

def orthonormalize_complex_pair(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b - complex_inner(a, b) * a
    b = b / (np.linalg.norm(b) + 1e-12)
    return a, b

def optical_phase_arm_b(field, mode_a, mode_b, phi):
    mode_a, mode_b = orthonormalize_complex_pair(mode_a, mode_b)
    alpha = complex_inner(mode_a, field)
    beta = complex_inner(mode_b, field)
    perp = field - alpha * mode_a - beta * mode_b
    return perp + alpha * mode_a + (np.exp(1j * phi) * beta) * mode_b

def optical_interferometer(field, mode_a, mode_b, theta=0.0, phi=0.0):
    mode_a, mode_b = orthonormalize_complex_pair(mode_a, mode_b)
    alpha = complex_inner(mode_a, field)
    beta = complex_inner(mode_b, field)
    mixed = su2(theta, phi) @ np.array([alpha, beta], dtype=complex)
    perp = field - alpha * mode_a - beta * mode_b
    return perp + mixed[0] * mode_a + mixed[1] * mode_b

def optical_subspace_power(field, mode_a, mode_b):
    mode_a, mode_b = orthonormalize_complex_pair(mode_a, mode_b)
    return float(abs(complex_inner(mode_a, field)) ** 2 + abs(complex_inner(mode_b, field)) ** 2)

@dataclass
class FringeResult:
    phi: np.ndarray
    readout: np.ndarray
    random_readout: np.ndarray
    plane_power: np.ndarray
    total_power: np.ndarray
    contrast: float
    unitary_ok: bool
    random_contrast: float
    subtractive_energy_change: float
    notes: Dict[str, float] = field(default_factory=dict)

def sweep_phase(R, u, v, n_phi=33, theta=0.0, mode="phase", rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    u, v = orthonormalize_pair(u, v)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi)
    read, ppow, tpow = [], [], []
    R0_power = total_power(R)
    plane0 = plane_power(R, u, v)
    for phi in phis:
        Rp = unitary_interferometer(R, u, v, theta=theta, phi=phi, mode=mode)
        read.append(interference_readout(Rp, u, v))
        ppow.append(plane_power(Rp, u, v))
        tpow.append(total_power(Rp))
    read, ppow, tpow = map(np.asarray, (read, ppow, tpow))
    ru, rv = orthonormalize_pair(unit(rng.normal(size=u.shape)), unit(rng.normal(size=v.shape)))
    rread = np.array([interference_readout(unitary_interferometer(R, ru, rv, theta=theta, phi=phi, mode=mode), ru, rv) for phi in phis])
    energy_change = abs(total_power(subtractive_mix(R, u, v, 1.0)) - R0_power) / (R0_power + 1e-12)
    unitary_ok = bool(np.max(np.abs(ppow - plane0)) < 1e-6 * (plane0 + 1.0) and np.max(np.abs(tpow - R0_power)) < 1e-6 * (R0_power + 1.0))
    return FringeResult(phi=phis, readout=read, random_readout=rread, plane_power=ppow, total_power=tpow,
                        contrast=fringe_contrast(read), unitary_ok=unitary_ok, random_contrast=fringe_contrast(rread),
                        subtractive_energy_change=float(energy_change),
                        notes={"plane_power0": plane0, "total_power0": R0_power})

def extract_2d_core(X_sig, X_held):
    resid = X_sig - X_held.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(resid, full_matrices=False)
    u, v = orthonormalize_pair(Vt[0], Vt[1])
    return u, v, S[:2]
