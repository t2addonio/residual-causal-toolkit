"""Causal filter suite: path-patch, zero, random, residual-atom selectivity."""
import numpy as np
import torch

def residual_power(X, direction):
    if isinstance(X, torch.Tensor):
        d = direction / (direction.norm() + 1e-12)
        coeffs = X @ d
        return float((coeffs**2).mean().item())
    X = np.asarray(X)
    d = direction / (np.linalg.norm(direction) + 1e-12)
    coeffs = X @ d
    return float(np.mean(coeffs**2))

def apply_causal_filters(X_sig, X_held, direction, name="port", rng=None):
    if isinstance(X_sig, torch.Tensor):
        X_sig = X_sig.detach().cpu().numpy()
    if isinstance(X_held, torch.Tensor):
        X_held = X_held.detach().cpu().numpy()
    if isinstance(direction, torch.Tensor):
        direction = direction.detach().cpu().numpy()
    X_sig = np.asarray(X_sig, dtype=np.float64)
    X_held = np.asarray(X_held, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
    d = direction / (np.linalg.norm(direction) + 1e-12)
    c_sig = X_sig @ d
    c_held_mean = float(np.mean(X_held @ d))
    power_sig = float(np.mean(c_sig**2))
    X_path = X_sig - np.outer(c_sig, d) + c_held_mean * d
    power_path = residual_power(X_path, d)
    X_zero = X_sig - np.outer(c_sig, d)
    power_zero = residual_power(X_zero, d)
    r = rng.standard_normal(d.shape)
    r /= (np.linalg.norm(r) + 1e-12)
    power_rand = residual_power(X_sig, r)
    passed = bool(
        power_path < 0.30 * power_sig + 1e-8
        and power_zero < 0.08 * power_sig + 1e-10
        and power_rand > 0.35 * power_sig - 1e-8
        and power_sig > 1e-6
    )
    return {"name": name, "power": power_sig, "path": power_path, "zero": power_zero,
            "rand": power_rand, "pass": passed, "direction": d.copy()}

def extract_and_filter_ports(X_sig, X_held, stream_name="R", max_svd=3, rng=None):
    if isinstance(X_sig, torch.Tensor):
        X_sig = X_sig.detach().cpu().numpy()
    if isinstance(X_held, torch.Tensor):
        X_held = X_held.detach().cpu().numpy()
    X_sig = np.asarray(X_sig, dtype=np.float64)
    X_held = np.asarray(X_held, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng(0)
    results = []
    contrastive = X_sig.mean(axis=0) - X_held.mean(axis=0)
    if np.linalg.norm(contrastive) > 1e-8:
        results.append(apply_causal_filters(X_sig, X_held, contrastive, name=f"{stream_name}_contrastive", rng=rng))
    resid_mat = X_sig - X_held.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(resid_mat, full_matrices=False)
        for k in range(min(max_svd, len(S))):
            if S[k] < 1e-8:
                break
            results.append(apply_causal_filters(X_sig, X_held, Vt[k], name=f"{stream_name}_svd_{k}", rng=rng))
    except Exception:
        pass
    for ax in range(min(X_sig.shape[1], 6)):
        direction = np.zeros(X_sig.shape[1]); direction[ax] = 1.0
        results.append(apply_causal_filters(X_sig, X_held, direction, name=f"{stream_name}_ax_{ax}", rng=rng))
    return results
