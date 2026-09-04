"""Domain-agnostic residual causal primitives (R1/R2/R3 parallel ports)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np

def unit(v): return v / (np.linalg.norm(v) + 1e-12)
def pure_quadratic_14d(x):
    x0,x1,x2,x3 = x[...,0],x[...,1],x[...,2],x[...,3]
    return np.stack([x0,x1,x2,x3,x0**2,x1**2,x2**2,x3**2,x0*x1,x0*x2,x0*x3,x1*x2,x1*x3,x2*x3], axis=-1)
def second_moment_features(x):
    power = np.sum(x**2, axis=-1, keepdims=True)
    maxabs = np.max(np.abs(x), axis=-1, keepdims=True)
    return np.concatenate([power, maxabs, power**2, maxabs**2, power*maxabs], axis=-1)
def residual_power(X, d): return float(np.mean((X @ unit(d))**2))
def contrastive_direction(Xs, Xh): return unit(Xs.mean(0)-Xh.mean(0))
def svd_directions(X, k=3):
    try:
        _,_,Vt = np.linalg.svd(X-X.mean(0,keepdims=True), full_matrices=False)
        return [unit(Vt[i]) for i in range(min(k, Vt.shape[0]))]
    except Exception:
        return []

@dataclass
class PortResult:
    name: str; power: float; path: float; zero: float; rand: float; held: float; passed: bool; direction: np.ndarray

def apply_causal_filters(X_sig, X_held, direction, name="port", rng=None):
    rng = rng or np.random.default_rng(abs(hash(name))%(2**32))
    d = unit(direction); c = X_sig @ d; ch = float(np.mean(X_held @ d))
    p0 = float(np.mean(c**2)); held_p = float(np.mean((X_held@d)**2))
    p_path = residual_power(X_sig - np.outer(c,d) + ch*d, d)
    p_zero = residual_power(X_sig - np.outer(c,d), d)
    p_rand = residual_power(X_sig, unit(rng.normal(size=d.shape)))
    passed = bool((p_path < 0.35*p0 + 1e-8 or abs(p_path-held_p) < 0.30*p0) and p_zero < 0.10*p0 + 1e-9 and p_rand > 0.30*p0 and p0 > 1e-8)
    return PortResult(name, p0, p_path, p_zero, p_rand, held_p, passed, d.copy())

def residual_atom_selectivity(X_sig, residual_dir, rng=None):
    rng = rng or np.random.default_rng(0)
    d = unit(residual_dir); p0 = residual_power(X_sig, d)
    p_res = residual_power(X_sig - np.outer(X_sig@d, d), d)
    r = unit(rng.normal(size=d.shape))
    p_rand = residual_power(X_sig - np.outer(X_sig@r, r), d)
    return {"residual_power_before": p0, "residual_atom_retained_frac": p_res/(p0+1e-12), "random_atom_retained_frac": p_rand/(p0+1e-12)}

def run_parallel_ports(X_sig, X_held, base_dim=4, rng=None):
    rng = rng or np.random.default_rng(42)
    if X_sig.shape[1] > base_dim:
        basis = [contrastive_direction(X_sig, X_held)] + svd_directions(X_sig-X_held.mean(0), k=base_dim-1)
        while len(basis) < base_dim: basis.append(unit(rng.normal(size=X_sig.shape[1])))
        B,_ = np.linalg.qr(np.stack(basis[:base_dim],1)); B = B[:,:base_dim]
        X4s, X4h = X_sig@B, X_held@B
    else:
        X4s, X4h, B = X_sig, X_held, None
    ports = [apply_causal_filters(X4s, X4h, contrastive_direction(X4s,X4h), "R1_contrastive", rng)]
    for i,sd in enumerate(svd_directions(X4s-X4h.mean(0), k=3)):
        ports.append(apply_causal_filters(X4s,X4h,sd,f"R1_svd_{i}",rng))
    X2s, X2h = pure_quadratic_14d(X4s), pure_quadratic_14d(X4h)
    ports.append(apply_causal_filters(X2s,X2h,contrastive_direction(X2s,X2h),"R2_contrastive",rng))
    for i,sd in enumerate(svd_directions(X2s-X2h.mean(0), k=4)):
        ports.append(apply_causal_filters(X2s,X2h,sd,f"R2_svd_{i}",rng))
    X3s, X3h = second_moment_features(X4s), second_moment_features(X4h)
    ports.append(apply_causal_filters(X3s,X3h,contrastive_direction(X3s,X3h),"R3_contrastive",rng))
    survivors = [p for p in ports if p.passed]
    return {"ports": ports, "survivors": survivors, "n_tested": len(ports), "n_pass": len(survivors),
            "consensus_power": float(np.mean([p.power for p in survivors])) if survivors else 0.0,
            "residual_atom": residual_atom_selectivity(X4s, contrastive_direction(X4s,X4h), rng),
            "X4_sig": X4s, "X4_held": X4h, "basis": B}
