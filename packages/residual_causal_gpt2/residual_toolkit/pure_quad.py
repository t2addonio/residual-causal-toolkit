"""Pure-quadratic residual expansion + second-moment residual features."""
import numpy as np
import torch

def pure_quadratic_14d(x):
    """x: (..., 4) -> (..., 14). 4 linear + 4 pure squares + 6 products, no bias."""
    if isinstance(x, torch.Tensor):
        x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        return torch.stack([
            x0, x1, x2, x3,
            x0**2, x1**2, x2**2, x3**2,
            x0*x1, x0*x2, x0*x3, x1*x2, x1*x3, x2*x3
        ], dim=-1)
    x = np.asarray(x)
    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    return np.stack([
        x0, x1, x2, x3,
        x0**2, x1**2, x2**2, x3**2,
        x0*x1, x0*x2, x0*x3, x1*x2, x1*x3, x2*x3
    ], axis=-1)

def second_moment_features(X):
    """R3-style second-moment residual features. X: (n, d) -> (n, 5)"""
    if isinstance(X, torch.Tensor):
        power = (X**2).sum(dim=-1, keepdim=True)
        maxabs = X.abs().amax(dim=-1, keepdim=True)
        return torch.cat([power, maxabs, power**2, maxabs**2, power * maxabs], dim=-1)
    X = np.asarray(X)
    power = np.sum(X**2, axis=-1, keepdims=True)
    maxabs = np.max(np.abs(X), axis=-1, keepdims=True)
    return np.concatenate([power, maxabs, power**2, maxabs**2, power * maxabs], axis=-1)
