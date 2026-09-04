"""Extract residual stream matrices for contrastive residual analysis."""
from __future__ import annotations
import numpy as np
import torch
from typing import Tuple
from .model import MiniGPT
from .data import ModArithConfig, make_loaders

@torch.no_grad()
def collect_residual_matrix(model: MiniGPT, loader, device: torch.device, layer: int = -1, token_pos: int = -1) -> np.ndarray:
    model.eval()
    mats = []
    for x, y in loader:
        x = x.to(device)
        _, _, resid = model(x, return_residuals=True)
        L = resid.shape[0]
        li = layer if layer >= 0 else (L - 1)
        li = max(0, min(li, L - 1))
        h = resid[li, :, token_pos, :]
        mats.append(h.cpu().numpy())
    return np.concatenate(mats, axis=0)

def collect_sig_held_residuals(model: MiniGPT, cfg: ModArithConfig, device: torch.device, layer: int = -1, token_pos: int = -1, batch_size: int = 128):
    loaders, _ = make_loaders(cfg, batch_size=batch_size)
    X_sig = collect_residual_matrix(model, loaders["sig"], device, layer=layer, token_pos=token_pos)
    X_held = collect_residual_matrix(model, loaders["held"], device, layer=layer, token_pos=token_pos)
    return X_sig, X_held
