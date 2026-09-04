"""Modular arithmetic residual task for clean contrastive residual ensembles."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict

@dataclass
class ModArithConfig:
    p: int = 113
    n_train: int = 8000
    n_val: int = 2000
    n_sig: int = 512
    n_held: int = 512
    seed: int = 0

class ModArithDataset(Dataset):
    def __init__(self, pairs: np.ndarray, p: int):
        self.pairs = pairs
        self.p = p
        self.equals = p
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        ans = (int(a) + int(b)) % self.p
        x = torch.tensor([a, b, self.equals], dtype=torch.long)
        y = torch.tensor(ans, dtype=torch.long)
        return x, y

def make_mod_arith_splits(cfg: ModArithConfig):
    rng = np.random.default_rng(cfg.seed)
    all_pairs = np.array([(a, b) for a in range(cfg.p) for b in range(cfg.p)], dtype=np.int64)
    rng.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = min(cfg.n_train, int(0.7 * n))
    n_val = min(cfg.n_val, n - n_train)
    train = all_pairs[:n_train]
    val = all_pairs[n_train:n_train + n_val]
    rest = all_pairs[n_train + n_val:]
    sig_mask = rest[:, 0] < cfg.p // 2
    sig = rest[sig_mask][:cfg.n_sig]
    held = rest[~sig_mask][:cfg.n_held]
    if len(sig) < cfg.n_sig:
        sig = rest[:cfg.n_sig]
    if len(held) < cfg.n_held:
        held = rest[cfg.n_sig:cfg.n_sig + cfg.n_held]
    return {"train": train, "val": val, "sig": sig, "held": held}

def make_loaders(cfg: ModArithConfig, batch_size: int = 128):
    splits = make_mod_arith_splits(cfg)
    loaders = {}
    for k, pairs in splits.items():
        loaders[k] = DataLoader(ModArithDataset(pairs, cfg.p), batch_size=batch_size, shuffle=(k == "train"))
    return loaders, splits
