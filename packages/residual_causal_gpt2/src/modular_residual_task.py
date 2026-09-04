"""Modular arithmetic residual task for GPT-2 residual causal experiments."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

def make_modular_prompts(p: int = 113, n_sig: int = 128, n_held: int = 128, seed: int = 0, residual_mode: str = "sum_mod"):
    rng = np.random.RandomState(seed)
    a = rng.randint(0, p, size=n_sig + n_held)
    b = rng.randint(0, p, size=n_sig + n_held)
    s = (a + b) % p
    texts = [f"{int(aa)} + {int(bb)} =" for aa, bb in zip(a, b)]
    labels = s.tolist()
    if residual_mode == "sum_mod":
        order = np.argsort(-np.abs(s - p / 2.0))
    elif residual_mode == "a_high":
        order = np.argsort(-a.astype(float))
    elif residual_mode == "parity":
        odd = np.where(s % 2 == 1)[0]; even = np.where(s % 2 == 0)[0]
        rng.shuffle(odd); rng.shuffle(even)
        return ([texts[i] for i in odd[:n_sig]], [texts[i] for i in even[:n_held]],
                [labels[i] for i in odd[:n_sig]], [labels[i] for i in even[:n_held]])
    else:
        order = np.arange(len(texts)); rng.shuffle(order)
    return ([texts[i] for i in order[:n_sig]], [texts[i] for i in order[-n_held:]],
            [labels[i] for i in order[:n_sig]], [labels[i] for i in order[-n_held:]])
