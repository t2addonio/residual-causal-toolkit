"""Residual stream extraction hooks for HF GPT-2."""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, List, Optional

class ResidualStreamHook:
    def __init__(self, model: nn.Module, layers: Optional[List[int]] = None):
        self.model = model
        self.layers = layers
        self.cache: Dict[str, torch.Tensor] = {}
        self._handles = []
        self._register()

    def _register(self):
        transformer = getattr(self.model, "transformer", self.model)
        blocks = getattr(transformer, "h", None)
        if blocks is None:
            raise ValueError("Could not find GPT-2 transformer blocks (model.transformer.h)")
        n_layers = len(blocks)
        target_layers = self.layers if self.layers is not None else list(range(n_layers))
        for i in target_layers:
            if 0 <= i < n_layers:
                self._handles.append(blocks[i].register_forward_hook(self._make_hook(i)))
        wte = getattr(transformer, "wte", None)
        if wte is not None:
            self._handles.append(wte.register_forward_hook(self._make_embed_hook()))

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self.cache[f"resid_post_{layer_idx}"] = h.detach()
        return hook

    def _make_embed_hook(self):
        def hook(module, inputs, output):
            self.cache["resid_embed"] = output.detach()
        return hook

    def clear(self):
        self.cache = {}

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.cache = {}

    def get(self, key: str):
        return self.cache.get(key)

def get_residual_at_layer(hook: ResidualStreamHook, layer: int, position: int = -1):
    key = f"resid_post_{layer}" if layer >= 0 else "resid_embed"
    h = hook.get(key)
    if h is None:
        raise KeyError(f"Residual key {key} not in cache. Available: {list(hook.cache.keys())}")
    return h[:, position, :]
