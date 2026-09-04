"""GPT-2 residual stream extraction via HuggingFace hooks."""
from __future__ import annotations
import torch
from typing import List, Optional, Dict

def load_gpt2(model_name: str = "gpt2", device: str = "cuda"):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tok

class ResidualStreamCache:
    def __init__(self, model, layer: int = 6):
        self.model = model
        self.layer = layer
        self.cache: Dict[str, torch.Tensor] = {}
        self._handles = []
        self._register()

    def _register(self):
        block = self.model.transformer.h[self.layer]
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.cache["residual"] = h.detach()
        self._handles.append(block.register_forward_hook(hook_fn))

    def clear(self):
        self.cache = {}

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

@torch.no_grad()
def extract_residual_batch(model, tok, texts: List[str], layer: int = 6, position: int = -1, device: str = "cuda", max_length: int = 64):
    cache = ResidualStreamCache(model, layer=layer)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    _ = model(**enc)
    h = cache.cache["residual"]
    cache.close()
    if position < 0:
        lengths = enc["attention_mask"].sum(dim=1) - 1
        return torch.stack([h[i, L] for i, L in enumerate(lengths.tolist())], dim=0)
    return h[:, position, :]

@torch.no_grad()
def extract_residual_matrix(model, tok, texts: List[str], layer: int = 6, position: int = -1, device: str = "cuda", batch_size: int = 16):
    import numpy as np
    chunks = []
    for i in range(0, len(texts), batch_size):
        v = extract_residual_batch(model, tok, texts[i:i+batch_size], layer=layer, position=position, device=device)
        chunks.append(v.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)
