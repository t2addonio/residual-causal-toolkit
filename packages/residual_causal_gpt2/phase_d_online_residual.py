#!/usr/bin/env python3
"""
Phase D — Online residual control on GPT-2 residual stream (standalone)
Residual intervention DURING forward (write hook).

FIX (2026-08-21): residual power / coefficient are now measured POST-edit.
Previous version recorded pre-edit c, so baseline/zero/path_held always
reported identical residual_power and the causal checks could never pass.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def make_modular_prompts(p=113, n_sig=64, n_held=64, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.randint(0, p, size=n_sig + n_held)
    b = rng.randint(0, p, size=n_sig + n_held)
    s = (a + b) % p
    texts = [f"{int(aa)} + {int(bb)} =" for aa, bb in zip(a, b)]
    labels = s.tolist()
    strength = np.abs(s - p / 2.0)
    order = np.argsort(-strength)
    sig_idx, held_idx = order[:n_sig], order[-n_held:]
    return ([texts[i] for i in sig_idx], [texts[i] for i in held_idx],
            [labels[i] for i in sig_idx], [labels[i] for i in held_idx])

def contrastive_direction(sig, held):
    d = sig.mean(0) - held.mean(0)
    return d / (np.linalg.norm(d) + 1e-12)

def residual_power(mat, d):
    return float(np.mean((mat @ d) ** 2))

def random_direction(dim, seed):
    rng = np.random.RandomState(seed)
    r = rng.randn(dim)
    return r / (np.linalg.norm(r) + 1e-12)

@torch.no_grad()
def collect_residual(model, tok, texts, layer, device):
    model.eval()
    cache = {}
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cache["h"] = h.detach()
    handle = model.transformer.h[layer].register_forward_hook(hook_fn)
    vecs = []
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
        enc = {k: v.to(device) for k, v in enc.items()}
        _ = model(**enc)
        vecs.append(cache["h"][0, -1].float().cpu().numpy())
    handle.remove()
    return np.stack(vecs, axis=0)

class OnlineResidualEditHook:
    """resid ← resid − (resid · d) d + c_target · d at last token. Metrics POST-edit."""
    def __init__(self, model, layer, d, c_target=None, enabled=True):
        self.model, self.layer = model, layer
        self.d = torch.tensor(d, dtype=torch.float32)
        self.c_target, self.enabled = c_target, enabled
        self._handle = None
        self.last_c_mean = self.last_power = float("nan")
        self._power_sum = self._c_sum = 0.0
        self._n = 0

    def _hook_fn(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        d = self.d.to(device=h.device, dtype=h.dtype)
        d = d / (d.norm() + 1e-12)
        c = (h[:, -1, :] * d.view(1, -1)).sum(dim=-1)
        if self.enabled and self.c_target is not None:
            delta = (self.c_target - c).view(-1, 1) * d.view(1, -1)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + delta
            c = (h[:, -1, :] * d.view(1, -1)).sum(dim=-1)
        c_mean = float(c.detach().mean().cpu())
        p = float((c.detach() ** 2).mean().cpu())
        self.last_c_mean = c_mean
        self.last_power = p
        self._c_sum += c_mean
        self._power_sum += p
        self._n += 1
        return (h,) + output[1:] if isinstance(output, tuple) else h

    def attach(self):
        self.detach()
        self._handle = self.model.transformer.h[self.layer].register_forward_hook(self._hook_fn)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def mean_power(self):
        return self._power_sum / max(self._n, 1)

    def mean_c(self):
        return self._c_sum / max(self._n, 1)

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *a):
        self.detach()

@torch.no_grad()
def score_batch(model, tok, texts, labels, device):
    model.eval()
    logprobs, hits = [], 0
    for text, y in zip(texts, labels):
        prompt_ids = tok.encode(text, add_special_tokens=False)
        ans_ids = tok.encode(f" {int(y)}", add_special_tokens=False) or tok.encode(str(int(y)), add_special_tokens=False)
        first_ans = ans_ids[0]
        logits = model(input_ids=torch.tensor([prompt_ids], device=device)).logits[0, -1]
        logprobs.append(float(torch.log_softmax(logits, dim=-1)[first_ans].cpu()))
        if int(torch.argmax(logits).item()) == first_ans:
            hits += 1
    return {"mean_logprob": float(np.mean(logprobs)), "accuracy_first_token": hits / max(len(texts), 1), "n": len(texts)}

@torch.no_grad()
def run_condition(model, tok, texts, labels, layer, d, c_target, enabled, device):
    with OnlineResidualEditHook(model, layer, d, c_target=c_target, enabled=enabled) as hook:
        m = score_batch(model, tok, texts, labels, device)
        m["residual_c_mean"] = hook.mean_c()
        m["residual_power"] = hook.mean_power()
        m["residual_c_last"] = hook.last_c_mean
        m["residual_power_last"] = hook.last_power
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n-sig", type=int, default=64)
    ap.add_argument("--n-held", type=int, default=64)
    ap.add_argument("--modulus", type=int, default=113)
    ap.add_argument("--out-dir", type=str, default="results/phase_d")
    args = ap.parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable — falling back to cpu")
        device = "cpu"
    print(f"[Phase D] seed={args.seed} device={device} layer={args.layer}")
    tok = GPT2Tokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    sig_texts, held_texts, sig_labels, held_labels = make_modular_prompts(
        p=args.modulus, n_sig=args.n_sig, n_held=args.n_held, seed=args.seed)
    print("  collecting residual activations…")
    sig_resid = collect_residual(model, tok, sig_texts, args.layer, device)
    held_resid = collect_residual(model, tok, held_texts, args.layer, device)
    d = contrastive_direction(sig_resid, held_resid)
    c_held = float((held_resid @ d).mean())
    c_sig = float((sig_resid @ d).mean())
    p0_off = residual_power(sig_resid, d)
    print(f"  residual c_sig={c_sig:.4f} c_held={c_held:.4f} power={p0_off:.4f}")
    d_rand = random_direction(sig_resid.shape[1], seed=args.seed + 97)
    c_held_rand = float((held_resid @ d_rand).mean())
    conditions = {
        "baseline":  dict(d=d, c_target=None, enabled=False),
        "zero":      dict(d=d, c_target=0.0, enabled=True),
        "path_held": dict(d=d, c_target=c_held, enabled=True),
        "random":    dict(d=d_rand, c_target=c_held_rand, enabled=True),
    }
    results = {}
    for name, cfg in conditions.items():
        print(f"  residual condition: {name}")
        m = run_condition(model, tok, sig_texts, sig_labels, args.layer,
                          d=cfg["d"], c_target=cfg["c_target"], enabled=cfg["enabled"], device=device)
        results[name] = m
        print(f"    power={m['residual_power']:.4f} c={m['residual_c_mean']:.4f} "
              f"logprob={m['mean_logprob']:.4f} acc={m['accuracy_first_token']:.3f}")
    p0 = results["baseline"]["residual_power"]
    p_zero = results["zero"]["residual_power"]
    p_path = results["path_held"]["residual_power"]
    p_rand = results["random"]["residual_power"]
    lp0 = results["baseline"]["mean_logprob"]
    summary = {
        "seed": args.seed, "layer": args.layer,
        "offline": {"c_sig": c_sig, "c_held": c_held, "power": p0_off},
        "online": results,
        "checks": {
            "power_baseline": p0, "power_zero": p_zero,
            "power_path_held": p_path, "power_random": p_rand,
            "zero_collapses": bool(p_zero < 0.15 * p0 + 1e-8),
            "path_moves": bool(p_path < 0.70 * p0 + 1e-8),
            "random_intact": bool(p_rand > 0.20 * p0),
            "delta_logprob_zero": results["zero"]["mean_logprob"] - lp0,
            "delta_logprob_path": results["path_held"]["mean_logprob"] - lp0,
            "delta_logprob_random": results["random"]["mean_logprob"] - lp0,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase_d_seed{args.seed}_layer{args.layer}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Phase D summary (POST-edit residual power) ===")
    print(f"  residual power: base={p0:.4f} zero={p_zero:.4e} path={p_path:.4f} rand={p_rand:.4f}")
    print(f"  zero_collapses={summary['checks']['zero_collapses']} "
          f"path_moves={summary['checks']['path_moves']} "
          f"random_intact={summary['checks']['random_intact']}")
    print(f"  wrote {out_path}")

if __name__ == "__main__":
    main()
