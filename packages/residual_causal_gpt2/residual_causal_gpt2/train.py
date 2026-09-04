"""Train MiniGPT on modular arithmetic and optionally log residual trajectory."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
from torch.optim import AdamW
from .model import MiniGPT, GPTConfig
from .data import ModArithConfig, make_loaders
from .residual_extract import collect_sig_held_residuals
from .residual_math import run_parallel_ports

def train_one_seed(seed, out_dir, p=113, n_layer=4, n_embd=128, n_head=4, steps=3000,
                  batch_size=128, lr=3e-4, device=None, residual_every=500, residual_layer=-1):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    data_cfg = ModArithConfig(p=p, seed=seed)
    loaders, _ = make_loaders(data_cfg, batch_size=batch_size)
    model = MiniGPT(GPTConfig(vocab_size=p+1, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=4)).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_iter = iter(loaders["train"]); history, residual_logs = [], []
    t0 = time.time(); model.train()
    for step in range(1, steps+1):
        try: x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"]); x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 100 == 0 or step == 1:
            acc = model.accuracy(loaders["val"], device)
            history.append({"step": step, "loss": float(loss.item()), "val_acc": float(acc)})
            print(f"[seed={seed}] step {step:5d} loss={loss.item():.4f} val_acc={acc:.3f}")
        if residual_every > 0 and (step % residual_every == 0 or step == steps):
            model.eval()
            X_sig, X_held = collect_sig_held_residuals(model, data_cfg, device, layer=residual_layer, batch_size=batch_size)
            ports = run_parallel_ports(X_sig, X_held, rng=np.random.default_rng(seed+step))
            residual_logs.append({"step": step, "n_pass": ports["n_pass"], "n_tested": ports["n_tested"],
                                  "consensus_power": ports["consensus_power"]})
            model.train()
    ckpt_path = out_dir / f"seed_{seed}.pt"
    torch.save({"model_state": model.state_dict(), "seed": seed, "history": history, "residual_logs": residual_logs}, ckpt_path)
    print(f"[seed={seed}] saved {ckpt_path}")
    return {"seed": seed, "ckpt": str(ckpt_path)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="runs/train")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--residual_every", type=int, default=500)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    train_one_seed(seed=args.seed, out_dir=Path(args.out_dir), steps=args.steps, n_embd=args.n_embd,
                   n_layer=args.n_layer, batch_size=args.batch_size, residual_every=args.residual_every, device=args.device)

if __name__ == "__main__":
    main()
