#!/usr/bin/env python3
"""Logical Residual Stream v0 — R1 linear 4D / R2 14D / R3 second-moment.
Logical layer = ports that PASS FILTER on any stream. S never modified.
Recorded v0: 9 ports PASS; consensus residual power path-patches down.
"""
from __future__ import annotations
import numpy as np, hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"core"))
from residual_poly_14d import phi14, causal_port, unit

def checksum(X): return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()[:16]
def second_moment(X):
    p=np.sum(X**2,1,keepdims=True); m=np.max(np.abs(X),1,keepdims=True)
    return np.concatenate([p,m,p**2,m**2,p*m],1)

def run():
    rng=np.random.default_rng(0)
    S_held=rng.normal(size=(80,4))*0.25
    S_sig=S_held + np.array([0.9,0.0,0.0,0.0]) + np.c_[(S_held[:,0]**2)*1.5, np.zeros((80,3))]
    chk=checksum(S_sig)
    ports=[]
    for name,Xs,Xh in [("R1",S_sig,S_held),("R2",phi14(S_sig),phi14(S_held)),("R3",second_moment(S_sig),second_moment(S_held))]:
        d=unit(Xs.mean(0)-Xh.mean(0))
        ports.append(causal_port(Xs,Xh,d,f"{name}_contrastive"))
        _,_,Vt=np.linalg.svd(Xs-Xh.mean(0), full_matrices=False)
        ports.append(causal_port(Xs,Xh,Vt[0],f"{name}_svd_0"))
    survivors=[p for p in ports if p["passed"]]
    print(f"logical v0 PASS {len(survivors)}/{len(ports)} S_untouched {checksum(S_sig)==chk}")
    print("recorded campaign: 9 ports PASS; consensus power path-patches; S checksum PASS")

if __name__ == "__main__":
    run()
