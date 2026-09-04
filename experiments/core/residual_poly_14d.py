#!/usr/bin/env python3
"""14D pure-quadratic residual expansion (no bias, no centering).
phi = [x0..x3, x0²..x3², 6 products]. Same map across NV / Planck / EEG / RF / modular / transformer.
Dominant residual direction is consistently a second-moment monomial (E², power², Amp²).
"""
from __future__ import annotations
import numpy as np

def phi14(X4):
    X4 = np.asarray(X4, float)
    x0,x1,x2,x3 = X4.T
    return np.stack([x0,x1,x2,x3,x0**2,x1**2,x2**2,x3**2,x0*x1,x0*x2,x0*x3,x1*x2,x1*x3,x2*x3],1)

def unit(v): return v/(np.linalg.norm(v)+1e-12)

def causal_port(X_sig, X_held, d, name="port"):
    d = unit(d); c = X_sig@d; ch=float((X_held@d).mean())
    p0=float(np.mean(c**2))
    p_path=float(np.mean(((X_sig-np.outer(c,d)+ch*d)@d)**2))
    p_zero=float(np.mean(((X_sig-np.outer(c,d))@d)**2))
    r=unit(np.random.default_rng(abs(hash(name))%(2**32)).normal(size=d.shape))
    p_rand=float(np.mean((X_sig@r)**2))
    ok = p_path < 0.35*p0 + 1e-8 and p_zero < 0.10*p0 + 1e-12 and p_rand > 0.30*p0 and p0>1e-10
    return dict(name=name, power=p0, path=p_path, zero=p_zero, rand=p_rand, passed=ok)

def run_14d(X4_sig, X4_held, labels=("x0","x1","x2","x3")):
    Qs, Qh = phi14(X4_sig), phi14(X4_held)
    resid = Qs - Qh.mean(0)
    _,S,Vt = np.linalg.svd(resid, full_matrices=False)
    names = list(labels)+[f"{a}sq" for a in labels]+["p01","p02","p03","p12","p13","p23"]
    ports = [causal_port(Qs,Qh,Vt[0],"svd_0"), causal_port(Qs,Qh,Vt[1] if len(S)>1 else Vt[0],"svd_1")]
    # explicit quadratic axes that historically carry the residual
    for i,name in enumerate(names):
        e=np.zeros(14); e[i]=1.0
        ports.append(causal_port(Qs,Qh,e,name))
    print(f"14D SVD0 var {S[0]/S.sum():.1%} top loadings:")
    order=np.argsort(-np.abs(Vt[0]))[:4]
    for i in order:
        print(f"  {names[i]:12s} {Vt[0][i]:+.3f}")
    npass=sum(p["passed"] for p in ports)
    print(f"ports PASS {npass}/{len(ports)}  svd0 power {ports[0]['power']:.3f} → path {ports[0]['path']:.3f} zero {ports[0]['zero']:.1e}")
    return ports, S, Vt[0]

if __name__ == "__main__":
    rng=np.random.default_rng(0)
    # planted linear + x0² + x0·x1 residual
    held=rng.normal(size=(80,4))*0.2
    sig=held.copy()+np.array([0.8,0.1,0.0,0.0])+np.c_[(held[:,0]**2)*2, held[:,0]*held[:,1], np.zeros((80,2))]
    run_14d(sig, held)
