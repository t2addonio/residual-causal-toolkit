#!/usr/bin/env python3
"""Mueller / Stokes residual — carrier H-pol (I,Q); residual type V (circular)."""
import numpy as np
N=48; X=np.arange(N)-N/2; XX,YY=np.meshgrid(X,X); R2=XX**2+YY**2
def jones_to_stokes(Ex,Ey):
    I=np.abs(Ex)**2+np.abs(Ey)**2; Q=np.abs(Ex)**2-np.abs(Ey)**2
    U=2*np.real(Ex*np.conj(Ey)); V=2*np.imag(Ex*np.conj(Ey))
    return np.stack([I,Q,U,V],0)
def gauss(): return np.exp(-R2/11**2)
def ensemble(n, residual=True, rng=None):
    rng = rng or np.random.default_rng(2); out=[]
    for _ in range(n):
        Ex = gauss()*np.exp(1j*rng.uniform(0,0.1)); Ey=np.zeros_like(Ex)
        if residual:
            amp=(0.5+0.2*rng.random())*np.exp(-((XX-8)**2+(YY+5)**2)/4.2**2)
            Ex = Ex + 0.15*amp; Ey = Ey + 0.15*amp*np.exp(1j*np.pi/2)
        out.append(jones_to_stokes(Ex,Ey))
    return np.stack(out)
def pack(S): return S.ravel()
sig, held = ensemble(40,True), ensemble(40,False)
d = pack(sig.mean(0))-pack(held.mean(0)); d/=np.linalg.norm(d)+1e-12
def rpow(S): return float((pack(S)@d)**2)
test=sig[0]; c_held=float(np.mean([pack(h)@d for h in held]))
S_path = pack(test) + (c_held - pack(test)@d)*d
S_zero = pack(test) - (pack(test)@d)*d
print(f"Stokes-V power {rpow(test):.3f} path {(S_path@d)**2:.3f} zero {(S_zero@d)**2:.2e}")
print("carrier <I>,<Q> relatively stable; residual type V")
