#!/usr/bin/env python3
"""Jones polarization residual — vertical residual on a horizontal beam."""
import numpy as np
N=48; X=np.arange(N)-N/2; XX,YY=np.meshgrid(X,X); R2=XX**2+YY**2
Ex=np.exp(-R2/10**2)
Ey=0.45*np.exp(-((XX-8)**2+(YY+5)**2)/4.2**2)*np.exp(1j*0.85)
def pack(ex,ey): return np.concatenate([ex.real.ravel(),ex.imag.ravel(),ey.real.ravel(),ey.imag.ravel()])
sig=np.stack([pack(Ex,Ey) for _ in range(20)])
held=np.stack([pack(Ex,np.zeros_like(Ex)) for _ in range(20)])
d=sig.mean(0)-held.mean(0); d/=np.linalg.norm(d)+1e-12
# split direction back into Ex/Ey to report Ey fraction
half=d.size//2; ey=d[half:]
print(f"Jones |Ey| fraction of residual dir ~ {np.linalg.norm(ey)/np.linalg.norm(d):.3f}")
print(f"α=1 residual power {((sig[0]-(sig[0]@d)*d)@d)**2:.2e}")
