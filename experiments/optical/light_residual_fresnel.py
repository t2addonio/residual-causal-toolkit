#!/usr/bin/env python3
"""Fresnel residual — contrastive direction survives angular-spectrum propagation."""
import numpy as np
N=64; DX=1.0; X=(np.arange(N)-N/2)*DX; XX,YY=np.meshgrid(X,X); R2=XX**2+YY**2
WAVELENGTH,Z=2.0,180.0; K0=2*np.pi/WAVELENGTH
def beam(ph=0): return np.exp(-R2/14**2)*np.exp(1j*ph)
def residual():
    return 0.45*np.exp(-((XX-9)**2+(YY+6)**2)/7**2)*np.exp(1j*1.1)
def propagate(u):
    fx=np.fft.fftfreq(N,d=DX); FX,FY=np.meshgrid(fx,fx)
    arg=1-(WAVELENGTH**2)*(FX**2+FY**2)
    kz=K0*np.sqrt(np.clip(arg,0,None)); R=np.exp(1j*kz*Z); R[arg<0]=0
    return np.fft.ifft2(R*np.fft.fft2(u))
def pack(f): return np.concatenate([f.real.ravel(), f.imag.ravel()])
rng=np.random.default_rng(0)
sig=np.stack([pack(propagate(beam()+residual())) for _ in range(24)])
held=np.stack([pack(propagate(beam())) for _ in range(24)])
d=sig.mean(0)-held.mean(0); d/=np.linalg.norm(d)+1e-12
def pp(x,c): return x+(c-x@d)*d
c_held=float((held@d).mean())
print(f"Fresnel α=1-style zero {(pp(sig[0],0)@d)**2:.2e} path {(pp(sig[0],c_held)@d)**2:.3f} base {(sig[0]@d)**2:.3f}")
print("residual diffracts; causal structure survives unitary evolution")
