#!/usr/bin/env python3
"""Continuum P2 — joint 2-D residual core AFTER G, then PHASE(φ)+BEAM(θ).
Lock: if residuals co-occur, do NOT extract the core from sig-vs-held SVD-1.
Use two single-feature probe arms. S never written.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

N, DX = 64, 1.0
X = (np.arange(N)-N/2)*DX
XX, YY = np.meshgrid(X, X); R2 = XX**2 + YY**2
LOC1, LOC2 = (8.5, -5.5), (-9.0, 7.0)
W_SKIP, BETA_RES, PHI_RES, KMAX = 0.12, 2.20, 0.70, 0.28
WAVELENGTH, Z_PROP = 2.0, 180.0
K0 = 2*np.pi/WAVELENGTH

def checksum(u): return hashlib.sha256(np.ascontiguousarray(u).view(np.uint8)).hexdigest()[:16]
def cdot(a,b): return np.vdot(a.ravel(), b.ravel())
def unit(f): return f/(np.linalg.norm(f)+1e-15)
def gs(u,v):
    u = unit(u); v = unit(v - cdot(u,v)*u); return u, v
def blob(amp,w0,x0,y0,ph):
    return amp*np.exp(-((XX-x0)**2+(YY-y0)**2)/w0**2)*np.exp(1j*ph)
def beam(ph=0.0):
    return np.exp(-R2/10**2)*np.exp(1j*ph)
def fresnel_R():
    fx = np.fft.fftfreq(N, d=DX); FX, FY = np.meshgrid(fx, fx)
    FR = np.sqrt(FX**2+FY**2)
    arg = 1 - (WAVELENGTH**2)*(FX**2+FY**2)
    kz = K0*np.sqrt(np.clip(arg,0,None))
    R = np.exp(1j*kz*Z_PROP); R[arg<0]=0
    M = ((FR>=0.045)&(FR<=0.110)).astype(float)
    keep = (FR<=KMAX).astype(float)
    return R*(1+BETA_RES*M*np.exp(1j*PHI_RES))*keep, M
def G(u, R): return W_SKIP*u + np.fft.ifft2(R*np.fft.fft2(u))
def plane_power(field, u, v):
    return float(abs(cdot(u,field))**2 + abs(cdot(v,field))**2)
def phase_arm(field, u, v, phi):
    u,v = gs(u,v)
    a, b = cdot(u,field), cdot(v,field)
    perp = field - a*u - b*v
    return perp + a*u + (np.exp(1j*phi)*b)*v

def run():
    rng = np.random.default_rng(0)
    R,_ = fresnel_R()
    src_sig = np.stack([beam(rng.uniform(0,0.2)) + blob(0.9,4.2,*LOC1,0.85) + blob(0.88,4.0,*LOC2,-0.4) for _ in range(24)])
    src_held = np.stack([beam(rng.uniform(0,0.2)) for _ in range(24)])
    chk = checksum(src_sig)
    out_sig = np.stack([G(u,R) for u in src_sig])
    out_held = np.stack([G(u,R) for u in src_held])
    # single-feature probe arms at the two planted locations (NOT SVD-1)
    m1 = ((XX-LOC1[0])**2+(YY-LOC1[1])**2) < 16
    m2 = ((XX-LOC2[0])**2+(YY-LOC2[1])**2) < 16
    psi0 = unit(out_sig.mean(0)*m1); psi1 = unit(out_sig.mean(0)*m2)
    psi0, psi1 = gs(psi0, psi1)
    test = out_sig[0]
    p0 = plane_power(test, psi0, psi1)
    # zero plane
    a,b = cdot(psi0,test), cdot(psi1,test)
    zero = test - a*psi0 - b*psi1
    # path to held mean coeffs
    ah = np.mean([cdot(psi0,h) for h in out_held]); bh = np.mean([cdot(psi1,h) for h in out_held])
    path = test - a*psi0 - b*psi1 + ah*psi0 + bh*psi1
    phis = np.linspace(0, 2*np.pi, 9)
    fringe = [abs(cdot(psi0, phase_arm(test, psi0, psi1, ph)))**2 for ph in phis]
    ptp = max(fringe)-min(fringe)
    print(f"P2 plane {p0:.2f} path {plane_power(path,psi0,psi1):.2f} zero {plane_power(zero,psi0,psi1):.2e} ptp {ptp:.2f} S_ok {checksum(src_sig)==chk}")
    print("lock: probe arms, not co-occurrence SVD-1")

if __name__ == "__main__":
    run()
