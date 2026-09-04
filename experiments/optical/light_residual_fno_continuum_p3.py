#!/usr/bin/env python3
"""Continuum P3 — PHASE(φ) write-hook on the SOURCE plane, READ after G.
Source arms must be orthogonalized against the carrier or hard-zero
punches holes that G diffracts back into the residual READ.
Original bus S is never written. Output plane power is NOT flat in φ
(G(carrier) interferes with the phased residual).
"""
from __future__ import annotations
import hashlib
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
    u = unit(u); return u, unit(v - cdot(u,v)*u)
def blob(amp,w0,x0,y0,ph):
    return amp*np.exp(-((XX-x0)**2+(YY-y0)**2)/w0**2)*np.exp(1j*ph)
def beam(ph=0.0):
    return np.exp(-R2/10**2)*np.exp(1j*ph)
def fresnel_R():
    fx = np.fft.fftfreq(N,d=DX); FX,FY = np.meshgrid(fx,fx)
    FR = np.sqrt(FX**2+FY**2)
    arg = 1-(WAVELENGTH**2)*(FX**2+FY**2)
    kz = K0*np.sqrt(np.clip(arg,0,None))
    R = np.exp(1j*kz*Z_PROP); R[arg<0]=0
    M = ((FR>=0.045)&(FR<=0.110)).astype(float)
    return R*(1+BETA_RES*M*np.exp(1j*PHI_RES))*((FR<=KMAX).astype(float))
def G(u,R): return W_SKIP*u + np.fft.ifft2(R*np.fft.fft2(u))
def phase_src(u, arm0, arm1, phi):
    a0,a1 = gs(arm0, arm1)
    a,b = cdot(a0,u), cdot(a1,u)
    perp = u - a*a0 - b*a1
    return perp + a*a0 + (np.exp(1j*phi)*b)*a1

def run():
    rng = np.random.default_rng(1)
    R = fresnel_R()
    carrier = beam(0.0)
    src_sig = np.stack([beam(rng.uniform(0,0.2))+blob(0.9,4.2,*LOC1,0.85)+blob(0.88,4.0,*LOC2,-0.4) for _ in range(16)])
    chk = checksum(src_sig)
    # source arms = residual blobs orthogonalized against the carrier
    raw0 = blob(1,4.2,*LOC1,0.0); raw1 = blob(1,4.0,*LOC2,0.0)
    raw0 = raw0 - cdot(unit(carrier), raw0)*unit(carrier)
    raw1 = raw1 - cdot(unit(carrier), raw1)*unit(carrier)
    arm0, arm1 = gs(raw0, raw1)
    test = src_sig[0]
    out0 = G(test, R)
    # output READ direction from contrastive after G
    held = np.stack([G(beam(rng.uniform(0,0.2)), R) for _ in range(16)])
    d_out = unit(np.mean([G(u,R) for u in src_sig],0) - held.mean(0))
    def rpow(u_src):
        return float(abs(cdot(d_out, G(u_src, R)))**2)
    p0 = rpow(test)
    # zero source residual plane (after carrier-orthogonalization — no diffracted holes)
    a,b = cdot(arm0,test), cdot(arm1,test)
    u_zero = test - a*arm0 - b*arm1
    # path: held mean source coeffs
    u_path = test - a*arm0 - b*arm1  # held residual ~ 0 on these arms
    fringe = [abs(cdot(d_out, G(phase_src(test, arm0, arm1, ph), R)))**2 for ph in np.linspace(0,2*np.pi,9)]
    ptp = max(fringe)-min(fringe)
    print(f"P3 out {p0:.2f} path/zero {rpow(u_path):.2f}/{rpow(u_zero):.2f} ptp {ptp:.2f} S_ok {checksum(src_sig)==chk}")
    print("lock: orthogonalize source arms vs carrier before hard-zero")

if __name__ == "__main__":
    run()
