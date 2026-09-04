#!/usr/bin/env python3
"""SDR Residual Stream ISA — offline IQ drop-in.
IQ windows are bus S. Never overwrite the source capture.
LOAD → EXTRACT (probe arms, not co-occurrence SVD) → FILTER → PHASE(φ) → READ.
Live TX rewrite is illegal until this offline FILTER passes on files the radio wrote.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path("/home/workdir/artifacts")
CAPTURE_DIR = ROOT / "sdr_capture"
FS, N, N_SIG, N_HELD = 2_048_000.0, 2048, 64, 64
F_CARRIER, F_RES, F_SPUR, NOISE, SEED = 200_000.0, 450_000.0, 50_000.0, 0.018, 20260903

def checksum(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
def unit(v):
    v = np.asarray(v, float).ravel(); return v/(np.linalg.norm(v)+1e-12)
def tone_iq(f, n, fs, amp, phase):
    t = np.arange(n)/fs; return amp*np.exp(1j*(2*np.pi*f*t + phase))
def pack_real(x):
    x = np.asarray(x); out = np.empty(x.size*2); out[0::2]=x.real; out[1::2]=x.imag; return out
def unpack_real(s):
    s = np.asarray(s, float); return s[0::2] + 1j*s[1::2]
def probe_cos_sin(f, n, fs):
    t = np.arange(n)/fs
    u = unit(pack_real(np.exp(1j*2*np.pi*f*t)))
    v = unit(pack_real(1j*np.exp(1j*2*np.pi*f*t)))
    v = unit(v - np.dot(u,v)*u)
    return u, v
def band_power(x, f, fs, half=2):
    X = np.fft.fft(x); k = int(np.round(f/fs*x.size)) % x.size
    sl = np.r_[k-half:k+half+1] % x.size
    return float(np.mean(np.abs(X[sl])**2))
def synthesize(rng):
    sig, held = [], []
    for _ in range(N_SIG):
        sig.append(tone_iq(F_CARRIER,N,FS,1.0,rng.uniform(0,2*np.pi))
                   + tone_iq(F_RES,N,FS,0.42*(0.85+0.30*rng.rand()),rng.uniform(0,2*np.pi))
                   + tone_iq(F_SPUR,N,FS,0.12,rng.uniform(0,2*np.pi))
                   + NOISE*(rng.randn(N)+1j*rng.randn(N)))
    for _ in range(N_HELD):
        held.append(tone_iq(F_CARRIER,N,FS,1.0,rng.uniform(0,2*np.pi))
                    + tone_iq(F_RES,N,FS,0.03*rng.rand(),rng.uniform(0,2*np.pi))
                    + tone_iq(F_SPUR,N,FS,0.12,rng.uniform(0,2*np.pi))
                    + NOISE*(rng.randn(N)+1j*rng.randn(N)))
    return np.stack(sig), np.stack(held)
def try_load():
    if not CAPTURE_DIR.exists(): return None
    sigs = sorted(CAPTURE_DIR.glob("sig_*.npy")); helds = sorted(CAPTURE_DIR.glob("held_*.npy"))
    if len(sigs)>=8 and len(helds)>=8:
        return (np.stack([np.load(p).astype(np.complex128).ravel()[:N] for p in sigs[:64]]),
                np.stack([np.load(p).astype(np.complex128).ravel()[:N] for p in helds[:64]]))
    return None
def reconstruct_one(s,u,v,a,b):
    return s - np.dot(s,u)*u - np.dot(s,v)*v + a*u + b*v
def phase_rotate(a,b,phi):
    c,s = np.cos(phi), np.sin(phi); return c*a-s*b, s*a+c*b

def main():
    rng = np.random.RandomState(SEED)
    loaded = try_load()
    sig_iq, held_iq = loaded if loaded is not None else synthesize(rng)
    source = "real_capture" if loaded is not None else "synthetic_bench"
    S_sig = np.stack([pack_real(x) for x in sig_iq])
    S_held = np.stack([pack_real(x) for x in held_iq])
    chk = checksum(S_sig)
    u, v = probe_cos_sin(F_RES, sig_iq.shape[1], FS)
    def plane_pow(S): return float(np.mean((S@u)**2 + (S@v)**2))
    p0 = plane_pow(S_sig)
    # FILTER: path to held mean coeffs, zero plane, random plane
    ah, bh = float((S_held@u).mean()), float((S_held@v).mean())
    S_path = np.stack([reconstruct_one(s,u,v,ah,bh) for s in S_sig])
    S_zero = np.stack([reconstruct_one(s,u,v,0.0,0.0) for s in S_sig])
    ru, rv = unit(rng.randn(S_sig.shape[1])), None
    rv = unit(rng.randn(S_sig.shape[1])); rv = unit(rv-np.dot(ru,rv)*ru)
    S_rand = np.stack([reconstruct_one(s,ru,rv,ah,bh) for s in S_sig])
    # PHASE sweep on residual probe plane; complement (spur) must survive
    phis = np.linspace(0, 2*np.pi, 9)
    fringe, spur = [], []
    for phi in phis:
        patched = []
        for s in S_sig:
            a,b = float(s@u), float(s@v)
            a2,b2 = phase_rotate(a,b,phi)
            patched.append(unpack_real(reconstruct_one(s,u,v,a2,b2)))
        fringe.append(np.mean([band_power(x, F_RES, FS) for x in patched]))
        spur.append(np.mean([band_power(x, F_SPUR, FS) for x in patched]))
    ptp = max(fringe)-min(fringe)
    print(f"SDR-IQ source={source} plane {p0:.3f} path {plane_pow(S_path):.3f} zero {plane_pow(S_zero):.2e} rand {plane_pow(S_rand):.3f}")
    print(f"PHASE ptp_res {ptp:.3f} spur_flat {np.std(spur):.3e} S_untouched {checksum(S_sig)==chk}")
    print("lock: probe arms at F_RES, not co-occurrence SVD; no live TX rewrite")
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    (CAPTURE_DIR/"CAPTURE.txt").write_text(
        "Offline IQ first. Windows are bus S.\n"
        "Accepted: sig_XX.npy / held_XX.npy or sig.cf32 + held.cf32 + meta.json\n"
        f"Default fs={FS} n={N} f_c={F_CARRIER} f_res={F_RES}\n"
    )

if __name__ == "__main__":
    main()
