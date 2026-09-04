#!/usr/bin/env python3
"""Audio residual demo — recover a known 1250 Hz feedback tone from a 220 Hz stack."""
import numpy as np
from scipy.signal import stft
sr=16000; t=np.arange(int(sr*2.0))/sr
def tone(f,amp=0.3): return amp*np.sin(2*np.pi*f*t)
carrier=tone(220,0.45)+tone(440,0.22)+tone(660,0.12)+tone(880,0.07)
residual=tone(1250,0.18)
def mag(x):
    f,_,Z=stft(x,fs=sr,nperseg=1024,noverlap=768); return f, np.abs(Z).mean(1)
f,s=mag(carrier+residual); _,h=mag(carrier)
d=s-h; d/=np.linalg.norm(d)+1e-12
k=int(np.argmax(d)); print(f"audio residual peak {f[k]:.1f} Hz (planted 1250)")
print(f"path/zero protocol applies in STFT-mag space the same as RF-P1")
