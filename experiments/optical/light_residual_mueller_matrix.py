#!/usr/bin/env python3
"""Residual Mueller matrix — contrastive ΔM of a weak diattenuator; ablate M→I."""
import numpy as np

def residual_diattenuator(d=0.35, theta_deg=45.0):
    th=np.deg2rad(theta_deg); c2,s2=np.cos(2*th),np.sin(2*th); a=d
    M=np.eye(4)
    M[0,1]=M[1,0]=a*c2; M[0,2]=M[2,0]=a*s2
    s=np.sqrt(max(1-a**2,0))
    M[1,1]=c2**2+s2**2*s; M[2,2]=s2**2+c2**2*s
    M[1,2]=M[2,1]=c2*s2*(1-s)
    return M

def random_stokes(n, rng):
    S=np.zeros((n,4)); S[:,0]=1.0
    S[:,1]=rng.uniform(-0.7,0.7,n); S[:,2]=rng.uniform(-0.7,0.7,n); S[:,3]=rng.uniform(-0.4,0.4,n)
    nrm=np.sqrt(S[:,1]**2+S[:,2]**2+S[:,3]**2); cap=0.85
    scale=np.minimum(1.0, cap/(nrm+1e-12)); S[:,1:]*=scale[:,None]
    return S

def estimate_M(Sin, Sout):
    M=np.zeros((4,4))
    for i in range(4):
        M[i],_,_,_=np.linalg.lstsq(Sin, Sout[:,i], rcond=None)
    return M

rng=np.random.default_rng(3)
M_res, M_id = residual_diattenuator(), np.eye(4)
Sin=random_stokes(200,rng)
M_hat_sig=estimate_M(Sin, (M_res@Sin.T).T)
M_hat_held=estimate_M(Sin, (M_id@Sin.T).T)
dM = M_hat_sig - M_hat_held
true = M_res-M_id
cos=float(np.vdot(dM.ravel(), true.ravel())/(np.linalg.norm(dM)*np.linalg.norm(true)+1e-12))
# residual Stokes power from V/U induced by ΔM; ablate M→I
S_probe=np.array([1.0,0.6,0.2,0.0])
p_res=float(np.sum(((M_res-M_id)@S_probe)[1:]**2))
p_abl=float(np.sum(((M_id-M_id)@S_probe)[1:]**2))
print(f"ΔM |cos| to true {abs(cos):.4f}  residual Stokes power {p_res:.4f} → ablate {p_abl:.1e}")
print("key residual elements recovered (M01,M10,M02,M20)")
