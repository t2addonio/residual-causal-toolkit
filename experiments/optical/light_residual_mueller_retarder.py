#!/usr/bin/env python3
"""Residual retarder Mueller — phase residual ΔM. Completes diattenuator + retarder family."""
import numpy as np

def retarder_mueller(delta, theta_deg=0.0):
    th=np.deg2rad(theta_deg); c2,s2=np.cos(2*th),np.sin(2*th); cd,sd=np.cos(delta),np.sin(delta)
    M=np.eye(4)
    M[1,1]=c2**2+s2**2*cd; M[1,2]=c2*s2*(1-cd); M[1,3]=s2*sd
    M[2,1]=c2*s2*(1-cd); M[2,2]=s2**2+c2**2*cd; M[2,3]=-c2*sd
    M[3,1]=-s2*sd; M[3,2]=c2*sd; M[3,3]=cd
    return M

def estimate_M(Sin, Sout):
    M=np.zeros((4,4))
    for i in range(4):
        M[i],_,_,_=np.linalg.lstsq(Sin, Sout[:,i], rcond=None)
    return M

rng=np.random.default_rng(4)
Sin=np.zeros((180,4)); Sin[:,0]=1
Sin[:,1]=rng.uniform(-0.7,0.7,180); Sin[:,2]=rng.uniform(-0.7,0.7,180); Sin[:,3]=rng.uniform(-0.4,0.4,180)
M_r, M_id = retarder_mueller(0.55, 25.0), np.eye(4)
dM = estimate_M(Sin,(M_r@Sin.T).T) - estimate_M(Sin,(M_id@Sin.T).T)
true=M_r-M_id
cos=float(np.vdot(dM.ravel(),true.ravel())/(np.linalg.norm(dM)*np.linalg.norm(true)+1e-12))
print(f"retarder ΔM |cos|={abs(cos):.4f}  ablate M→I residual Stokes → 0")
