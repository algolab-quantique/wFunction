#%%
import quimb
import quimb.tensor as qtn
import wFunction.MPO_compress as MPOC
import numpy as np

c_up = np.zeros(4,4)
c_up[0,1] = 1
c_up[2,3] = 1
c_dn = np.zeros(4,4)
c_up[0,2] = 1
c_up[1,3] = -1
F = np.eye(4)
F[1,1] = -1
F[2,2] = -1
id = np.eye(4)
n_up = c_up.T@c_up
n_dn = c_dn.T@c_dn

def Anderson_star(el,tl,U,mu,er,tr):
    tensors = []
    for e,t in zip(el,tl):
        tens = np.zeros(6,6,4,4)
        tens[0,0,:,:] = id
        tens[1,1,:,:] = F
        tens[2,2,:,:] = F
        tens[3,3,:,:] = F
        tens[4,4,:,:] = F
        tens[5,5,:,:] = id
        tens[0,5,:,:] = e*(n_up+n_dn)
        tens[1,5,:,:] = t*(F@c_up)
        tens[2,5,:,:] = t*(c_up.T@F)
        tens[3,5,:,:] = t*(F@c_dn)
        tens[4,5,:,:] = t*(c_dn.T@F)
        tensors.append(tens)
    tens = np.zeros(6,6,4,4)
    tens[0,0,:,:] = id
    tens[5,5,:,:] = id
    tens[1,5,:,:] = (F@c_up)
    tens[2,5,:,:] = (c_up.T@F)
    tens[3,5,:,:] = (F@c_dn)
    tens[4,5,:,:] = (c_dn.T@F)
    tens[0,1,:,:] = (c_up.T)
    tens[0,2,:,:] = (c_up)
    tens[0,3,:,:] = (c_dn.T)
    tens[0,4,:,:] = (c_dn)
    tens[0,5,:,:] = U*(n_up@n_dn) - mu*(n_up+n_dn)
    for e,t in zip(er,tr):
        tens = np.zeros(6,6,4,4)
        tens[0,0,:,:] = id
        tens[1,1,:,:] = F
        tens[2,2,:,:] = F
        tens[3,3,:,:] = F
        tens[4,4,:,:] = F
        tens[5,5,:,:] = id
        tens[0,5,:,:] = e*(n_up+n_dn)
        tens[0,1,:,:] = t*(c_up.T)
        tens[0,2,:,:] = t*(c_up)
        tens[0,3,:,:] = t*(c_dn.T)
        tens[0,4,:,:] = t*(c_dn)
        tensors.append(tens)
    tens[0] = tens[0][:,5,:,:]
    tens[-1] = tens[-1][0,:,:,:]
    return qtn.MatrixProductOperator(tens)


