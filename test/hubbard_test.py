#%%
import quimb
import quimb.tensor as qtn
import wFunction.MPO_compress as MPOC
import numpy as np
from wFunction.MPO_compress import MPO_compressing_sum,MPO_compressor

MPO = qtn.MatrixProductOperator
MPS = qtn.MatrixProductState
#%%
c_up = np.zeros((4,4))
c_up[0,1] = 1
c_up[2,3] = 1
c_dn = np.zeros((4,4))
c_dn[0,2] = 1
c_dn[1,3] = -1
F = np.eye(4)
F[1,1] = -1
F[2,2] = -1
id = np.eye(4)
n_up = c_up.T@c_up
n_dn = c_dn.T@c_dn

def MPO_C(local_c,F,s,N):
    tens = []
    assert(F.shape == local_c.shape)
    i = 0
    while i<s:
        tens.append(np.array([[F]]))
        i+=1
    tens.append(np.array([[local_c]]))
    i+=1
    while i < N:
        tens.append(np.array([[np.eye(F.shape[0])]]))
        i+=1
    tens[0] = tens[0][:,0,:,:]
    tens[-1] = tens[-1][0,:,:,:]
    return MPO(tens)
    

def Anderson_star(el,tl,U,mu,er,tr):
    tensors = []
    for e,t in zip(el,tl):
        tens = np.zeros((6,6,4,4))
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
    tens = np.zeros((6,6,4,4))
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
    tensors.append(tens)
    for e,t in zip(er,tr):
        tens = np.zeros((6,6,4,4))
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
    tensors[0] = tensors[0][:,5,:,:]
    tensors[-1] = tensors[-1][0,:,:,:]
    return qtn.MatrixProductOperator(tensors)

#%%
el = [-1,-0.5,-.2,0]
tl = [0.1,0.1,.1,0.1]
er = [1,0.5,.2][-1::-1]
tr = [0.1,0.1,.1]
U=8
mu=4
H = Anderson_star(el,tl,U,mu,er,tr)
# H = MPO_compressor(H,1e-15,5e-16) #essentially lossless compression
H.site_tag_id = 'I{}' #quimb MPO methods are supper brittle regarding de site_tag
#%%
star_dmrg = qtn.DMRG(H,200,1e-12)
star_dmrg.solve(tol = 1e-8,max_sweeps=100)
psi = star_dmrg.state
# %%

def Liouvillian(H:MPO,O:MPO,eps:float):
    H.site_tag_id = O.site_tag_id
    HO = O.apply(H)
    OH = -1*H.apply(O)
    OH.upper_ind_id = HO.upper_ind_id
    OH.lower_ind_id = HO.lower_ind_id
    return MPO_compressing_sum([HO,OH],eps,eps*0.5)

def inner_product(A:MPO,B:MPO,ket:MPS):
    """The Liouvillian inner product for the computation
    of Fermionic Green's function: <{A,B^\dagger}>_{psi}"""
    BH = B.H
    bra = ket.H
    A = A.copy()#so we don't modify the input args.
    bra.site_ind_id = 'z{}'
    A.upper_ind_id = 'x{}'
    A.lower_ind_id = ket.site_ind_id
    BH.upper_ind_id = A.upper_ind_id
    BH.lower_ind_id = bra.site_ind_id
    BA = (bra|BH|A|ket).contract()
    BH.upper_ind_id = ket.site_ind_id
    BH.lower_ind_id = 'y{}'
    A.lower_ind_id = 'y{}'
    A.upper_ind_id = bra.site_ind_id
    AB = (bra|A|BH|ket).contract()
    return AB+BA


    

def Liouville_Lanczos(H:MPO,O_0:MPO,psi:MPS,N:int,eps:float):
    O_i = O_0.copy()
    b_i = np.sqrt(inner_product(O_i,O_i,psi))
    O_i /= b_i
    O_ip = Liouvillian(H,O_i,eps)
    a_i = inner_product(O_ip,O_i,psi)
    O_ip = MPO_compressing_sum([O_ip,-1*a_i*O_i],eps,eps*0.5)
    b = []
    a = []
    a.append(a_i)
    b.append(b_i)
    b_ip = np.sqrt(inner_product(O_ip,O_ip,psi))
    b.append(b_ip)
    O_ip /= b_ip
    b_i = b_ip
    O_i,O_im = O_ip,O_i
    for i in range(1,N):
        O_ip = Liouvillian(H,O_i,eps)
        a_i = inner_product(O_ip,O_i,psi)
        a.append(a_i)
        O_ip = MPO_compressing_sum([O_ip,-1*a_i*O_i,-b_i*O_im],eps,eps*0.5)
        b_ip = np.sqrt(inner_product(O_ip,O_ip,psi))
        if abs(b_ip) <  1e-14:
            break
        O_ip /= b_ip
        b.append(b_ip)
        b_i = b_ip
        O_im,O_i = O_i,O_ip
    return a,b[0:-1]

# %%
C_0 = MPO_C(c_up,F,len(el),H.L)
#%%
a,b = Liouville_Lanczos(H,C_0,psi,5,1e-12)
# %%
