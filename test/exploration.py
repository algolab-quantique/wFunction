#%%
import numpy as np
import quimb.tensor as qtn
from wFunction import MPO_compress as mc
#%%
P = qtn.Tensor([[1,0],[0,0]],inds = ('x0','y0'))

X = qtn.rand_tensor((2,2,2,2,2,2),inds = ('x0','x1','x2','y0','y1','y2'))
#%%
u,d,v = X.split(left_inds=['x0','x1','x2'],cutoff=0,absorb=None)
U = u@v #unitary for test
print(U.data)

PU = P@U
u,d,v = PU.split(left_inds=['x1','x2'],cutoff=0,absorb=None)
print(d.data)


# %%
print(PU.data.reshape(4,4))
# %%
inL=4
MPO = qtn.MPO_rand(inL,4,2)
P = qtn.Tensor(data = [[1,0],[0,0]],inds = (MPO.lower_ind_id.format(inL),MPO.upper_ind_id.format(inL)),tags=[MPO.site_tag_id.format(inL)])
#%%
Nm1shape = MPO[-1].data.shape
Nm1shape = (Nm1shape[0],1,*Nm1shape[1:])
MPOp = qtn.MatrixProductOperator([ *[MPO[i].data for i,t in enumerate(MPO.tensors[0:-1])], MPO[-1].data.reshape(Nm1shape), P.data.reshape(1,*P.data.shape) ])
# %%


#can we arbitrarily exchange a pair of matching covariant and contravariant indices and maintain unitarity?
Uc = U.transpose('x0','x1','y2','y0','y1','x2')
print(np.trace(Uc.data.reshape(8,8)@Uc.data.reshape(8,8).T))
print(Uc@Uc)
# %%
c = []
t = qtn.rand_tensor((2,2,2,2),['a','b','c','d'])
u,d,v = t.split(left_inds = ['a','b'],absorb=None,cutoff = 0.0)
t = u@v
u,v = t.split(left_inds=['a','c'])
u.transpose_(u.inds[-1],*u.inds[0:-1])
c.append(u)
c.append(v)
t = qtn.rand_tensor((2,2,2,2),['d','e','f','g'])
u,d,v = t.split(left_inds = ['d','e'],absorb=None,cutoff = 0.0)
t = u@v
u,v = t.split(left_inds = ['d','f'])
c[-1] = c[-1]@u
c[-1].transpose_(c[-1].inds[0],c[-1].inds[-1],*c[-1].inds[1:-1])
c.append(v)
c = qtn.MatrixProductOperator([t.data for t in c])
print(c)
c = mc.MPO_compressor(c,1e-15,1e-14)
#%% vérification des propriété de C
print('All c norms',c[0]@c[0],c[1]@c[1],c[2]@c[2])
mc.move_oc(c,-1)
print('All c norms',c[0]@c[0],c[1]@c[1],c[2]@c[2])
mc.move_oc(c,-1)#does nothing.
print('All c norms',c[0]@c[0],c[1]@c[1],c[2]@c[2])
print("CC",c@c.H)
Cd = c.H.partial_transpose(range(c.L))
D = c.apply(Cd)
print(D.contract().transpose(*['k0','k1','k2','b0','b1','b2']).data.reshape(8,8)) #Si c'est l'identité, on est OK
D = mc.MPO_compressor(D,1e-15,1e-14)
print(D.contract().transpose(*['k0','k1','k2','b0','b1','b2']).data.reshape(8,8)) #Si c'est l'identité, on est OK
# %%
N = c[-1]|c[-2]
Nd = N.reindex({'b1':'c1','b2':'c2'}).H
#%%
print( (Nd@N).transpose('b1','b2','c1','c2').data.reshape(4,4) )
print(N@N)

# %%


#%% Isometric matrix and projections

r = np.random.rand(8,4)
U,D,V = np.linalg.svd(r)
print(U, V)
print("V@Vd",U@U.T)
P = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
print(P)
VP = (U@P)@U.T
#%%
np.kron([[1,0],[0,1]],[[1,0],[0,0]])

# %%
B = qtn.rand_tensor((2,2,2,2,2,2),inds=['x1','z0','y0','y1','a0','a1'])
Ur,er,Vr = B.split(['x1','y0','y1'],cutoff = 0,absorb=None)
B = Ur@Vr
A = qtn.rand_tensor((2,2,2,2),inds=['x0','x1','y0','y1'])
Ua,ea,Va = A.split(['x0','x1'],cutoff = 0, absorb=None)
A = Ua@Va
C = qtn.rand((2,2,2,2),inds=['x0','z0','a0','a1'])
ep = qtn.tensor()
# %%
X = np.zeros((4,4))
X[:,0:2] = np.random.rand(4,2)
print(X)
print("udv")
u,d,v = np.linalg.svd(X)
print(u,d,v)
# %%
