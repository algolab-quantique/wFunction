
#%%
from wFunction import mpo2qbitsgates as mqg
from wFunction.Chebyshev import controled_MPO,func2MPO,make_MPO,test_diagonnal
from wFunction.MPO_compress import MPO_compressing_sum
from wFunction import Generate_g_circuit
import numpy as np
import qiskit as qs
from qiskit.circuit import qpy_serialization
import quimb.tensor as qtn

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import jax.numpy as jnp
from wFunction.mps2qbitsgates import staircaselayer
sb.set_theme()
# %%

nqbit = 5
precision = 0.000001
osqrtpi = 1/np.sqrt(np.pi)
def f(x):
    return np.exp(-(x)**2)*osqrtpi
def g(x):
    return np.sqrt(1-f(x)**2)

# print("----------")
# MPO_f = make_MPO(f,nqbit,precision/2) 
# print("----------")
# MPO_g = make_MPO(g,nqbit,precision/2) 
# print("----------")
MPO = func2MPO(f,nqbit,precision)
# cMPO = controled_MPO(f,nqbit+1,precision**2)
# print("orthog:", cMPO.calc_current_orthog_center())
# print(cMPO)
# print("----------")
#%%
rM = np.random.rand(16)
rM /= np.sum(rM**2)
lg = mqg.stacklayer(-1,rM)
stl = staircaselayer(data=np.eye(4))
lg_stl = mqg.layer_compose(lg,stl)
# gates,error = mqg.MPSO2Gates(cMPO,precision,1,layer=lg)
# gates, error = mqg.Proj_MPSO2Gates(MPO,precision,10,layer = lg)
# iter_gate,mpo = mqg.Iterative_MPOembedGates(MPO,precision,maxlayers=10)
# print(circuit)
# #%% for non iterative method
# cgates = gates.contract()[]
# cgt = cgates.transpose(*['k{}'.format(i) for i in range(nqbit+1)],*['b{}'.format(i) for i in range(nqbit+1)])
# cmt = cMPO.contract().transpose_like(cgt)
# plt.semilogy(np.abs(cgt.data.flatten() - cmt.data.flatten()))
# plt.show()
# plt.plot(np.diagonal(cmt.data.reshape(2**(nqbit+1),2**(nqbit+1))))
# plt.plot(np.diagonal(cgt.data.reshape(2**(nqbit+1),2**(nqbit+1))))
# plt.plot(np.diagonal(cmt.data.reshape(2**(nqbit+1),2**(nqbit+1)),offset=1))
# plt.plot(np.diagonal(cgt.data.reshape(2**(nqbit+1),2**(nqbit+1)),offset=1))
# plt.show()
# #%% for iterative method
# projector = qtn.Tensor([[1,0],[0,0]], inds=['b{}'.format(nqbit),'k{}'.format(nqbit)])
# cgates = (iter_gate|projector).contract()
# cgt = cgates.transpose(*['k{}'.format(i) for i in range(nqbit)],*['b{}'.format(i) for i in range(nqbit)])
# cmt = MPO.contract().transpose_like(cgt)
# plt.semilogy(np.abs(cgt.data.flatten() - cmt.data.flatten()))
# plt.show()
# plt.plot(np.diagonal(cmt.data.reshape(2**(nqbit),2**(nqbit))))
# plt.plot(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit))))
# plt.plot(np.diagonal(cmt.data.reshape(2**(nqbit),2**(nqbit)),offset=1))
# plt.plot(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit)),offset=1))
# plt.show()
#%%
#%%
# from wFunction.mps2qbitsgates import generate_staircase_operators
# # lg = mqg.layer_compose(generate_staircase_operators)
# Net = lg("in{}","out{}",2,0)
# print(Net)
# Net.draw(iterations=20,initial_layout="kamada_kawai")
# print(Net.contract().data.reshape(8,8))

#%%
# register = qs.QuantumRegister(nqbit+1)
# circuit = Generate_g_circuit(f,1E-8,1e-8,nqbit+1,(-1,1),register,12)
# #%%
# circuit.draw("mpl")

# %%
uid = qtn.rand_uuid() + '{}'
layer_link_id = qtn.rand_uuid() + '{}'
intind = uid.format(0)+'{}'
accumulator = mqg.generate_swap_MPO(intind,MPO.lower_ind_id,MPO.L+1)#is that right?
stl.data = np.eye(4).reshape(2,2,2,2)
layer0 = stl( intind,MPO.upper_ind_id,MPO.L, 0)
layer = mqg.layer_SVD_optimizer(layer0,accumulator,MPO,'L{}',10,1e-4)
projector = qtn.Tensor([[1,0],[0,0]], inds=[MPO.lower_ind_id.format(nqbit),MPO.upper_ind_id.format(nqbit)])
cgates = (accumulator|layer|projector).contract()
cgt = cgates.transpose(*[MPO.lower_ind_id.format(i) for i in range(nqbit)],*[MPO.upper_ind_id.format(i) for i in range(nqbit)])
cmt = MPO.contract().transpose_like(cgt)
plt.semilogy(np.abs(cgt.data.flatten() - cmt.data.flatten()))
plt.show()
print(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit))))
plt.plot(np.diagonal(cmt.data.reshape(2**(nqbit),2**(nqbit))))
plt.plot(np.diagonal(cmt.data.reshape(2**(nqbit),2**(nqbit)),offset=1))
plt.plot(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit)),offset=1),label = "gates")
plt.plot(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit))),label="gates on d")
# print(cgt.data.reshape(2**nqbit,2**nqbit))
plt.legend()
plt.show()
