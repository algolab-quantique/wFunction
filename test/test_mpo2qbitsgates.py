
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
layercount = 0
mpo = MPO.copy()
uid = qtn.rand_uuid() + '{}'
layer_link_id = qtn.rand_uuid() + '{}'
accuhigh = uid.format(layercount)+'h{}'
acculow = uid.format(layercount)+'l{}'
accumulator = mqg.generate_id_MPO(accuhigh ,acculow,mpo.L+1,1)
circuit = qtn.TensorNetwork([])
layer_gen = staircaselayer(uuid=layer_link_id,data = np.eye(4,4))
accumulator = mqg.generate_id_MPO(accuhigh ,acculow,mpo.L+1,1)
layer_gen = staircaselayer(uuid=layer_link_id,data = np.eye(4,4))
left_layer = layer_gen(acculow,mpo.upper_ind_id,mpo.L,0)
right_layer = layer_gen(accuhigh,mpo.lower_ind_id,mpo.L,0)
X = left_layer|accumulator|right_layer
# X.draw()

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
stl.data = np.eye(4).reshape(2,2,2,2)
vlayer = mqg.V_layer(stl.data)
intind = uid.format(0)+'{}'
accumulator = mqg.generate_id_MPO(intind,MPO.lower_ind_id,MPO.L+1)#is that right?
layer0 = vlayer(intind,MPO.upper_ind_id,MPO.L)
(layer0|accumulator).draw()
# circuit.draw("mpl")
circuit, mpoop = mqg.Iterative_MPOembedGatesA(MPO,1e-6,1)
proj = qtn.Tensor(data = [[1,0],[0,0]],inds = ('k{}'.format(nqbit),'b{}'.format(nqbit)))
cgates = (circuit|proj).contract()
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
# %%
# uid = qtn.rand_uuid() + '{}'
# layer_link_id = qtn.rand_uuid() + '{}'
# intind = uid.format(0)+'{}'
# accumulator = mqg.generate_id_MPO(intind,MPO.lower_ind_id,MPO.L+1)#is that right?
# # accumulator = mqg.generate_swap_MPO(intind,MPO.lower_ind_id,MPO.L+1)#is that right?
# stl.data = np.eye(4).reshape(2,2,2,2)
# vlayer = mqg.V_layer(stl.data)
# # layer0 = stl( intind,MPO.upper_ind_id,MPO.L, 0)
# layer0 = vlayer( intind,MPO.upper_ind_id,MPO.L)
# # print(layer0)
# # layer0.draw(iterations=100, initial_layout='spiral')
# layer = mqg.layer_SVD_optimizer(layer0,accumulator,MPO,'L{}',10,1e-4)
# projector = qtn.Tensor([[1,0],[0,0]], inds=[MPO.lower_ind_id.format(nqbit),MPO.upper_ind_id.format(nqbit)])
# cgates = (accumulator|layer|projector).contract()
# cgt = cgates.transpose(*[MPO.lower_ind_id.format(i) for i in range(nqbit)],*[MPO.upper_ind_id.format(i) for i in range(nqbit)])
# cmt = MPO.contract().transpose_like(cgt)
# plt.semilogy(np.abs(cgt.data.flatten() - cmt.data.flatten()))
# plt.show()
# print(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit))))
# plt.plot(np.diagonal(cmt.data.reshape(2**(nqbit),2**(nqbit))))
# plt.plot(np.diagonal(cmt.data.reshape(2**(nqbit),2**(nqbit)),offset=1))
# plt.plot(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit)),offset=1),label = "gates")
# plt.plot(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit))),label="gates on d")
# # print(cgt.data.reshape(2**nqbit,2**nqbit))
# plt.legend()
# plt.show()
#%%


    


