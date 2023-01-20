
#%%
from importlib import reload
from wFunction import mpo2qbitsgates as mqg
from wFunction import MPO_compress
from wFunction import mps2qbitsgates
from wFunction.Chebyshev import controled_MPO,func2MPO,make_MPO,test_diagonnal
from wFunction.generate_MPO import cMPO
from wFunction.MPO_compress import MPO_compressing_sum,Square_Root,MPO_compressor
from wFunction import Generate_g_circuit
import numpy as np
import qiskit as qs
from qiskit.circuit import qpy_serialization
import quimb.tensor as qtn

from wFunction import generate_simple_MPO as gsm
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import jax.numpy as jnp
from wFunction.mps2qbitsgates import staircaselayer
Chebyshev = np.polynomial.Chebyshev
sb.set_theme()
# %%

nqbit = 5
precision = 0.000001
osqrtpi = 1/np.sqrt(np.pi)
def one(x):
    return x/1.1
def f(x):
    return np.exp(-(x)**2)*osqrtpi
def g(x):
    return np.sqrt(1-f(x)**2)

# print("----------")
# MPO_f = make_MPO(f,nqbit,precision/2) <
# print("----------")
# MPO_g = make_MPO(g,nqbit,precision/2) 
# print("----------")
MPO = func2MPO(one,nqbit,precision)
ID = gsm.generate_id_MPO(MPO.upper_ind_id,MPO.lower_ind_id,MPO.L) 
IDm2 = ID - MPO.apply(MPO)
IDm2 = MPO_compressor(IDm2,1e-12,1e-13)
print("complement start")
Complement_MPO = Square_Root(IDm2,1e-9,1e-5)
print("complement computed")
# print("complement start")
# Complement_MPO = Square_Root(IDm2,1e-9,1e-9,out=Complement_MPO)
# print("complement computed")
ctrlMPO = cMPO(Complement_MPO,MPO,1e-12)
#%%
order = [*['b{}'.format(i) for i in range(nqbit)],*['k{}'.format(i) for i in range(nqbit)]]
C = Complement_MPO.contract()
# M = MPO.contract()
C.transpose_(*order)
c = C.data.reshape(2**nqbit,2**nqbit)
# M.transpose_(*order)
# m = M.data.reshape(2**nqbit,2**nqbit)
plt.plot(np.diag(c),label='Complement MPO')
# # plt.plot(np.diag(c,1),label='zero')
# # plt.plot(np.diag(m),label='f')
# plt.plot(np.sqrt(1-np.diag(m)**2),label='c')
# # c_corr = c/c[0,0]*np.sqrt(1-m[0,0]**2)
# # plt.plot(np.diag(c_corr),label='corrected c')
# plt.legend()
# plt.show()
# cMPO = controled_MPO(f,nqbit+1,precision**2)
# print("orthog:", cMPO.calc_current_orthog_center())
# print(cMPO)
# print("----------")
#%%
# rM = np.random.rand(16)
# rM /= np.sum(rM**2)
# lg = mqg.stacklayer(-1,rM)
# stl = staircaselayer(data=np.eye(4))
# lg_stl = mqg.layer_compose(lg,stl)
# gates,error = mqg.MPSO2Gates(cMPO,precision,1,layer=lg)
# # gates, error = mqg.Proj_MPSO2Gates(MPO,precision,10,layer = lg)
# # iter_gate,mpo = mqg.Iterative_MPOembedGates(MPO,precision,maxlayers=10)
# layercount = 0
# mpo = MPO.copy()
# uid = qtn.rand_uuid() + '{}'
# layer_link_id = qtn.rand_uuid() + '{}'
# accuhigh = uid.format(layercount)+'h{}'
# acculow = uid.format(layercount)+'l{}'
# accumulator = mqg.generate_id_MPO(accuhigh ,acculow,mpo.L+1,1)
# circuit = qtn.TensorNetwork([])
# layer_gen = staircaselayer(uuid=layer_link_id,data = np.eye(4,4))
# accumulator = mqg.generate_id_MPO(accuhigh ,acculow,mpo.L+1,1)
# layer_gen = staircaselayer(uuid=layer_link_id,data = np.eye(4,4))
# left_layer = layer_gen(acculow,mpo.upper_ind_id,mpo.L,0)
# right_layer = layer_gen(accuhigh,mpo.lower_ind_id,mpo.L,0)
# X = left_layer|accumulator|right_layer
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
# stl.data = np.eye(4).reshape(2,2,2,2)
# vlayer = mqg.V_layer(stl.data)
# intind = uid.format(0)+'{}'
# accumulator = mqg.generate_id_MPO(intind,MPO.lower_ind_id,MPO.L+1)#is that right?
# layer0 = vlayer(intind,MPO.upper_ind_id,MPO.L)
# # (layer0|accumulator).draw()
# # circuit.draw("mpl")
mqg = reload(mqg)
circuit, mpoop = mqg.Iterative_MPOGatesA(ctrlMPO,1e-6,20,layer_gen=staircaselayer)
proj = qtn.Tensor(data = [[1,0],[0,0]],inds = ('k{}'.format(nqbit),'b{}'.format(nqbit)))
proj1 = qtn.Tensor(data = [[0,1],[0,0]],inds = ('k{}'.format(nqbit),'b{}'.format(nqbit)))
cgates = (circuit|proj).contract()
cgates1 = (circuit|proj1).contract()
cgt = cgates.transpose(*[MPO.lower_ind_id.format(i) for i in range(nqbit)],*[MPO.upper_ind_id.format(i) for i in range(nqbit)])
cgt1 = cgates1.transpose(*[MPO.lower_ind_id.format(i) for i in range(nqbit)],*[MPO.upper_ind_id.format(i) for i in range(nqbit)])
cmt = MPO.contract().transpose_like(cgt)
T = circuit.contract().transpose(*[MPO.lower_ind_id.format(i) for i in range(nqbit+1)],*[MPO.upper_ind_id.format(i) for i in range(nqbit+1)]).data.reshape(64,64)
print("is unitary!?:", np.allclose(T@T.T,np.eye(64)))
plt.semilogy(np.abs(cgt.data.flatten() - cmt.data.flatten()))
plt.show()
print(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit))))
plt.plot(np.diagonal(cmt.data.reshape(2**(nqbit),2**(nqbit))))
plt.plot(np.diagonal(cmt.data.reshape(2**(nqbit),2**(nqbit)),offset=1))
plt.plot(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit)),offset=1),label = "gates")
plt.plot(np.diagonal(cgt.data.reshape(2**(nqbit),2**(nqbit))),label="gates on d")
plt.plot(np.diagonal(cgt1.data.reshape(2**(nqbit),2**(nqbit))),label = "complement")
# print(cgt.data.reshape(2**nqbit,2**nqbit))
plt.legend()
plt.show()
 # %%
# uid = qtn.rand_uuid() + '{}'
# layer_link_id = qtn.rand_uuid() + '{}'
# intind = uid.format(0)+'{}'
# accumulator = gsm.generate_id_MPO(intind,MPO.lower_ind_id,MPO.L+1)#is that right?
# # accumulator = mqg.generate_swap_MPO(intind,MPO.lower_ind_id,MPO.L+1)#is that right?
# cacc = accumulator.contract().transpose(*[accumulator.lower_ind_id.format(i) for i in range(MPO.L+1)],*[accumulator.upper_ind_id.format(i) for i in range(MPO.L+1)]).data.reshape(64,64)
# print("is swap unitary!?:", np.allclose(cacc@cacc.T,np.eye(64)))
# stl.data = np.eye(4).reshape(2,2,2,2)
# vlayer = mqg.V_layer(stl.data)
# # layer0 = stl( intind,MPO.upper_ind_id,MPO.L, 0)
# layer0,left_inds,right_inds = vlayer( intind,MPO.upper_ind_id,MPO.L)
# # print(layer0)
# # layer0.draw(iterations=100, initial_layout='spiral')
# # layer0.draw()
# T = ((layer0|accumulator).contract())
# T.transpose_(*['b{}'.format(i) for i in range(MPO.L+1)],*['k{}'.format(i) for i in range(MPO.L+1)])
# print("initial working ansatz",np.allclose(T.data.reshape(2**(nqbit+1),2**(nqbit+1)),np.eye(2**(nqbit+1)) ))
# layer = mqg.layer_SVD_optimizer(layer0,left_inds, accumulator,ctrlMPO,'L{}',prec = 1e-4,max_it = 20)
# projector = qtn.Tensor([[1,0],[0,0]], inds=[MPO.lower_ind_id.format(nqbit),MPO.upper_ind_id.format(nqbit)])
# T = (accumulator|layer).contract().transpose(*['b{}'.format(i) for i in range(MPO.L+1)],*['k{}'.format(i) for i in range(MPO.L+1)]).data.reshape(64,64)
# print("is unitary!?:", np.allclose(T@T.T,np.eye(64)))
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
# #%%
# for t in layer.tensors_sorted():
#     print(t.tags)
#     print(t.data.reshape(4,4))


    



# %%
