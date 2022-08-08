
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
qtn.MPO_rand_herm(5,2).calc_current_orthog_center()
cMPO = controled_MPO(f,nqbit+1,precision**2)
print("orthog:", cMPO.calc_current_orthog_center())
print(cMPO)
# print("----------")
#%%
rM = np.random.rand(16)
rM /= np.sum(rM**2)
lg = mqg.stacklayer(-1,rM)
stl = staircaselayer(rM)
lg_stl = mqg.layer_compose(lg,stl)
gates,error = mqg.MPSO2Gates(cMPO,precision,6,layer=stl)
# #%%
# cgates = gates.contract()
# #%%
# # print(gates.contract().data.reshape(8,8) , cMPO.contract().data.reshape(8,8))
# # plt.semilogy(jnp.abs(gates.contract().data - cMPO.contract().data).flatten())
# # plt.show()
# cgt = cgates.transpose('k0','k1','k2','b0','b1','b2')
# cmt = cMPO.contract().transpose_like(cgt)
# plt.semilogy(np.abs(cgt.data.reshape(8*8) - cmt.data.reshape(8*8)))
#%%
# from wFunction.mps2qbitsgates import generate_staircase_operators
# # lg = mqg.layer_compose(generate_staircase_operators)
# Net = lg("in{}","out{}",2,0)
# print(Net)
# Net.draw(iterations=20,initial_layout="kamada_kawai")
# print(Net.contract().data.reshape(8,8))

#%%
register = qs.QuantumRegister(nqbit+1)
circuit = Generate_g_circuit(f,1E-8,1e-8,nqbit+1,(-1,1),register,12)
#%%
circuit.draw("mpl")

# %%
