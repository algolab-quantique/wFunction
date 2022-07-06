
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
sb.set_theme()
# %%

nqbit = 2
precision = 0.00001
osqrtpi = 1/np.sqrt(np.pi)
def f(x):
    return np.exp(-(x)**2)*osqrtpi
def g(x):
    return np.sqrt(1-f(x)**2)

# print("----------")
# MPO_f = make_MPO(f,nqbit,precision/2) 
# print("----------")
# MPO_g = make_MPO(g,nqbit,precision/2) 
print("----------")
cMPO = controled_MPO(f,nqbit+1,precision**2)
print(cMPO)
print("----------")
gates,error = mqg.MPSO2Gates(cMPO,precision,1)

print(gates.tensors[0])
print([gates.tensors[0].data.reshape(8,8)[i,i] for i in range(8)])
print([gates.tensors[0].data.reshape(8,8)[i,i+1] for i in range(7)])
# print(error)
# nqbit = 5
# register = qs.QuantumRegister(nqbit)
# circuit = Generate_g_circuit(f,1E-8,1e-8,nqbit,(-1,1),register,1)
# circuit.draw()

# %%
