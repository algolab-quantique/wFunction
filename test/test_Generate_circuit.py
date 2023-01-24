#%%
from wFunction.mps2qbitsgates import MPS2gates2,trivial_state
from wFunction.Generate_circuit import Generate_unitary_net,reverse_net2circuit,Generate_MPS
import numpy as np
import qiskit as qs
from qiskit_aer.primitives import Sampler
from qiskit import transpile
from qiskit_aer.backends import aer_simulator,aer_compiler,AerSimulator,aerbackend
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import seaborn as sb
#%%
nqbit = 5
mps_precision = 1e-14
domain = [-1,1]
x = np.linspace(-1,1,2**nqbit,endpoint=True)
def f(x):
    return np.exp(-x**2)
fx = f(x)
fx /= np.sqrt(np.sum(fx**2))
#%% Teste le générateur de MPS

mps = Generate_MPS(f,mps_precision,nqbit,domain)
oc = mps.calc_current_orthog_center()[0]
mps[oc]/= np.sqrt(mps[oc].H@mps[oc])#Set the norm to one, freshly computed to correct any norm error in the opimization
test = []
for i in range(2**nqbit):
    ts = trivial_state(nqbit,mps.site_ind_id,i)
    n = ts@mps
    test.append(n)
plt.plot(fx/fx[0]*test[0])
plt.plot(test)

#%% Test le réseau d'unitaire
register = qs.QuantumRegister(nqbit)
net = Generate_unitary_net(f,mps_precision,0.01,5,domain,30,MPS2gates2)
test = []
for i in range(2**nqbit):
    ts = trivial_state(nqbit,'lfq{}',i)
    n = ts@net
    test.append(n.data[0,0,0,0,0])
plt.plot(fx)
plt.plot(test)
plt.show()
plt.semilogy(np.abs(fx-test))

#%%
Circ = reverse_net2circuit(net,'L{}','Op{}',register,'gaussian')
print(net)
Circ.measure_all()
# %%
# %%
Circ.draw('mpl')
#%%
sampler = Sampler()
job = sampler.run(Circ,shots=None)
# %%
result = job.result()
dict_data = result.quasi_dists[0]
x,y = [],[]
for key in dict_data:
    x.append(key)
    y.append(dict_data[key])

plt.scatter(x,y)
# %%
from scipy.stats import lognorm as scplog

def lognorm(x,mu,sigma):
    return scplog.pdf(np.exp(-mu)*x,sigma )
 
def lognorm11(x):
    return lognorm(x,1,1)

domain = [0,7]

#%% Teste le générateur de MPS

mps = Generate_MPS(lognorm11,mps_precision,nqbit,domain)
oc = mps.calc_current_orthog_center()[0]
mps[oc]/= np.sqrt(mps[oc].H@mps[oc])#Set the norm to one, freshly computed to correct any norm error in the opimization
test = []
for i in range(2**nqbit):
    ts = trivial_state(nqbit,mps.site_ind_id,i)
    n = ts@mps
    test.append(n)
x = np.linspace(domain[0],domain[1],2**nqbit,endpoint=True)
fx = lognorm11(x)
plt.plot(fx/fx[19]*test[19])
plt.plot(test)

#%%
register = qs.QuantumRegister(nqbit)
net = Generate_unitary_net(lognorm11,mps_precision,0.01,5,domain,30,MPS2gates2)
test = []
for i in range(2**nqbit):
    ts = trivial_state(nqbit,'lfq{}',i)
    n = ts@net
    test.append(n.data[0,0,0,0,0])
plt.plot(fx/fx[19]*test[19])
plt.plot(test)
plt.show()
plt.semilogy(np.abs(fx/fx[19]*test[19]-test))

#%%
Circ = reverse_net2circuit(net,'L{}','Op{}',register,'gaussian')