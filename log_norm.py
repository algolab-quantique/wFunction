#%%
import wFunction as wf
import numpy as np
import qiskit as qs
from qiskit.circuit import qpy_serialization

from scipy.stats import lognorm as scplog
def lognorm(x,mu,sigma):
    return scplog.pdf(np.exp(-mu)*x,sigma )
#%%
nqbit = 5
threeqb = qs.QuantumRegister(nqbit)
circ = wf.Generate_f_circuit(lambda x:(lognorm(x,1,1)),1e-5,1e-12,nqbit,[0.00000001,7],threeqb,2,"lognormal")
circ.draw('mpl')
#%%
filename = "lognorm_{}q.qpy".format(nqbit)
with open(filename, 'wb') as fd:
        qpy_serialization.dump(circ,fd)
# %%

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
simulator = QasmSimulator()
#%%
tr_circ = transpile(circ,simulator)

tr_circ.reverse_bits()
tr_circ.draw("mpl")
#%%
tr_circ = transpile(tr_circ.decompose(),simulator,["cx", "id", "rz", "sx", "x"])

#%%
tr_circ.decompose().draw("mpl")
#%%
tr_circ.qasm(True,"3qblognorm11.qasm")

# %%
