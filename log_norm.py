#%%
import wFunction as wf
import numpy as np
import qiskit as qs

from scipy.stats import lognorm as scplog
def lognorm(x,mu,sigma):
    return scplog.pdf(np.exp(-mu)*x,sigma )
#%%
threeqb = qs.QuantumRegister(3)
circ = wf.gen_circuit(lambda x:(lognorm(x,1,1)),1e-5,1e-12,3,[0.00000001,7],threeqb,1,"lognormal")
circ.draw('mpl')
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
