#%%

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
import matplotlib.pyplot as plt
import pandas as pd

from qiskit.circuit import qpy_serialization
#%%
with open("lognorm_5q.qpy",'rb') as fd:
    normal = qpy_serialization.load(fd)[0]

circuit2 = normal

normal.save_statevector()
normal.measure_all()
circuit = normal
# print(normal)
simulator = QasmSimulator()
compiled_circuit = transpile(normal, simulator)
shots = 2000
job = simulator.run(compiled_circuit, shots=shots)
# print(compiled_circuit.decompose())
result = job.result()
counts = result.get_counts(compiled_circuit)
SV = result.get_statevector()
nqbit = int(np.log2(len(SV.data)))
fmt_str = '0'+str(nqbit) + 'b'
W_sv = [int(format(i,fmt_str)[-1::-1],2) for i in range(2**nqbit)]
# print(result['statevector'])
# print("\nTotal counts are:",counts)
#%%
w = []
c = []
for bin in counts:
    w.append(int(bin[-1::-1],2))
    c.append(counts[bin]/shots)
#%%
sv_data = pd.DataFrame({'w':W_sv,'SV':SV.data})
sv_data = sv_data.sort_values(by = ['w'])
data = pd.DataFrame({'w':w,'c':c})
data = data.sort_values(by = ['w'])
plt.plot(data['w'],np.sqrt(data['c']),label = "tomo")
plt.plot(sv_data['w'],sv_data['SV'],label = "SV")
plt.legend()
plt.savefig("output.pdf")
plt.show()
# %%

circuit2.decompose().draw("mpl")
#%%
c = qiskit.ClassicalRegister(5)
#%%
c0 = c[0]
print(c0._repr )
# %%
