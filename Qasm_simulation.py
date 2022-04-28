#%%

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
import matplotlib.pyplot as plt
import pandas as pd

from qiskit.circuit import qpy_serialization

with open("normal.qpy",'rb') as fd:
    normal = qpy_serialization.load(fd)[0]

normal.measure_all()
circuit = normal
simulator = QasmSimulator()
compiled_circuit = transpile(normal, simulator)
shots = 1000000
job = simulator.run(compiled_circuit, shots=shots)
result = job.result()
counts = result.get_counts(compiled_circuit)
# print("\nTotal counts are:",counts)
#%%
w = []
c = []
for bin in counts:
    w.append(int("0b{}".format(bin),2))
    c.append(counts[bin]/shots)
#%%
data = pd.DataFrame({'w':w,'c':c})
data = data.sort_values(by = ['w'])
plt.plot(data['w'],data['c'])
# %%
