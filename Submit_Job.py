#%%
import IQ_provider
import qiskit
from qiskit import IBMQ, transpile
import pandas as pd
import numpy as np
from qiskit.circuit import qpy_serialization
from qiskit.providers.jobstatus import JobStatus
import matplotlib.pyplot as plt
from qiskit.providers.aer import QasmSimulator
import seaborn as sb
import os

sb.set_theme()
provider = IQ_provider.finquant2

#%%
shots = 2000
Qcomp = provider.get_backend("ibmq_kolkata")
# Qcomp = provider.get_backend("ibm_washington")
Qcomp_name = Qcomp.name().split('_')[1]
#%%
name = "lognorm"
with open("lognorm_5q.qpy",'rb') as fd:
    circuit = qpy_serialization.load(fd)[0]
nqbit = len(circuit.qubits)
circuit.measure_all()
circuit.data = [*circuit.data[:-nqbit-1],*circuit.data[-nqbit:]]#remove the measurement barrier
circuit.draw('mpl')
#%%
filename = '{}_{}_{}.qpy'.format(Qcomp_name,nqbit,name)
if os.path.isfile(filename): 
    with open(filename, 'rb') as fd:
        compiled_circuit = qpy_serialization.load(fd)[0]
else:
    compiled_circuit = transpile(circuit, Qcomp)
    with open(filename, 'wb') as fd:
        qpy_serialization.dump(compiled_circuit,fd)
#%%
compiled_circuit.draw('mpl',idle_wires=False)
#%%

job = Qcomp.run(compiled_circuit, shots=shots,job_name = filename)
#%%
try:
    job_status = job.status()  # Query the backend server for job status.
    if job_status is JobStatus.RUNNING:
        print("The job is still running")
    if job_status is JobStatus.QUEUED:
        print("The job is in Queue")
    print(job.status())
except Exception as ex:
    print("Something wrong happened!: {}".format(ex))
# %%

results = job.result()
#%%
counts = results.get_counts(compiled_circuit)
#%%
w = []
c = []
for bin in counts:
    w.append(int("0b{}".format(bin[-1::-1]),2))
    c.append(counts[bin]/shots)
#%%
data = pd.DataFrame({'w':w,'c':c})
data = data.sort_values(by = ['w'])
plt.plot(data['w'],np.sqrt(data['c']),label = "tomo")
# plt.legend()
plt.savefig("washington_output.pdf")
plt.show()
#%%

# %%
simulator = QasmSimulator()
#%%
qnormal = transpile(circuit, simulator)
qnormal.draw("mpl")
#%%
qcompiled = transpile(compiled_circuit, simulator)
qcompiled.draw("mpl",idle_wires=False)
#%%
sjob = simulator.run(qnormal, shots=shots)
#%%
sresult = sjob.result()
ncounts = sresult.get_counts(circuit)
nw = []
nc = []
for bin in ncounts:
    nw.append(int("0b{}".format(bin[-1::-1]),2))
    nc.append(ncounts[bin]/shots)
#%%
ndata = pd.DataFrame({'w':nw,'c':nc})
ndata = ndata.sort_values(by = ['w'])
plt.plot(ndata['w'],np.sqrt(ndata['c']),label = "simulation_original")
# plt.plot(data['w'],np.sqrt(data['c']),label = "tomo")
plt.legend()
plt.show()
# %%
