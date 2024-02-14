"""
 Copyright (c) 2024 Alexandre Foley - Université de Sherbrooke

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <https://www.gnu.org/licenses/>.
 """

#%%
from wFunction.scalarQubitization import (
    QuantumCircuit,
    eval_SU2_func,
    evalUmats_SU2_func,
    exp,
    get_angles,
    get_rotations,
    get_unitary_transform,
    np,
    numbaList,
    phi2rots,
    pi,
    qubitize_scalar,
    W,
    ZeroPiDomain,
)
from matplotlib import pyplot as plt


def gauss(x):
    return np.exp(-(x**2))


def gauss99(x):
    return np.exp(-(x**2)) * 0.99


def gauss95(x):
    return np.exp(-(x**2)) * 0.95


#%%

domain = [-2, 2]
max_layers = 1024
precision = 1e-3
# on the test functions, we only need 10 layers
n_layer = 10
ttheta = np.linspace(
    0, pi, 10 * n_layer
)  # for the test evaluation of the cost function
gop = ZeroPiDomain(gauss, domain)
g95op = ZeroPiDomain(gauss95, domain)
g99op = ZeroPiDomain(gauss99, domain)
#%%
# C = qubitize(gauss,5,domain,max_layers,precision)
# print(get_phi_rotation(gauss,domain,max_layers,precision))
U, doubled = get_unitary_transform(gauss95, domain, max_layers, precision)
phi99 = [*get_angles(gauss99, domain, max_layers, precision)]
rots99 = [*get_rotations(gauss99, domain, max_layers, precision)]
phi95 = [*get_angles(gauss95, domain, max_layers, precision)]
rots95 = [*get_rotations(gauss95, domain, max_layers, precision)]

#%%
err = (np.array(phi2rots(phi95))) - (np.array(rots95))
# plt.plot(np.abs(err[:,0,0]),label='00')
# plt.plot(np.abs(err[:,0,1]),label='01')
# plt.plot(np.abs(err[:,1,0]),label='10')
# plt.plot(np.abs(err[:,1,1]),label='11')
# plt.legend()
print(exp(phi95[-1]))
print(rots95[-1])

#%%
theta = np.linspace(0, np.pi, 200)
f95 = eval_SU2_func(phi95)
f99 = eval_SU2_func(phi99)
flU = evalUmats_SU2_func(U)
ft95 = np.array([f95(tt) for tt in theta])
flUt = np.array([flU(tt) for tt in theta])
ft99 = np.array([f99(tt) for tt in theta])
g99 = g99op(theta)
g95 = g95op(theta)
g = gauss(theta)
plt.plot(theta, ft95[:, 0, 0], label="ft9500")
plt.plot(theta, ft99[:, 0, 0], label="ft9900")
plt.plot(theta, flUt[:, 0, 0], label="flU00")
plt.plot(theta, ft95[:, 1, 1], label="ft9511")
plt.plot(theta, ft99[:, 1, 1], label="ft9911")
plt.plot(theta, g99, label="g99")
plt.plot(theta, g95, label="g95")
plt.legend()
plt.show()
#%%
g95w = g95op(ttheta)
g99w = g99op(ttheta)
gw = gop(ttheta)
f95w = np.array([f95(tt) for tt in ttheta])
f99w = np.array([f99(tt) for tt in ttheta])

print(np.sum(np.abs(g95w - f95w[:, 0, 0]) ** 2))
print(np.sum(np.abs(g99w - f99w[:, 0, 0]) ** 2))
print(np.sum(np.abs(g99w - f95w[:, 0, 0]) ** 2))
print(W(numbaList(phi95), ttheta, g99w))
#%%
nqbits = 5
circ = qubitize_scalar(gauss, nqbits, domain, 256, 1e-2)

from qiskit.quantum_info import SparsePauliOp


def z(n, N):
    s = ""
    for i in range(nqbits):
        if i == n:
            s += "Z"
        else:
            s += "I"
    return s


Obs = [SparsePauliOp.from_list([(z(i, nqbits), 1)]) for i in range(nqbits)]

sumop = Obs[0]
for a in Obs[1:]:
    sumop += a
#%%
# from qiskit.primitives import Sampler
from qiskit_aer.primitives import Sampler, Estimator
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit import QuantumCircuit

# QRS = QiskitRuntimeService()
#%%
# print(QRS.backends())
# sherbrooke_backend = QRS.backend("ibm_sherbrooke")
# sherbrooke_gates = sherbrooke_backend.gates
# print(sherbrooke_gates)
#%%
C0 = QuantumCircuit(nqbits)
for i in range(nqbits - 1):
    C0.h(i)

C0.draw("mpl")
#%%
circ.draw("mpl")
#%%
# circ = C0
circ.measure_all()
circ.draw("mpl")
#%%
C1 = C0.compose(circ, [0, 1, 2, 3, 4])
C1.draw("mpl")
sim = AerSimulator()
tcirc = transpile(C1, sim, ["rz", "ecr", "x", "sx"])
tcirc.draw("mpl", idle_wires=False)


#%%
# with Session(service=service,backend='aer_simulator') as session:
sampler = Sampler(run_options={"shots": None})
estimator = Estimator()

job = sampler.run(tcirc)
#%%
result = job.result()
print(result.quasi_dists)
data = np.zeros(32)
print(data)
for key in result.quasi_dists[0]:
    data[int(key)] = result.quasi_dists[0][key]
plt.plot(data[:16], label="qubit 4 = 0")
plt.plot(data[16:], label="qubit 4 = 1")
# plt.legend()
# plt.title("Probabilités de mesure, qubits 0 à 3 initialisé en superposition uniforme")
# plt.savefig("dist.pdf")

# %%
