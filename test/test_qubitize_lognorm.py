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
from scipy.stats import lognorm as ln_sp
from wFunction import scalarQubitization as sQ
from wFunction.scalarQubitization import (
    eval_SU2_func,
    evalUmats_SU2_func,
    get_angles,
    get_rotations,
    get_unitary_transform,
    opt_qubitizetrealangles,
    phi_SU2_func,
    qubitize_scalar,
)
from matplotlib import pyplot as plt

max_layers = 1024
precision = 5e-3
# on the test functions, we only need 10 layers
n_layer = 10


def lognorm(x):
    return ln_sp.pdf(x * np.exp(-1), 1)


domain = [0, 7]
t = np.linspace(0, 7, 300)
lnt = lognorm(t)
plt.plot(t, lognorm(t))
lnt2 = np.array([*lnt[:-1], *lnt[-1::-1]])
#%%
LNt = np.fft.fft(lnt2, norm="forward")
plt.plot(np.real(LNt))
plt.plot(np.imag(LNt))

#%%

#%%
U, doubled = get_unitary_transform(lognorm, domain, max_layers, precision)
rotsln = [*get_rotations(lognorm, domain, max_layers, precision)]
philn = [*get_angles(lognorm, domain, max_layers, precision)]

# err =(np.array(phi2rots(philn)))-(np.array(rotsln))

# plt.plot(err[:,0,0])
#%%
fln = eval_SU2_func(rotsln)
pln = eval_SU2_func(philn)
flU = evalUmats_SU2_func(U)
flUt = np.array([flU(tt) for tt in np.linspace(0, np.pi, 599)])
flnt = np.array([fln(tt) for tt in np.linspace(0, np.pi, 599)])
plnt = np.array([pln(tt) for tt in np.linspace(0, np.pi, 599)])
psuf = np.array([phi_SU2_func(tt, philn) for tt in np.linspace(0, np.pi, 599)])
plt.plot(lnt2, label="exact")
plt.plot(flUt[:, 0, 0], label="flut")
plt.plot(flnt[:, 0, 0], label="flnt")
# plt.plot(plnt[:,0,0])
plt.plot(plnt[:, 1, 1], label="plnt")
plt.plot(psuf[:, 0, 0], label="psuf")
plt.legend()
plt.savefig("lntest.pdf")
# plt.plot(np.real(flnt[:,0,1]))
# plt.plot(np.imag(flnt[:,0,1]))
#%%
print(W(philn, np.linspace(0, np.pi, 599), lnt2))

#%%
ft2 = flnt[:, 0, 0]
Ft2 = np.fft.fft(ft2, norm="forward")

#%%

plt.plot(np.real(Ft2[-30:]))
plt.plot(np.real(Ft2[:30]))

#%%
plt.plot(np.real(Ft2[:30]))
plt.plot(np.real(-Ft2[:30]))
# plt.plot(np.imag(Ft2))
plt.plot(np.real(LNt[:30]))
# plt.plot(np.imag(LNt))
#%%

plt.plot(np.real(Ft2[-30:]))
plt.plot(np.real(-Ft2[-30:]))
# plt.plot(np.imag(Ft2))
plt.plot(np.real(LNt[-30:]))
# plt.plot(np.imag(LNt))

#%%Le circuit produit la bonne réponse à une superposition uniforme.
nqbits = 10
circ = qubitize_scalar(
    lognorm, nqbits, [0, 7], 256, 1e-3, cpt_rotations_matrices=sQ.get_unitary_transform
)
# circ.draw('mpl')

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

# C0.draw('mpl')
#%%
# circ.draw('mpl')
#%%
# circ = C0
circ.measure_all()
# circ.draw('mpl')
#%%
C1 = C0.compose(circ, [*range(nqbits)])
# C1.draw('mpl')
sim = AerSimulator()
tcirc = transpile(C1, sim, ["rz", "ecr", "x", "sx"])
# tcirc.draw('mpl',idle_wires=False)


#%%
# with Session(service=service,backend='aer_simulator') as session:
sampler = Sampler(run_options={"shots": None})
estimator = Estimator()

job = sampler.run(tcirc)
#%%
result = job.result()
# print(result.quasi_dists)
data = np.zeros(2**nqbits)
# print(data)
x = np.linspace(*domain, 2 ** (nqbits - 1))
plt.plot(lognorm(x), label="exact")
for key in result.quasi_dists[0]:
    data[int(key)] = result.quasi_dists[0][key]
plt.plot(np.sqrt(data[: 2 ** (nqbits - 1)] * 2 ** (nqbits - 1)), label="qubit 4 = 0")
plt.plot(np.sqrt(data[2 ** (nqbits - 1) :] * 2 ** (nqbits - 1)), label="qubit 4 = 1")

plt.legend()
plt.show()

# plt.title("Probabilités de mesure, qubits 0 à 3 initialisé en superposition uniforme")
# plt.savefig("dist.pdf")

# %%
ophi = opt_qubitizetrealangles(lognorm, domain, 1e-3, 256)
#%%
opln = eval_SU2_func(ophi)
oplnt = np.array([pln(tt) for tt in np.linspace(0, np.pi / 2, 300)])


plt.plot(t, lnt)
plt.plot(t, oplnt[:, 0, 0])
# %%
