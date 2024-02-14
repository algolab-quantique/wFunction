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
import quimb.tensor as qtn
import numpy as np
from wFunction.MPO_compress import MPO_compressing_sum

MPO = qtn.MatrixProductOperator
MPS = qtn.MatrixProductState
#%%
c_up = np.zeros((4, 4))
c_up[0, 1] = 1
c_up[2, 3] = 1
c_dn = np.zeros((4, 4))
c_dn[0, 2] = 1
c_dn[1, 3] = -1
F = np.eye(4)
F[1, 1] = -1
F[2, 2] = -1
id = np.eye(4)
n_up = c_up.T @ c_up
n_dn = c_dn.T @ c_dn


def MPO_C(local_c, F, s, N):
    tens = []
    assert F.shape == local_c.shape
    i = 0
    while i < s:
        tens.append(np.array([[F]]))
        i += 1
    tens.append(np.array([[local_c]]))
    i += 1
    while i < N:
        tens.append(np.array([[np.eye(F.shape[0])]]))
        i += 1
    tens[0] = tens[0][:, 0, :, :]
    tens[-1] = tens[-1][0, :, :, :]
    return MPO(tens)


def Hubbard_1D(t, mu, U, N):
    tens = np.zeros((6, 6, 4, 4))
    tens[0, 0, :, :] = id
    tens[5, 5, :, :] = id
    tens[5, 1, :, :] = F @ c_up
    tens[5, 2, :, :] = c_up.T @ F
    tens[5, 3, :, :] = F @ c_dn
    tens[5, 4, :, :] = c_dn.T @ F
    tens[1, 0, :, :] = t * (c_up.T)
    tens[2, 0, :, :] = t * (c_up)
    tens[3, 0, :, :] = t * (c_dn.T)
    tens[4, 0, :, :] = t * (c_dn)
    tens[5, 0, :, :] = U * (n_up @ n_dn) - mu * (n_up + n_dn)
    tensors = [tens for i in range(N)]
    tensors[0] = tensors[0][5, :, :, :]
    if len(tensors) > 1:
        tensors[-1] = tensors[-1][:, 0, :, :]
    else:
        tensors[0] = tensors[0][0, :, :]
    return qtn.MatrixProductOperator(tensors)


def Anderson_star(el, tl, U, mu, er, tr):
    tensors = []
    for e, t in zip(el, tl):
        tens = np.zeros((6, 6, 4, 4))
        tens[0, 0, :, :] = id
        tens[1, 1, :, :] = F
        tens[2, 2, :, :] = F
        tens[3, 3, :, :] = F
        tens[4, 4, :, :] = F
        tens[5, 5, :, :] = id
        tens[5, 0, :, :] = e * (n_up + n_dn)
        tens[5, 1, :, :] = t * (F @ c_up)
        tens[5, 2, :, :] = t * (c_up.T @ F)
        tens[5, 3, :, :] = t * (F @ c_dn)
        tens[5, 4, :, :] = t * (c_dn.T @ F)
        tensors.append(tens)
    tens = np.zeros((6, 6, 4, 4))
    tens[0, 0, :, :] = id
    tens[5, 5, :, :] = id
    tens[5, 1, :, :] = F @ c_up
    tens[5, 2, :, :] = c_up.T @ F
    tens[5, 3, :, :] = F @ c_dn
    tens[5, 4, :, :] = c_dn.T @ F
    tens[1, 0, :, :] = c_up.T
    tens[2, 0, :, :] = c_up
    tens[3, 0, :, :] = c_dn.T
    tens[4, 0, :, :] = c_dn
    tens[5, 0, :, :] = U * (n_up @ n_dn) - mu * (n_up + n_dn)
    tensors.append(tens)
    for e, t in zip(er, tr):
        tens = np.zeros((6, 6, 4, 4))
        tens[0, 0, :, :] = id
        tens[1, 1, :, :] = F
        tens[2, 2, :, :] = F
        tens[3, 3, :, :] = F
        tens[4, 4, :, :] = F
        tens[5, 5, :, :] = id
        tens[5, 0, :, :] = e * (n_up + n_dn)
        tens[1, 0, :, :] = t * (c_up.T)
        tens[2, 0, :, :] = t * (c_up)
        tens[3, 0, :, :] = t * (c_dn.T)
        tens[4, 0, :, :] = t * (c_dn)
        tensors.append(tens)
    tensors[0] = tensors[0][5, :, :, :]
    if len(tensors) > 1:
        tensors[-1] = tensors[-1][:, 0, :, :]
    else:
        tensors[0] = tensors[0][0, :, :]
    return qtn.MatrixProductOperator(tensors)


#%%

el = [-1, -0.5, -0.2, 0]
tl = [0.1, 0.1, 0.1, 0.1]
er = [1, 0.5, 0.2][-1::-1]
tr = [0.1, 0.1, 0.1]
U = 8
mu = 4
H = Anderson_star(el, tl, U, mu, er, tr)
# H = MPO_compressor(H,1e-15,5e-16) #essentially lossless compression
H.site_tag_id = "I{}"  # quimb MPO methods are supper brittle regarding de site_tag
#%%
star_dmrg = qtn.DMRG(H, 200, 1e-12)
star_dmrg.solve(tol=1e-12, max_sweeps=1000)
psi = star_dmrg.state
# %%


def Liouvillian(H: MPO, O: MPO, eps: float):
    H.site_tag_id = O.site_tag_id
    # une implémentation variationnel du commutateur pourrait donné un speedup significatif
    HO = O.apply(H)
    OH = -1 * H.apply(O)
    OH.upper_ind_id = HO.upper_ind_id
    OH.lower_ind_id = HO.lower_ind_id
    return MPO_compressing_sum([HO, OH], eps, eps * 0.5)


def inner_product(A: MPO, B: MPO, ket: MPS):
    """The Liouvillian inner product for the computation
    of Fermionic Green's function: <{A,B^\dagger}>_{psi}"""
    BH = B.H
    bra = ket.H
    A = A.copy()  # so we don't modify the input args.
    bra.site_ind_id = "z{}"
    A.upper_ind_id = "x{}"
    A.lower_ind_id = ket.site_ind_id
    BH.upper_ind_id = A.upper_ind_id
    BH.lower_ind_id = bra.site_ind_id
    BA = (bra | BH | A | ket).contract()
    BH.upper_ind_id = ket.site_ind_id
    BH.lower_ind_id = "y{}"
    A.lower_ind_id = "y{}"
    A.upper_ind_id = bra.site_ind_id
    AB = (bra | A | BH | ket).contract()
    return AB + BA


def Liouville_Lanczos(H: MPO, O_0: MPO, psi: MPS, N: int, eps: float):
    O_i = O_0.copy()
    b_i = np.sqrt(inner_product(O_i, O_i, psi))
    O_i /= b_i
    O_ip = Liouvillian(H, O_i, eps)
    a_i = inner_product(O_ip, O_i, psi)
    O_ip = MPO_compressing_sum([O_ip, -1 * a_i * O_i], eps, eps * 0.5)
    b = []
    a = []
    a.append(a_i)
    b.append(b_i)
    b_ip = np.sqrt(inner_product(O_ip, O_ip, psi))
    b.append(b_ip)
    O_ip /= b_ip
    b_i = b_ip
    O_i, O_im = O_ip, O_i
    for i in range(1, N):
        O_ip = Liouvillian(H, O_i, eps)
        print("Largest bond at iteration {}: {}".format(i, O_ip.max_bond()))
        a_i = inner_product(O_ip, O_i, psi)
        a.append(a_i)
        O_ip = MPO_compressing_sum([O_ip, -1 * a_i * O_i, -b_i * O_im], eps, eps * 0.5)
        print("Largest bond at iteration {}: {}".format(i, O_ip.max_bond()))
        b_ip = np.sqrt(inner_product(O_ip, O_ip, psi))
        if abs(b_ip) < 1e-14:
            break
        O_ip /= b_ip
        b.append(b_ip)
        b_i = b_ip
        O_im, O_i = O_i, O_ip
    return a, b[0:-1]


# %%
C_0 = MPO_C(c_up, F, len(el), H.L)
#%%
# a,b = Liouville_Lanczos(H,C_0,psi,5,1e-12)
# %%


el = [-1, 0]
tl = [0.1, 0.1]
er = [1][-1::-1]
tr = [0.1]
U = 8
mu = 4
H = Anderson_star(el, tl, U, mu, er, tr)
M = (
    H.contract()
    .transpose(
        *["k{}".format(i) for i in range(4)], *["b{}".format(i) for i in range(4)]
    )
    .data.reshape(4**4, 4**4)
)

star_dmrg = qtn.DMRG(H, 200, 1e-12)
star_dmrg.solve(tol=1e-12, max_sweeps=1400)
# %%
el = [-1, 0]
tl = [0.5, 0.5]
er = [1][-1::-1]
tr = [0.5]
U = 8
mu = 4
H = Anderson_star(el, tl, U, mu, er, tr)
M = (
    H.contract()
    .transpose(
        *["k{}".format(i) for i in range(4)], *["b{}".format(i) for i in range(4)]
    )
    .data.reshape(4**4, 4**4)
)

star_dmrg = qtn.DMRG(H, 400, 1e-12)
res = star_dmrg.solve()

#%%
State = star_dmrg.state

CUP = [MPO_C(c_up, F, i, 4) for i in range(4)]
CDUP = [MPO_C(c_up.T, F, i, 4) for i in range(4)]
CDN = [MPO_C(c_dn, F, i, 4) for i in range(4)]
CDDN = [MPO_C(c_dn.T, F, i, 4) for i in range(4)]

rhoup = np.zeros((4, 4))
rhodn = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        rhoup[i, j] = (CDUP[i].apply(CUP[j].apply(State))) @ State
        rhodn[i, j] = (CDDN[i].apply(CDN[j].apply(State))) @ State
print(rhoup.round(4))
print(rhodn.round(4))


# %%
Hub = Hubbard_1D(1, 4, 8, 4)
hub_dmrg = qtn.DMRG(Hub, 200, 1e-12)
hub_dmrg.solve(tol=1e-12, max_sweeps=1400)
print(hub_dmrg.energy)
#%%
state = hub_dmrg.state


def electron_to_qubit_MPO(mpo: qtn.MatrixProductOperator, eps=1e-12):
    """
    split the size four physical indices of each electron sites into 2 qubits sites
    """
    mpo.permute_arrays("ldur")
    tensors = []
    t = mpo[0]
    # split the second index in two
    e_in_ind_id = mpo.lower_ind_id
    e_out_ind_id = mpo.upper_ind_id
    q_in_ind = qtn.rand_uuid() + "{}"
    q_out_ind = qtn.rand_uuid() + "{}"
    i = 0
    j = 0
    e_in_ind = e_in_ind_id.format(j)
    e_out_ind = e_out_ind_id.format(j)
    j += 1
    q_in_A = q_in_ind.format(i)
    q_out_A = q_out_ind.format(i)
    i += 1
    q_in_B = q_in_ind.format(i)
    q_out_B = q_out_ind.format(i)
    i += 1
    unfuse_map = {e_in_ind: (q_in_A, q_in_B), e_out_ind: (q_out_A, q_out_B)}
    shape_map = {e_in_ind: (2, 2), e_out_ind: (2, 2)}
    t = t.unfuse(unfuse_map, shape_map)
    a, b = qtn.tensor_split(t, [q_in_A, q_out_A], cutoff=0)
    tensors.append(a.data)
    tensors.append(b.data)
    for t in mpo[1:]:
        e_in_ind = e_in_ind_id.format(j)
        e_out_ind = e_out_ind_id.format(j)
        q_in_A = q_in_ind.format(i)
        q_out_A = q_out_ind.format(i)
        i += 1
        q_in_B = q_in_ind.format(i)
        q_out_B = q_out_ind.format(i)
        i += 1
        unfuse_map = {e_in_ind: (q_in_A, q_in_B), e_out_ind: (q_out_A, q_out_B)}
        shape_map = {e_in_ind: (2, 2), e_out_ind: (2, 2)}
        t = t.unfuse(unfuse_map, shape_map)
        a, b = t.split([mpo.bond(j - 1, j), q_in_A, q_out_A], cutoff=0)
        j += 1
        tensors.append(a.data)
        tensors.append(b.data)
    return qtn.MatrixProductOperator(tensors, shape="ldur")


def electron_to_qubit_MPS(mps: qtn.MatrixProductState, eps=1e-12):
    """
    split the size four physical indices of each electron sites into 2 qubits sites
    """
    mps.permute_arrays("lpr")
    tensors = []
    t = mps[0]
    # split the second index in two
    in_ind = mps.site_ind_id
    out_ind = qtn.rand_uuid() + "{}"
    i = 0
    j = 0
    e_ind = in_ind.format(j)
    j += 1
    qubit_A = out_ind.format(i)
    i += 1
    qubit_B = out_ind.format(i)
    i += 1
    unfuse_map = {e_ind: (qubit_A, qubit_B)}
    shape_map = {e_ind: (2, 2)}
    t = t.unfuse(unfuse_map, shape_map)
    a, b = t.split(qubit_A, cutoff=0)
    tensors.append(a.data)
    tensors.append(b.data)
    for t in mps[1:]:
        e_ind = in_ind.format(j)
        qubit_A = out_ind.format(i)
        i += 1
        qubit_B = out_ind.format(i)
        i += 1
        unfuse_map = {e_ind: (qubit_A, qubit_B)}
        shape_map = {e_ind: (2, 2)}
        t = t.unfuse(unfuse_map, shape_map)
        a, b = t.split([mps.bond(j - 1, j), qubit_A], cutoff=0)
        j += 1
        tensors.append(a.data)
        tensors.append(b.data)
    return qtn.MatrixProductState(tensors, shape="lpr")


qubit_state = electron_to_qubit_MPS(
    state
)  # There is a significant loss of precision..,
#%%
qubit_Hub = electron_to_qubit_MPO(Hub)
qhub_dmrg = qtn.DMRG(qubit_Hub, 200, 1e-12)
qhub_dmrg.solve(tol=1e-12, max_sweeps=1400)
qubit_state = qhub_dmrg.state
print(qhub_dmrg.energy)
# %%
from wFunction.mps2qbitsgates import MPS2gates2, trivial_state

circuit, error = MPS2gates2(qubit_state, 5e-2, 200)
# %% Test Circuit
import re

p_ind = qubit_state.site_ind_id
p_ind = "^" + p_ind.replace("{}", "\d+") + "$"
fout_inds = []
other_ind = []
for out_ind in circuit.outer_inds():
    match = re.search(p_ind, out_ind)
    if re.search(p_ind, out_ind) is not None:
        fout_inds.append(out_ind)
    else:
        other_ind.append(out_ind)
#%%
other_ind_id = other_ind[0][0:-1] + "{}"
Zero_state = trivial_state(len(other_ind), other_ind_id)
circ_state = Zero_state & circuit
circ_state.reindex_({ind: qtn.rand_uuid() for ind in circ_state.inner_inds()})
T = circ_state & qubit_Hub
#%%
T.reindex_({ind: qtn.rand_uuid() for ind in T.inner_inds()})
for i in range(qubit_Hub.L):
    T.reindex({qubit_Hub.lower_ind_id.format(i): fout_inds[i]}, inplace=True)

# T = Hub.apply(circ_state)
#%%
E_circ = T.H @ circ_state
E = qhub_dmrg.energy
rel_err = (E - E_circ) / E
print("Expected Energy:", E)
print("Circuit energy:", E_circ)
print("relative error", rel_err * 100, "%")
# %%
