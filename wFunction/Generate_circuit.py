import quimb.tensor as qtn
import qiskit as qs
from qiskit.quantum_info.operators import Operator
import qiskit.quantum_info as qi
import numpy as np
import re
from collections import defaultdict
import interpolate as terp
import mps2qbitsgates as mpqb
import compress_algs as calgs
from qiskit.converters import circuit_to_gate
import quantit as qtt
from jax.config import config
import torch
config.update("jax_enable_x64", True)

TN = qtn.TensorNetwork

def qtens2operator(tens:qtn.Tensor):
    """ To transpose or not to transpose, that is the question."""
    # print(tens.tags)
    # print(tens.data.reshape(4,4))
    ar = np.array(tens.data).transpose([3,2,1,0]).reshape(4,4).conj()
    return Operator(ar)

def extract_layer_link(tags,regexp):
    for tag in tags:
        match = re.search(regexp,tag)
        if match:
            layer = match.group(1)
            link = match.group(2)
    return int(layer),int(link)

def extract_layer(inds,regexp):
    layer = int(re.search(regexp,inds[0]).group(1))
    layer = max(layer,int(re.search(regexp,inds[1]).group(1)))
    return layer

def extract_qbits(inds,regexp):
    qbits0 = int(re.search(regexp,inds[0]).group(1))
    qbits1 = qbits0
    for ind in inds:
        nqbits = int(re.search(regexp,ind).group(1))
        if abs(qbits0 - nqbits) > 1: #we always act on neighbour qbits.
            return None
        qbits0 = min(qbits0,nqbits)
        qbits1 = max(qbits1,nqbits)
    return qbits0,qbits1

def prep_list_dict(tn:TN):
    md = defaultdict(list)
    for t in tn:
        layer,link = extract_layer_link(t.tags,"O(\d+),(\d+)")
        tags = [tag for tag in t.tags]
        md[layer].append({"obj":qtens2operator(t), "qubits":((link),(link)+1),'label':tags[2]})
    return md

def net2circuit(net:TN,nqbits,registers):
    circuit = qs.QuantumCircuit(registers)
    op_dict = prep_list_dict(net)
    largest_key = max(op_dict.keys())
    # for layer in range(largest_key,-1,-1): #reverse order of application of the gates.
    for layer in range(largest_key+1):
        op_list = op_dict[layer]
        for op in op_list:
            circuit.unitary(**op)
    return circuit

def poly_by_part(f,precision,nqbit,domain,qbmask=0,fulldomain=None):
    """Recursive separtion of the domain until the interpolation is able to reconstruct with a small enough error. Each polynomial in the return is paired with its bit domain"""
    # if fulldomain is None:
    fulldomain = domain #Fulldomain was a mistake. only domain matter. refactor to remove it.
    if nqbit >1:
        poly = terp.interpolate(f,10,domain,fulldomain)
    else:
        #we ran out of qbits... use a linear interpolation sampling the actual value the qbit can access.
        #There's no point in having a proper interpolation in this case.
        x0,x1 = domain
        y0,y1 = f(x0),f(x1)
        poly = np.polynomial.Polynomial([(y1*x0-y0*x1)/(x0-x1),(y0-y1)/(x0-x1)],domain,domain)
    if (np.abs(f(domain[0])-poly(domain[0])) < precision and np.abs(f(domain[1])-poly(domain[1])) < precision ):
        return [(poly,(qbmask,qbmask+(1<<nqbit)-1))]
    else:
        nqbit-=1
        midpoint = (domain[0]+domain[1])/2
        leftdomain = (domain[0],midpoint)
        rightdomain=(midpoint,domain[1])
        leftbitmask = 1<<nqbit|qbmask
        return [*poly_by_part(f,precision,nqbit,leftdomain,qbmask,fulldomain),*poly_by_part(f,precision,nqbit,rightdomain,leftbitmask,fulldomain)]


def Generate_unitary_net(f,MPS_precision,Gate_precision,nqbit,domain,Nlayer):
    polys = poly_by_part(f,MPS_precision,nqbit,domain)
    mpses = [terp.polynomial2MPS(poly,nqbit,pdomain,domain) for poly,pdomain in polys]
    Norm2 = np.sum([qtt.networks.contract(m,m) for m in mpses])
    mps = calgs.MPS_compressing_sum(mpses,Norm2,0.1*MPS_precision,MPS_precision)
    oc = mps.orthogonality_center
    mps[oc]/= np.sqrt(torch.tensordot(mps[oc],mps[oc],dims=([0,1,2],[0,1,2])))#Set the norm to one, freshly computer to correct any norm error in the opimization
    unitary_set,Infidelity = mpqb.MPS2Gates(mps,Gate_precision,Nlayer)
    print(Infidelity)
    return unitary_set

def Generate_circuit(f,MPS_precision,Gate_precision,nqbit,domain,register,Nlayer,name="function_gate"):
    unitary_set=Generate_unitary_net(f,MPS_precision,Gate_precision,nqbit,domain,Nlayer)
    circuit = net2circuit(unitary_set,nqbit,register)
    # circuit.name = name
    return circuit

def Generate_gate(f,MPS_precision,Gate_precision,nqbit,domain,Nlayer,name="function_gate"):
    register = qs.QuantumRegister(nqbit)
    return circuit_to_gate(Generate_circuit(f,MPS_precision,Gate_precision,nqbit,domain,register,Nlayer,name))


if __name__=='__main__':
    # import mps2qbitsgates as mpqb 
    # import quantit as qtt
    import matplotlib.pyplot as plt
    import seaborn as sb
    sb.set_theme()
    import jax.numpy as jnp
    def f(x):
        return np.exp(-x**2)
    nqbit = 10
    Nlayer = 3
    domain = (-3,3)
    Gate_precision = 1e-12
    MPS_precision = 0.001
    register = qs.QuantumRegister(nqbit)
    circuit = Generate_circuit(f,MPS_precision,Gate_precision,register.size,domain,register,Nlayer)
    print(circuit)
    # polys = poly_by_part(f,precision,nqbit,domain)
    # for poly,bitdomain in polys:
    #     subdomain = (terp.bits2range(bitdomain[0],domain,nqbit),terp.bits2range(bitdomain[1],domain,nqbit))
    #     w = np.linspace(*subdomain,300)
    #     plt.plot(w,poly(w))
    # plt.show()
    # mpses = [terp.polynomial2MPS(poly,nqbit,pdomain,domain) for poly,pdomain in polys]
    # X = calgs.MPS_compressing_sum(mpses,0.1*precision,precision)
    # # X = qtt.networks.random_MPS(5,5,2)
    # qX = mpqb.qttMPS2quimbTN(X,'lfq')
    # qX.compress_all(inplace=True).compress_all(inplace=True)
    # qX /= jnp.sqrt((qX@qX.conj()))
    # O = mpqb.TwoqbitsPyramid(qX)
    # print(O)
    # O.draw()
    # reg,circuit = net2circuit(O,5)
    # print(circuit)
    # list_dict = prep_list_dict(O)
    # print(qi.TwoQubitBasisDecomposer( qs.extensions.UnitaryGate(list_dict[0][0])))
    # register = qs.QuantumRegister(nqbit)
    # gate = Generate_circuit(f,precision,nqbit,domain,register,"normalp=0.001")
    # print(gate)
    from qiskit.circuit import qpy_serialization
    with open('normal.qpy', 'wb') as fd:
        qpy_serialization.dump(circuit,fd)
