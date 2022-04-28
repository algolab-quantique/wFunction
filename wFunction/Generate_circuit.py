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
config.update("jax_enable_x64", True)

TN = qtn.TensorNetwork

def qtens2operator(tens:qtn.Tensor):
    """ To transpose or not to transpose, that is the question."""
    ar = np.array(tens.data).reshape(4,4).transpose().conj()
    return Operator(ar)

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
        layer = extract_layer(t.inds,"l(\d+)")
        tags = [tag for tag in t.tags]
        T = t.data.reshape(4,4)
        md[layer].append({"obj":qtens2operator(t), "qubits":extract_qbits(t.inds,"q(\d+)"),'label':tags[1]})
    return md

def net2circuit(net:TN,nqbits,registers):
    circuit = qs.QuantumCircuit(registers)
    op_dict = prep_list_dict(net)
    largest_key = max(op_dict.keys())
    for layer in range(largest_key+1):
        op_list = op_dict[layer]
        for op in op_list:
            circuit.unitary(**op)
    return circuit

def poly_by_part(f,precision,nqbit,domain,qbmask=0,fulldomain=None):
    """Recursive separtion of the domain until the interpolation is able to reconstruct with a small enough error. Each polynomial in the return is paired with its bit domain"""
    if fulldomain is None:
        fulldomain = domain
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


def Generate_unitary_net(f,precision,nqbit,domain):
    polys = poly_by_part(f,precision,nqbit,domain)
    mpses = [terp.polynomial2MPS(poly,nqbit,pdomain,domain) for poly,pdomain in polys]
    mps = calgs.MPS_compressing_sum(mpses,0.1*precision,precision)
    mps[mps.orthogonality_center]/= np.sqrt(qtt.networks.contract(mps,mps))
    unitary_set,Infidelity = mpqb.MPS2Gates(mps,precision)
    return unitary_set

def Generate_circuit(f,precision,nqbit,domain,register,name="function_gate"):
    unitary_set=Generate_unitary_net(f,precision,nqbit,domain)
    circuit = net2circuit(unitary_set,nqbit,register)
    # circuit.name = name
    return circuit

def Generate_gate(f,precision,nqbit,domain,name="function_gate"):
    register = qs.QuantumRegister(nqbit)
    return circuit_to_gate(Generate_circuit(f,precision,nqbit,domain,register,name))


if __name__=='__main__':
    # import mps2qbitsgates as mpqb 
    # import quantit as qtt
    import jax.numpy as jnp
    def f(x):
        return np.exp(-x**2)
    nqbit = 10
    domain = (-1,1)
    precision = 0.001
    # polys = poly_by_part(f,0.001,nqbit,domain)
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
    register = qs.QuantumRegister(nqbit)
    gate = Generate_circuit(f,precision,nqbit,domain,register,"normalp=0.001")
    print(gate)
    from qiskit.circuit import qpy_serialization
    with open('normal.qpy', 'wb') as fd:
        qpy_serialization.dump(gate,fd)
