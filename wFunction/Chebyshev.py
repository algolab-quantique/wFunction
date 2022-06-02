
if __name__=='__main__' and (__package__ is None or __package__ == ''):
    #fiddling to make this file work when loaded directly...
    import sys
    import os 
    from pathlib import Path

    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[1]
    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError: # Already removed
        pass
    import wFunction
    __package__ = 'wFunction'
    dir_path = os.path.dirname(os.path.realpath(__file__))



from . import interpolate as terp
from . import compress_algs as calgs
import jax.numpy as jnp
import numpy as np
import quimb.tensor as qtn

def gen_Xi(i,domain,nbits):
    X =  np.zeros((2,2,2,2))
    X[0,:,0,:] = np.eye(2)
    X[1,1,0,1] = terp.bits2range(2**i,domain,nbits)-domain[0]
    X[1,:,1,:] = np.eye(2)
    return X


def gen_X(domain,nbits):
    arr = [gen_Xi(i,domain,nbits).transpose(0,2,1,3) for i in range(nbits)]
    arr[0] = arr[0][1,:,:,:]
    arr[-1][1,0,0,0] += domain[0]
    arr[-1][1,0,1,1] += domain[0]
    arr[-1] = arr[-1][:,0,:,:]
    return qtn.MatrixProductOperator(arr)


"""Ça va prendre  un algo de compression pour la récursion à trois termes."""

def Chebyshevs(order,nqbits,tol=1e-13):
    C0 = [np.array([[[1,1]]]) for i in range(nqbits)]
    C0[0] = C0[0][0,:,:]
    C0[-1] = C0[-1][0,:,:]
    X = gen_X((-1,1),nqbits)
    C0 = qtn.MatrixProductState(C0,site_tag_id='b{}')
    # # C1 = three_term_compress(X,2*C0) 
    # out = [C0,C1]
    # for i in range(order):
    #     tmp1 = 2*X@out[-1]
    #     tmp1.site_ind_id = 'b{}'
    #     tmp2 = -1*out[-2]
    #     norm = tmp1@tmp1.H + tmp2@tmp2.H + tmp1.H@tmp2 + tmp1@tmp2.H
    #     out.append(calgs.MPS_compressing_sum([tmp1,tmp2],norm,tol,tol*2))
    # return out

if __name__=='__main__':
    import seaborn as sns
    sns.set_theme()
    import matplotlib.pyplot as plt
    def f(x):
        return jnp.exp(-x**2/2)
    nqbit = 4
    # Chebs = Chebyshevs(5,nqbit)
    # for c in Chebs:
    #     print(c)
