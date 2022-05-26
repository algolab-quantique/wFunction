

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

import numpy as np
# import quantit as qtt
# import torch
from scipy.special import binom
import quimb.tensor as qtn
from . import compress_algs

numpy = np
Poly=numpy.polynomial.polynomial.Polynomial

"""
seperate the function in 2^k (with k not too large) subdomains.
Fit the subdomains on smooth low-degree polynomials.
Turn each polynomial into a MPS where the qbits adress a value of the indepent variable of the function on the domain.
Sum and condense all those (2^k) MPS into a single one.
"""

def bits2range(bits:int,domain:tuple[float,float],nbits:int):
    """
    Linear map from an int in [0,2**nbits[ to [a,b[ 
    """
    b = domain[1]
    a = domain[0]
    return (b-a)*bits/(2**nbits-1)+a


def interpolate(f,degree:int,domain,fulldomain=None) -> Poly:
    """
    return a polynomial object that interpolate f on the specified domain, with a low degree polynomial.
    if the maximum degree doesn't give you good enough result, try using more numerous, smaller domains.
    """
    if fulldomain is None:
        fulldomain = domain
    assert(degree<= 10)
    poly = np.polynomial.chebyshev.Chebyshev.interpolate(f, degree, domain=domain)
    return Poly(np.polynomial.chebyshev.cheb2poly(poly.coef),poly.domain,poly.window).convert(domain=domain,window=fulldomain)

def Phi_s(poly:Poly,s:int,x:float):
    p = poly.degree()
    return np.sum([poly.coef[k]*binom(k,s)*x**(k-s) for k in range(s,p+1)])

def G(poly:Poly,polydomain_start:int,fulldomain,n:int,nqbits:int):
    out = np.zeros([poly.degree()+1,2,poly.degree()+1])
    x0 = 0
    x1 = bits2range(1<<n,fulldomain,nqbits)-fulldomain[0]
    for i in range(poly.degree()+1):
        for j in range(0,i+1):
            bin = binom(i,i-j)
            out[i,0,j] = bin*x0**(i-j)
            out[i,1,j] = bin*x1**(i-j)
    return out

# def G(poly:Poly,polydomain_start:int,fulldomain,n:int,nqbits:int):
#     out = torch.zeros([poly.degree()+1,2,poly.degree()+1])
#     x0 = 0
#     x1 = bits2range(1<<n,fulldomain,nqbits)-fulldomain[0]
#     for i in range(poly.degree()+1):
#         for j in range(0,i+1):
#             bin = binom(i,i-j)
#             out[i,0,j] = bin*x0**(i-j)
#             out[i,1,j] = bin*x1**(i-j)
#     return out

def polynomial2MPS(poly:Poly, nqbits:int,polydomain:tuple[int,int], fulldomain:tuple[float,float]):
    """
    takes the polynomial defined on the polydomain and turn it into a qbits MPS with nqbits that exist on the fulldomain.
    polydomain must be given as two integers, those integer should convert to the support of the polynomial when given to bits2range with fulldomain for domain.
    """
    #tensor at index 0 is most significant qbit.
    pd0 = polydomain[0]
    pd1 = polydomain[1]
    Ntensor_support = int(np.ceil(np.log2((pd0^pd1)+1)))
    out = [None for x in range(nqbits)]
    trivial_bits = polydomain[0]>>(Ntensor_support)
    for i in range(-Ntensor_support-1,-len(out)-1,-1):
        out[i] = np.zeros([1,2,1])
        bit = trivial_bits & 1
        trivial_bits = trivial_bits>>1
        out[i][0,bit,0] = 1
    if Ntensor_support > 1:
        out[-Ntensor_support] = np.zeros((1,2,poly.degree()+1))
        for i in range(poly.degree()+1):
            T = out[-Ntensor_support]
            x = 0
            T[0,0,i] = Phi_s(poly,i,x)
            x = bits2range( (1<<(Ntensor_support-1)),fulldomain,nqbits)-fulldomain[0]
            T[0,1,i] = Phi_s(poly,i,x)
        for n in range(1,Ntensor_support-1):
            out[-n-1] = G(poly,pd0,fulldomain,n,nqbits)
        out[-1] = np.zeros((poly.degree()+1,2,1))
        out[-1][:,1,0] = np.array([bits2range(pd0+1,fulldomain,nqbits)**p for p in range(poly.degree()+1)])
        out[-1][:,0,0] = np.array([bits2range(pd0,fulldomain,nqbits)**p for p in range(poly.degree()+1)])
    else:
        out[-1] = np.zeros((1,2,1))
        out[-1][0,0,0] = poly(bits2range(pd0,fulldomain,nqbits))
        out[-1][0,1,0] = poly(bits2range(pd1,fulldomain,nqbits))
    out = [x.transpose(0,2,1) for x in out]
    out[0] = out[0][0,:,:]
    out[-1] = out[-1][:,0,:]
    out = qtn.MatrixProductState(out,site_ind_id='lfq{}')
    return out

# def polynomial2MPS(poly:Poly, nqbits:int,polydomain:tuple[int,int], fulldomain:tuple[float,float],dtype = torch.float64):
#     """
#     takes the polynomial defined on the polydomain and turn it into a qbits MPS with nqbits that exist on the fulldomain.
#     polydomain must be given as two integers, those integer should convert to the support of the polynomial when given to bits2range with fulldomain for domain.
#     """
#     #tensor at index 0 is most significant qbit.
#     torch.set_default_dtype(dtype)
#     pd0 = polydomain[0]
#     pd1 = polydomain[1]
#     Ntensor_support = int(np.ceil(np.log2((pd0^pd1)+1)))
#     out = qtt.networks.MPS(nqbits)
#     trivial_bits = polydomain[0]>>(Ntensor_support)
#     for i in range(-Ntensor_support-1,-len(out)-1,-1):
#         out[i] = torch.zeros([1,2,1])
#         bit = trivial_bits & 1
#         trivial_bits = trivial_bits>>1
#         out[i][0,bit,0] = 1
#     if Ntensor_support > 1:
#         out[-Ntensor_support] = torch.Tensor(1,2,poly.degree()+1)
#         for i in range(poly.degree()+1):
#             T = out[-Ntensor_support]
#             x = 0
#             T[0,0,i] = Phi_s(poly,i,x)
#             x = bits2range( (1<<(Ntensor_support-1)),fulldomain,nqbits)-fulldomain[0]
#             T[0,1,i] = Phi_s(poly,i,x)
#         for n in range(1,Ntensor_support-1):
#             out[-n-1] = G(poly,pd0,fulldomain,n,nqbits)
#         out[-1] = torch.zeros(poly.degree()+1,2,1)
#         out[-1][:,1,0] = torch.tensor([bits2range(pd0+1,fulldomain,nqbits)**p for p in range(poly.degree()+1)])
#         out[-1][:,0,0] = torch.tensor([bits2range(pd0,fulldomain,nqbits)**p for p in range(poly.degree()+1)])
#     else:
#         out[-1] = torch.Tensor(1,2,1)
#         out[-1][0,0,0] = poly(bits2range(pd0,fulldomain,nqbits))
#         out[-1][0,1,0] = poly(bits2range(pd1,fulldomain,nqbits))
#     return out
    
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    import seaborn as sb
    sb.set_theme()
    def y(x):
        return np.exp(-x**2)
    from scipy.stats import lognorm as scplog
    def lognorm(x,mu,sigma):
        return scplog.pdf(np.exp(-mu)*x,sigma )
    y = lambda x : lognorm(x,1,1)
    domain = (0,7)
    subdomain = [(domain[0],3),(4,domain[1])]
    X = np.linspace(domain[0],domain[1],1000)
    X1 = np.linspace(subdomain[0][0],subdomain[0][1],1000)
    X2 = np.linspace(subdomain[1][0],subdomain[1][1],1000)
    nbits = 3
    poly1 = interpolate(y,10,subdomain[0])
    poly2 = interpolate(y,10,subdomain[1])
    plt.plot(X1,poly1(X1))
    plt.plot(X2,poly2(X2))
    plt.show()
    mps1 = polynomial2MPS(poly1,nbits,[0b000,0b11],domain)
    mps2 = polynomial2MPS(poly2,nbits,[0b100,0b111],domain)
    bitsdomain = bits2range(np.array(range(2**nbits)),domain,nbits)
    test_samples = y(bitsdomain)
    mps1_samples = np.array([ qtn.MPS_computational_state(format(i,'0'+ str(nbits) +'b'),site_ind_id='lfq{}')@mps1 for i in range(0,2**nbits)])
    mps2_samples = np.array([ qtn.MPS_computational_state(format(i,'0'+ str(nbits) +'b'),site_ind_id='lfq{}')@mps2 for i in range(0,2**nbits)])
    # mps2_samples = np.array([ torch.matmul(mps2[0][:,(i>>3)&1,:],torch.matmul(torch.matmul(mps2[1][:,(i>>2)&1,:],mps2[2][:,(i>>1)&1,:]),mps2[3][:,i&1,:])).item() for i in range(0,16)])
    # print(bitsdomain)
    w = np.linspace(domain[0],domain[1],2**nbits)
    # Norm = np.sum((mps1_samples+mps2_samples)**2)
    Norm = mps1@mps1 + mps2@mps2
    cMPS = compress_algs.MPS_compressing_sum([mps1,mps2],Norm,1e-6,1e-6)
    # for t in cMPS:
    #     print(t.size())
    # oc = cMPS.orthogonality_center
    # print("tensor norm: ", torch.tensordot(cMPS[oc],cMPS[oc],dims=([0,1,2],[0,1,2])).item())
    # print("overlap : ",qtt.networks.contract(cMPS,mps1).item()+qtt.networks.contract(cMPS,mps2).item() )
    print("expected norm: ", np.sum(y(X))*(X[1]-X[0]))
    # print("expected norm from poly MPS: ", np.sum((mps1_samples+mps2_samples)**2 )*2/(2**4-1))
    cMPS_samples = np.array([ qtn.MPS_computational_state(format(i,'0'+ str(nbits) +'b'),site_ind_id='lfq{}')@cMPS for i in range(0,2**nbits)])
    mps1.show()
    mps2.show()
    cMPS.show()
    print("test_samples")
    print(test_samples)
    print(mps1_samples)
    print(mps2_samples)
    print(cMPS_samples)
    print("test_samples end")
    # cMPS_samples = np.array([ torch.matmul(cMPS[0][:,(i>>3)&1,:],torch.matmul(torch.matmul(cMPS[1][:,(i>>2)&1,:],cMPS[2][:,(i>>1)&1,:]),cMPS[3][:,i&1,:])).item() for i in range(0,16)])
    # print(cMPS_samples)
    # print("actual norm polyMPS",np.sum(cMPS_samples**2)*2/(2**4-1))
    plt.plot(w,mps1_samples)
    plt.plot(w,mps2_samples)
    plt.plot(w,cMPS_samples)
    plt.show()


