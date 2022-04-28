import numpy as np
import quantit as qtt
import torch
from scipy.special import binom

import compress_algs

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
    out = torch.zeros([poly.degree()+1,2,poly.degree()+1])
    x0 = 0
    x1 = bits2range(1<<n,fulldomain,nqbits)-fulldomain[0]
    for i in range(poly.degree()+1):
        for j in range(0,i+1):
            bin = binom(i,i-j)
            out[i,0,j] = bin*x0**(i-j)
            out[i,1,j] = bin*x1**(i-j)
    return out

def polynomial2MPS(poly:Poly, nqbits:int,polydomain:tuple[int,int], fulldomain:tuple[float,float],dtype = torch.float64):
    """
    takes the polynomial defined on the polydomain and turn it into a qbits MPS with nqbits that exist on the fulldomain.
    polydomain must be given as two integers, those integer should convert to the support of the polynomial when given to bits2range with fulldomain for domain.
    """
    #tensor at index 0 is most significant qbit.
    torch.set_default_dtype(dtype)
    pd0 = polydomain[0]
    pd1 = polydomain[1]
    Ntensor_support = int(np.ceil(np.log2(pd0^pd1)))
    out = qtt.networks.MPS(nqbits)
    trivial_bits = polydomain[0]>>(Ntensor_support)
    for i in range(-Ntensor_support-1,-len(out)-1,-1):
        out[i] = torch.zeros([1,2,1])
        bit = trivial_bits & 1
        trivial_bits>>1
        out[i][0,bit,0] = 1
    out[-Ntensor_support] = torch.Tensor(1,2,poly.degree()+1)
    for i in range(poly.degree()+1):
        T = out[-Ntensor_support]
        x = 0
        T[0,0,i] = Phi_s(poly,i,x)
        x = bits2range((1<<(Ntensor_support-1)),fulldomain,nqbits)-fulldomain[0]
        T[0,1,i] = Phi_s(poly,i,x)
    for n in range(1,Ntensor_support-1):
        out[-n-1] = G(poly,pd0,fulldomain,n,nqbits)
    out[-1] = torch.zeros(poly.degree()+1,2,1)
    out[-1][:,1,0] = torch.tensor([bits2range(pd0+1,fulldomain,nqbits)**p for p in range(poly.degree()+1)])
    out[-1][:,0,0] = torch.tensor([bits2range(pd0,fulldomain,nqbits)**p for p in range(poly.degree()+1)])
    return out
    
    
if __name__=="__main__":
    def y(x):
        return np.exp(-x**2)
    poly1 = interpolate(y,4,[-1,0])
    poly2 = interpolate(y,4,[0,1])
    mps1 = polynomial2MPS(poly1,4,[0b000,0b111],[-1,1])
    mps2 = polynomial2MPS(poly2,4,[0b1000,0b1111],[-1,1])
    bitsdomain = bits2range(np.array(range(16)),[-1,1],4)
    test_samples = y(bitsdomain)
    mps1_samples = np.array([ torch.matmul(mps1[0][:,(i>>3)&1,:],torch.matmul(torch.matmul(mps1[1][:,(i>>2)&1,:],mps1[2][:,(i>>1)&1,:]),mps1[3][:,i&1,:])).item() for i in range(0,16)])
    mps2_samples = np.array([ torch.matmul(mps2[0][:,(i>>3)&1,:],torch.matmul(torch.matmul(mps2[1][:,(i>>2)&1,:],mps2[2][:,(i>>1)&1,:]),mps2[3][:,i&1,:])).item() for i in range(0,16)])
    print(bitsdomain)
    print(test_samples)
    print(mps1_samples)
    print(mps2_samples)
    cMPS = compress_algs.MPS_compressing_sum([mps1,mps2],1e-6,1e-6)
    for t in cMPS:
        print(t.size())
    oc = cMPS.orthogonality_center
    print("tensor norm: ", torch.tensordot(cMPS[oc],cMPS[oc],dims=([0,1,2],[0,1,2])).item())
    print("overlap : ",qtt.networks.contract(cMPS,mps1).item()+qtt.networks.contract(cMPS,mps2).item() )
    X = np.linspace(-1,1,1000)
    print("expected norm: ", np.sum(y(X))*(X[1]-X[0]))
    print("expected norm from poly MPS: ", np.sum((mps1_samples+mps2_samples)**2 )*2/(2**4-1))
    cMPS_samples = np.array([ torch.matmul(cMPS[0][:,(i>>3)&1,:],torch.matmul(torch.matmul(cMPS[1][:,(i>>2)&1,:],cMPS[2][:,(i>>1)&1,:]),cMPS[3][:,i&1,:])).item() for i in range(0,16)])
    print(cMPS_samples)
    print("actual norm polyMPS",np.sum(cMPS_samples**2)*2/(2**4-1))


